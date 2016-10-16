import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
from collections import namedtuple

State = namedtuple('State', ['waypoint', 'light', 'oncoming', 'left', 'right'])

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env, alpha=0.5, epsilon=0.05, gamma=0.9):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        self.q = {}
        self.actions = self.env.valid_actions

        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self.t = 1.

    def getQ(self, state, action):
      return self.q.get((state, action), 0)

    def learnQ(self, state, action, reward, value):
      alpha = self.alpha / self.t
      oldv = self.q.get((state, action), None)
      if oldv is None:
        self.q[(state, action)] = reward
      else:
        self.q[(state, action)] = (1 - alpha) * oldv + alpha * value
      self.t += 1.
      return self.q[(state, action)]

    def learn(self, state1, action1, reward, state2):
      q_sprime = [self.getQ(state2, a) for a in self.actions]
      maxqnew = max(q_sprime)
      new_q = self.learnQ(state1, action1, reward, reward + self.gamma * maxqnew)
      return new_q

    def choose_action(self, state):
      if random.random() < self.epsilon:
        action = random.choice(self.actions)
      else:
        q = [self.getQ(state, a) for a in self.actions]
        maxQ = max(q)
        best_indices = [i for i in range(len(self.actions)) if q[i] == maxQ]
        best_actions = [self.actions[i] for i in best_indices]
        action = random.choice(best_actions)
      return action

    def best_qs(self, n=5):
      q_vals = self.q.values()
      indexed_q_vals = zip(range(len(q_vals)), q_vals)
      best_q_vals = sorted(indexed_q_vals, key=lambda (i,v): v, reverse=True)
      top_keys = [self.q.keys()[i] for i,v in best_q_vals]
      return top_keys[:n]

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        self.t = 1.

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # TODO: Update state
        self.state = State(
            waypoint = self.next_waypoint,
            light = inputs['light'],
            oncoming = inputs['oncoming'],
            left = inputs['left'],
            right = inputs['right']
            )

        # TODO: Select action according to your policy
        action = self.choose_action(self.state)

        # Execute action and get reward
        reward = self.env.act(self, action)

        # TODO: Learn policy based on state, action, reward
        new_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        new_inputs = self.env.sense(self)
        new_state = State(
            waypoint = new_waypoint,
            light = new_inputs['light'],
            oncoming = new_inputs['oncoming'],
            left = new_inputs['left'],
            right = new_inputs['right']
            )

        new_q = self.learn(self.state, action, reward, new_state)

        # print "LearningAgent.update(): deadline = {}, waypt = {}, light = {}, act = {}, reward = {}, new_q = {}".format(deadline, self.next_waypoint, inputs['light'], action, reward, new_q)  # [debug]


def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent, alpha=1, epsilon=0.05, gamma=0.5)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=0.0, display=False)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=100)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line
    return sim


if __name__ == '__main__':
    a = run()
