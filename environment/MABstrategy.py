import numpy as np


class CLIENT(object):
    def __init__(self, narms, horizon, strategy, **kwargs):
        self.K = narms
        self.T = horizon
        self.strategy = strategy
        self.agent = self.strategy(narms, horizon)

    def select_arm(self):
        return self.agent.select_arm()

    def update(self, play, reward):
        self.agent.update(play, reward)


class UCB1():
    def __init__(self, narms, horizon):
        self.K = narms
        self.T = horizon
        self.npulls = np.zeros(self.K, dtype=np.int32)
        self.sample_mean = np.zeros(self.K)
        self.t = 0

    def select_arm(self):
        """
        If an arm has never been selected, its UCB is viewed as infinity
        """
        for arm in range(self.K):
            if self.npulls[arm] == 0:
                return arm
        ucb = [0.0 for arm in range(self.K)]
        for arm in range(self.K):
            radius = np.sqrt((2 * np.log(self.t)) / float(self.npulls[arm]))
            ucb[arm] = self.sample_mean[arm] + radius
        return np.argmax(ucb)

    def update(self, chosen_arm, reward):
        self.npulls[chosen_arm] += 1
        n = self.npulls[chosen_arm]
        value = self.sample_mean[chosen_arm]
        new_value = ((n - 1) / float(n)) * value + (1 / float(n)) * reward
        self.sample_mean[chosen_arm] = new_value
        self.t += 1


class AnnealingEpsilonGreedy():
    def __init__(self, narms, horizon):
        self.c = c
        self.d = d
        self.K = narms
        self.T = horizon
        self.npulls = np.zeros(self.K, dtype=np.int32)
        self.sample_mean = np.zeros(self.K)
        self.t = 0

    def select_arm(self):
        t = self.t + 1
        epsilon = 1 / t
        if np.random.random() > epsilon:
            return np.argmax(self.sample_mean)
        else:
            return np.random.randint(self.K)

    def update(self, chosen_arm, reward):
        self.npulls[chosen_arm] += 1
        n = self.npulls[chosen_arm]
        value = self.sample_mean[chosen_arm]
        new_value = ((n - 1) / float(n)) * value + (1 / float(n)) * reward
        self.sample_mean[chosen_arm] = new_value
        self.t += 1

