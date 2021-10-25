import numpy as np
import random
from .MABstrategy import CLIENT


class MMABIncentive(object):
    def __init__(self, means, nplayers, narms, horizon, delta, strategy, **kwargs):
        self.K = narms
        self.M = nplayers
        self.means = np.array(means)
        self.T = horizon
        self.delta = delta
        self.strategy = strategy
        self.clients = [CLIENT(narms=self.K, horizon=self.T, strategy=self.strategy) for _ in range(self.M)]
        self.radius = np.zeros(self.K)
        self.sample_mean = np.zeros((self.M, self.K))
        self.global_sample_mean = np.zeros(self.K)
        self.npulls = np.zeros((self.M, self.K), dtype=np.int32)
        self.St = np.ones(self.K)
        self.fixed = -1
        self.t = 0
        self.incentive_cost_matrix = np.zeros((self.M, self.K))
        self.incentive_cost_record = np.zeros(self.T)
        self.INF = 8888

    def step(self):
        incentive = None

        if self.t < np.ceil(self.T/2) or self.fixed == 1:
            plays = [self.clients[i].select_arm() for i in range(self.M)]
            rewards = self.simulate_single_step(plays)
        else:
            plays = [self.clients[i].select_arm() for i in range(self.M)]

            kt = np.argmax(self.radius)
            mt = np.argmin(self.npulls[:, kt])
            plays[mt] = kt
            rewards = self.simulate_single_step(plays)
            incentive = [mt, kt]
        is_terminal = self.update(plays, rewards, incentive)  # is only one arm left?
        return is_terminal

    def simulate_single_step(self, plays):
        rewards = np.zeros(self.M)
        for i in range(self.M):
            if random.random() > self.means[i, plays[i]]:
                rewards[i] = 0
            else:
                rewards[i] = 1
        return rewards

    def update(self, plays, rewards, incentive=None):
        # Both principal and clients update the sample mean and n_pulls.
        for i in range(self.M):
            self.clients[i].update(plays[i], rewards[i])
            self.npulls[i][plays[i]] += 1
            n = self.npulls[i][plays[i]]
            old_value = self.sample_mean[i][plays[i]]
            self.sample_mean[i][plays[i]] = (n-1) / n * old_value + 1 / n * rewards[i]

        if incentive is None and self.t > 0:
            self.incentive_cost_record[self.t] = self.incentive_cost_record[self.t-1]

        if incentive is not None:
            self.incentive_cost_matrix[incentive[0], incentive[1]] += 1
            assert self.t != 0
            self.incentive_cost_record[self.t] = self.incentive_cost_record[self.t-1] + 1
            # Compute the radius
            for i in range(self.K):
                if self.St[i] == 0:
                    continue

                tmp_radius = 0
                tmp_sample_mean = 0
                for j in range(self.M):
                    tmp_radius += 0.1 * (np.log(self.K * self.T / self.delta)) / self.npulls[j][i]
                    tmp_sample_mean += self.sample_mean[j][i]
                self.radius[i] = np.sqrt(tmp_radius) / self.M
                self.global_sample_mean[i] = tmp_sample_mean / self.M
            # Update St
            LCB = self.global_sample_mean - self.radius
            UCB = self.global_sample_mean + self.radius
            threshold = np.max(LCB)
            for i in range(self.K):
                if UCB[i] <= threshold:
                    self.St[i] = 0
                    self.global_sample_mean[i] = -self.INF
                    self.radius[i] = 0
            # If there is only one active arm

            if np.sum(self.St) == 1:
                self.fixed = 1
                return True
        self.t += 1
        return False

    def simulate(self):
        for i in range(self.T):
            is_terminal = self.step()
            if is_terminal:
                break
        # if |S(t)| == 1, return that arm; else: return the active arm with minimal index.
        if np.sum(self.St) == 1:
            return np.argmax(self.St)
        else:
            return np.argmax(self.global_sample_mean)

    def reset(self):
        pass


class MMABNoIncentive(object):
    def __init__(self, means, nplayers, narms, horizon, strategy, **kwargs):
        self.K = narms
        self.M = nplayers
        self.means = np.array(means)
        self.T = horizon
        self.strategy = strategy
        self.clients = [CLIENT(narms=self.K, horizon=self.T, strategy=self.strategy) for _ in range(self.M)]
        self.sample_mean = np.zeros((self.M, self.K))
        self.global_sample_mean = np.zeros(self.K)
        self.npulls = np.zeros((self.M, self.K), dtype=np.int32)
        self.t = 0

    def step(self):
        plays = [self.clients[i].select_arm() for i in range(self.M)]
        rewards = self.simulate_single_step(plays)
        self.update(plays, rewards)

    def simulate_single_step(self, plays):
        rewards = np.zeros(self.M)
        for i in range(self.M):
            if random.random() > self.means[i, plays[i]]:
                rewards[i] = 0
            else:
                rewards[i] = 1
        return rewards

    def update(self, plays, rewards):
        for i in range(self.M):
            self.clients[i].update(plays[i], rewards[i])
            self.npulls[i][plays[i]] += 1
            n = self.npulls[i][plays[i]]
            old_value = self.sample_mean[i][plays[i]]
            self.sample_mean[i][plays[i]] = (n-1) / n * old_value + 1 / n * rewards[i]

        self.t += 1

    def simulate(self):
        for i in range(self.T):
            self.step()
        for i in range(self.K):
            tmp_sample_mean = 0
            for j in range(self.M):
                tmp_sample_mean += self.sample_mean[j][i]
            self.global_sample_mean[i] = tmp_sample_mean / self.M
        return np.argmax(self.global_sample_mean)

    def reset(self):
        pass
