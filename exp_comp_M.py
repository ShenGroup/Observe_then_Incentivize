from environment.stochastic_helper import gen_instances
from environment.bandit_environment import *
from environment.MABstrategy import *
import numpy as np

np.random.seed(5237)
narms = 30
M = np.arange(3,13)*10
horizon = 100000
delta = 0.01
strategy = UCB1
run = 100


global_gap = 0.005
mu = np.arange(narms)*global_gap+0.4
incentive_record = -np.ones((len(M), run))
output_arm_record = -np.ones((len(M), run))

threshold_lower = 0.9*global_gap
threshold_upper = 1.1*global_gap

for k in range(len(M)):
    nplayers = M[k]
    for i in range(run):
        means = gen_instances(nplayers, narms, mu=mu, sigma=global_gap)

        global_mean = np.mean(means, axis=0)
        sorted_mean = np.sort(global_mean, -1)[::-1]
        gap = sorted_mean[0] - sorted_mean[1]
        index = np.argmax(np.mean(means,axis=0))
        #while gap < threshold_lower or gap > threshold_upper or index != narms-1:
        while gap < threshold_lower or gap > threshold_upper:
            means = gen_instances(nplayers, narms, mu=mu, sigma = global_gap)
            global_mean = np.mean(means, axis=0)
            sorted_mean = np.sort(global_mean, -1)[::-1]
            gap = sorted_mean[0] - sorted_mean[1]
            index = np.argmax(np.mean(means,axis=0))

        # print("In {}th run with {} players, the global mean is {}, with gap {}".format(i+1, nplayers, global_mean, gap))
        mmab = MMABIncentive(means, nplayers, narms, horizon, delta, strategy)
        best_arm = mmab.simulate()
        tmp = global_mean[best_arm]
        print(index == best_arm, " In {}th run with {} players,  with gap {}, the best arm is identified as {}, with incentive {}".format(i+1, nplayers, gap, best_arm, mmab.incentive_cost_record[mmab.t-1]))

        if best_arm == index:
            incentive_record[k][i] = mmab.incentive_cost_record[mmab.t - 1]
            output_arm_record[k][i] = best_arm

print(np.mean(incentive_record, axis=1))
print(output_arm_record)