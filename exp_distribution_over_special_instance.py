from environment.stochastic_helper import gen_instances
from environment.bandit_environment import *
from environment.MABstrategy import *
import numpy as np

np.random.seed(5237)
narms = 3
nplayers = 2
horizon = 100000
delta = 0.05
strategy = UCB1
run = 100
incentive_record = -np.ones(run)
output_arm_record_incenticve = -np.ones(run)
incentive_matrix_record = np.zeros((run, nplayers, narms))

means = np.array(([[0.89, 0.47, 0.01],
                   [0.01, 0.47, 0.89]]))
global_mean = np.mean(means, axis=0)
sorted_mean = np.sort(global_mean, -1)[::-1]
gap = sorted_mean[0] - sorted_mean[1]
index = np.argmax(np.mean(means,axis=0))
print("The global mean is {}, with gap {}".format(global_mean, gap))

count = 0

for i in range(run):
    mmab = MMABIncentive(means, nplayers, narms, horizon, delta, strategy)
    best_arm = mmab.simulate()
    tmp = global_mean[best_arm]
    for j in range(narms):
        if tmp == sorted_mean[j]:
            output_arm_record_incenticve[i] = j + 1
            break
    if best_arm == index:
        count += 1
        incentive_matrix_record[i] = mmab.incentive_cost_matrix
        incentive_record[i] = mmab.incentive_cost_record[mmab.t - 1]

    print(index == best_arm, " In {}th run, the best arm is identified as {}".format(i+1, best_arm))

avg_incentive_matrix = np.sum(incentive_matrix_record, axis=0) / count
avg_incentive = np.sum(incentive_record, axis=0) / count
print('avg_incentive_matrix', avg_incentive_matrix)
print('avg_incentive', avg_incentive)
print('output_arm_record',output_arm_record_incenticve)
