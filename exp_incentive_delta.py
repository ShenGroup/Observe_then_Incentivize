from environment.stochastic_helper import gen_instances
from environment.bandit_environment import *
from environment.MABstrategy import *
import numpy as np

np.random.seed(5237)
narms = 3
nplayers = 2
horizon = 100000

strategy = UCB1
run = 5

means = np.array(([[0.89, 0.47, 0.01],
                   [0.01, 0.47, 0.89]]))

global_mean = np.mean(means, axis=0)
sorted_mean = np.sort(global_mean, -1)[::-1]
gap = sorted_mean[0] - sorted_mean[1]
index = np.argmax(np.mean(means,axis=0))
print("The global mean is {}, with gap {}".format(global_mean, gap))

delta_list = 1/np.exp(np.arange(0,21,2))


incentive_record = -np.ones((len(delta_list),run))
output_arm_record = -np.ones((len(delta_list),run))

for k in range(len(delta_list)):
    delta = delta_list[k]
    for i in range(run):
        mmab = MMABIncentive(means, nplayers, narms, horizon, delta, strategy)
        best_arm = mmab.simulate()
        output_arm_record[k][i] = best_arm + 1
        if best_arm == index:
            incentive_record[k][i] = mmab.incentive_cost_record[mmab.t - 1]
        #print(index == best_arm, "Incentive, delta {}, in {}th run, the best arm is identified as {} with incentive {}".format(delta, i+1, best_arm+1, mmab.incentive_cost_record[mmab.t - 1]))
    print("delta {}, avg incentive {}".format(delta, np.mean(incentive_record[k])))


print('avg_incentive',np.mean(incentive_record, axis = 1))
