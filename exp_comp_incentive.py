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
incentive_record = np.zeros(run)
output_arm_record_incenticve = -np.ones(run)
output_arm_record_no_incenticve = -np.ones(run)

means = np.array(([[0.89, 0.47, 0.01],
                   [0.01, 0.47, 0.89]]))
global_mean = np.mean(means, axis=0)
sorted_mean = np.sort(global_mean, -1)[::-1]
gap = sorted_mean[0] - sorted_mean[1]
index = np.argmax(np.mean(means,axis=0))
print("The global mean is {}, with gap {}".format(global_mean, gap))

for i in range(run):
    mmab = MMABNoIncentive(means, nplayers, narms, horizon, strategy)
    best_arm = mmab.simulate()
    tmp = global_mean[best_arm]
    output_arm_record_no_incenticve[i] = best_arm + 1
    print('No incentive', index == best_arm, " In {}th run, the best arm is identified as {}".format(i+1, best_arm+1))

for i in range(run):
    mmab = MMABIncentive(means, nplayers, narms, horizon, delta, strategy)
    best_arm = mmab.simulate()
    tmp = global_mean[best_arm]
    output_arm_record_incenticve[i] = best_arm + 1
    incentive_record[i] = mmab.incentive_cost_record[mmab.t - 1]
    print('Incentive', index == best_arm, " In {}th run, the best arm is identified as {}".format(i+1, best_arm+1))


print('incentive_record',incentive_record)
print('output_arm_record_no_incenticve', output_arm_record_no_incenticve)
print('output_arm_record_incenticve', output_arm_record_incenticve)

s1 = 0
s2 = 0
for i in range(run):
    s1 += (output_arm_record_no_incenticve[i] == index+1)
    s2 += (output_arm_record_incenticve[i] == index+1)


print('success_rate_no_incentive', s1/run)
print('success_rate_incentive', s2/run)
print('avg_incentive',np.mean(incentive_record))
