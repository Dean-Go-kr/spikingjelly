import torch
import torch.nn as nn
import numpy as np
from spikingjelly.activation_based import neuron
from spikingjelly import visualizing
from matplotlib import pyplot as plt
from spikingjelly.activation_based import lava_exchange


def max_error(x, y):
    return (x - y).abs().max()

threshold = 1.
current_decay = 0.1 #np.random.random()
voltage_decay = 0.2 #np.random.random()

##########################
####### SpkingJelly ######
##########################

T = 20
N = 1
C = 1


x_seq  = torch.rand(size=[T, N, C])

if T > 1:
    cuba = neuron.CubaLIFNode(v_threshold=threshold, voltage_decay=voltage_decay, current_decay=current_decay, step_mode='m')
else:
    cuba = neuron.CubaLIFNode(v_threshold=threshold, voltage_decay=voltage_decay, current_decay=current_decay)



s_list = []
v_list = []
i_list = []
# for t in range(T):
#     x = x_seq[t]
#     s_list.append(cuba(x).unsqueeze(0))
#     v_list.append(cuba.v.unsqueeze(0))
#     i_list.append(cuba.i.unsqueeze(0))

s_list.append(cuba(x_seq).unsqueeze(0))
v_list.append(cuba.v.unsqueeze(0))
i_list.append(cuba.i.unsqueeze(0))

s_list = torch.cat(s_list)
v_list = torch.cat(v_list)
i_list = torch.cat(i_list)


##########################
########### LAVA #########
##########################

from lava.lib.dl.slayer.neuron import cuba

lava_cuba = cuba.Neuron(
    threshold, current_decay, voltage_decay, persistent_state=True
)
neuron.debug = True

x_seq = lava_exchange.TNX_to_NXT(x_seq)
current, voltage = lava_cuba.dynamics(x_seq)
spike = lava_cuba.spike(voltage)

current = lava_exchange.NXT_to_TNX(current)
voltage = lava_exchange.NXT_to_TNX(voltage)
spike = lava_exchange.NXT_to_TNX(spike)

print(i_list)
sj_current = torch.reshape(i_list, (T, -1))
lava_current = torch.reshape(current, (T, -1))

sj_voltage = torch.reshape(v_list, (T, -1))
lava_voltage = torch.reshape(voltage, (T, -1))

sj_spike = torch.reshape(s_list, (T, -1))
lava_spike = torch.reshape(spike, (T, -1))

print('Left: SJ / Right: LAVA')

print('## Current ##')
for i in range(T):
    print(sj_current[i], lava_current[i])


print('## Voltage ##')
for i in range(T):
    print(sj_voltage[i], lava_voltage[i])


print('## Spike ##')
for i in range(T):
    print(sj_spike[i], lava_spike[i])


# print(i_list)
# print(v_list)
# print(s_list)
print(max_error(current, i_list))
print(max_error(voltage, v_list))
print(max_error(spike, s_list))
print(s_list.sum())

# print('############## Current ##############')
# print('=== SpikingJelly ===')
# print(i_list)
# print('=== LAVA ===')
# print(current)
# print('############################')

# print('\n')
# print('\n')

# print('******* Voltage *******')
# print('=== SpikingJelly ===')
# print(v_list)
# print('=== LAVA ===')
# print(voltage)
# print('############################')

# print('\n')
# print('\n')

# print('******* spike *******')
# print('=== SpikingJelly ===')
# print(s_list)
# print('=== LAVA ===')
# print(spike)
# print('############################')
