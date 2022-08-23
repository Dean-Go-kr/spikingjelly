import torch
import torch.nn as nn
import numpy as np
from spikingjelly.activation_based import neuron
from spikingjelly import visualizing
from matplotlib import pyplot as plt


##########################
####### SpkingJelly ######
##########################

cuba = neuron.CubaLIFNode()
x = torch.rand(size=[3,5,1])

T = 1
s_list = []
v_list = []
i_list = []
for t in range(T):
    s_list.append(cuba(x).unsqueeze(0))
    v_list.append(cuba.v.unsqueeze(0))
    i_list.append(cuba.i.unsqueeze(0))

s_list = torch.cat(s_list)
v_list = torch.cat(v_list)
i_list = torch.cat(i_list)


##########################
########### LAVA #########
##########################

from lava.lib.dl.slayer.neuron import cuba

threshold = 1
current_decay = 0 #np.random.random()
voltage_decay = 0 #np.random.random()

neuron = cuba.Neuron(
    threshold, current_decay, voltage_decay, persistent_state=True
)
neuron.debug = True

current, voltage = neuron.dynamics(x)
spike = neuron.spike(voltage)



print('############## Current ##############')
print('=== SpikingJelly ===')
print(i_list)
print('=== LAVA ===')
print(current)
print('############################')

print('\n')
print('\n')

print('******* Voltage *******')
print('=== SpikingJelly ===')
print(v_list)
print('=== LAVA ===')
print(voltage)
print('############################')

print('\n')
print('\n')

print('******* spike *******')
print('=== SpikingJelly ===')
print(s_list)
print('=== LAVA ===')
print(spike)
print('############################')
