# -*- coding: utf-8 -*-
"""
Created on Sat May  4 16:48:33 2024

@author: Gerard Joling
"""

import numpy as np
import matplotlib.pyplot as plt
import datetime
#import qutip_qip
import random as random
from qutip import *
#from qutip_qip import *
from qutip.core.gates import cnot, hadamard_transform

#import qutip.logging_utils as logging
#logger = logging.get_logger()
#Set this to None or logging.WARN for 'quiet' execution
#log_level = logging.INFO
example_name = 'Lindblad'
#Set this to None or logging.WARN for 'quiet' execution
#QuTiP control modules
import pulseoptim as cpo
example_name = 'Lindblad'


random.seed(20)
alpha = [random.random(),random.random()]
beta  = [random.random(),random.random()]

Sx = sigmax()
Sz = sigmaz()

H_d = (alpha[0]*tensor(Sx,identity(2)) + 
      alpha[1]*tensor(identity(2),Sx) +
      beta[0]*tensor(Sz,identity(2)) +
      beta[1]*tensor(identity(2),Sz))
H_c = [tensor(Sz,Sz)]
# Number of ctrls
n_ctrls = len(H_c)

q1_0 = q2_0 = Qobj([[1], [0]])
q1_targ = q2_targ = Qobj([[0], [1]])

psi_0 = tensor(q1_0, q2_0)
psi_targ = tensor(q1_targ, q2_targ)


# Number of time slots
n_ts = 100
# Time allowed for the evolution
evo_time = 18

refevolution = mesolve(H_d, psi_0, tlist=np.linspace(0,evo_time,n_ts))

# Fidelity error target
fid_err_targ = 1e-3
# Maximum iterations for the optisation algorithm
max_iter = 500
# Maximum (elapsed) time allowed in seconds
max_wall_time = 120

# pulse type alternatives: RND|ZERO|LIN|SINE|SQUARE|SAW|TRIANGLE|
p_type = 'DEF'

#Set to None to suppress output files

Sx = sigmax()
Sy = sigmay()
Sz = sigmaz()
Sm = sigmam()
Si = identity(2)
#Hadamard gate
had_gate = hadamard_transform(1)

# Hamiltonian
Del = 0.1    # Tunnelling term
wq = 1.0   # Energy of the 2-level system.
H0 = 0.5*wq*sigmaz() + 0.5*Del*sigmax()

#Amplitude damping#
#Damping rate:
gamma = 0.1
L0 = liouvillian(H0, [np.sqrt(gamma)*Sm])

#sigma X control
LC_x = liouvillian(Sx)
#sigma Y control
LC_y = liouvillian(Sy)
#sigma Z control
LC_z = liouvillian(Sz)

#Drift
drift = L0
#Controls - different combinations can be tried
ctrls = [LC_z, LC_x]
# Number of ctrls
n_ctrls = len(ctrls)

# start point for the map evolution
E0 = sprepost(Si, Si)

# target for map evolution
E_targ = sprepost(had_gate, had_gate)


vac = basis(2, 0)
a = destroy(2)

H_sys = a.dag()*a
H_con = [liouvillian(a + a.dag()), liouvillian(a*a.dag())]

T = 1
steps = 10
times = np.linspace(0,T,steps)

c1 = -0.6
c2 = -1

L0 = liouvillian(H_sys, c_ops=[c1*a, c2*a*a.dag()])

rho0 = operator_to_vector(Qobj([[0.8,0],[0,0.2]]))
print(rho0)
newr = vector_to_operator(rho0)
rhotar = operator_to_vector(Qobj([[0.5,0], [0, 0.5]]))

refevolution = mesolve(L0, newr, tlist=times).states
zerolist = []
onelist = []
twolist = []
threelist = []
print(refevolution[-1])
print(refevolution[0])
for state in refevolution:
    #print(state.tr())
    zerolist.append(abs(state.full()[0][0]))
    onelist.append(abs(state.full()[1][1]))
    #twolist.append(abs(state[2][2]))
    #threelist.append(abs(state[3][3]))

plt.plot(times, zerolist, label='0list')
plt.plot(times, onelist, label='1list')
#plt.plot(times, twolist, label='2list')
#plt.plot(times, threelist, label='3list')
plt.legend()
plt.show()

# Number of time slots
n_ts = 10
# Time allowed for the evolution
evo_time = 1

# Fidelity error target
fid_err_targ = 1e-5
# Maximum iterations for the optisation algorithm
max_iter = 200
# Maximum (elapsed) time allowed in seconds
max_wall_time = 30
# Minimum gradient (sum of gradients squared)
# as this tends to 0 -> local minima has been found
min_grad = 1e-20

p_type = 'RND'                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            

f_ext = "{}_n_ts{}_ptype{}.txt".format(example_name, n_ts, p_type)

result = cpo.optimize_pulse(L0, H_con, rho0, rhotar, refevolution, n_ts, evo_time, 
                fid_err_targ=fid_err_targ, min_grad=min_grad, 
                max_iter=max_iter, max_wall_time=max_wall_time, 
                out_file_ext=f_ext, init_pulse_type=p_type
                , gen_stats=True)


result.stats.report()
print("Final evolution\n{}\n".format(vector_to_operator(result.evo_full_final)))
print("********* Summary *****************")
print("Initial fidelity error {}".format(result.initial_fid_err))
print("Final fidelity error {}".format(result.fid_err))
print("Final gradient normal {}".format(result.grad_norm_final))
print("Terminated due to {}".format(result.termination_reason))
print("Number of iterations {}".format(result.num_iter))
print("Completed in {} HH:MM:SS.US".format(datetime.timedelta(seconds=result.wall_time)))
print(result.lam)

newevolution = mesolve([L0, [H_con[0], result.final_amps[:, 0]], [H_con[1], result.final_amps[:, 1]]], vector_to_operator(rho0), times)
zerolist = []
onelist = []
twolist = []
threelist = []
for state in newevolution.states:
    print(state)
    #print(state.tr())
    zerolist.append(abs(state.full()[0][0]))
    onelist.append(abs(state.full()[1][1]))
    #twolist.append(abs(state[2][2]))
    #threelist.append(abs(state[3][3]))

plt.plot(times, zerolist, label='0list')
plt.plot(times, onelist, label='1list')
#plt.plot(times, twolist, label='2list')
#plt.plot(times, threelist, label='3list')
plt.legend()
plt.show()


fig1 = plt.figure()
ax1 = fig1.add_subplot(2, 1, 1)
ax1.set_title("Initial control amps")
ax1.set_xlabel("Time")
ax1.set_ylabel("Control amplitude")
for j in range(n_ctrls):
    ax1.step(result.time, 
             np.hstack((result.initial_amps[:, j], result.initial_amps[-1, j])), 
             where='post')

ax2 = fig1.add_subplot(2, 1, 2)
ax2.set_title("Optimised Control Sequences")
ax2.set_xlabel("Time")
ax2.set_ylabel("Control amplitude")
for j in range(n_ctrls):
    ax2.step(result.time, 
             np.hstack((result.final_amps[:, j], result.final_amps[-1, j])), 
             where='post')
fig1.tight_layout()