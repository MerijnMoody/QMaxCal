import matplotlib.pyplot as plt 
import numpy as np
from qutip import *
from TorchIt import *

def _num_vec_to_op(rho, dim):
    return np.reshape(rho, (dim, dim))

glob_dim = 2
vac = basis(glob_dim, 0)
a = destroy(glob_dim)

con2 = Qobj([[0,0], [0, 1]])

H_sys = a.dag()*a

H_con = [[liouvillian(a), liouvillian(a.dag())]]
Ham_list = [[0.5 * H_sys.full(), 0.5 * H_sys.full()], [a.full(), a.dag().full()]]

n_ts = 8
evo_time = 1

times = np.linspace(0,evo_time,n_ts)
c1 = -2
c2 = -3

L0 = liouvillian(H_sys, c_ops=[c1*a, c2*a*a.dag()])

rho0 = operator_to_vector(Qobj([[0.8,0],[0,0.2]]))
newr = vector_to_operator(rho0)
rhotar = operator_to_vector(Qobj([[0.1,0],[0,0.9]]))
measurements = [[to_super(Qobj([[1, 0],[0, 0]])).full(), to_super(Qobj([[0,0],[0,1]])).full()] ,[1, 2, 7]]

othertimes = np.linspace(0, evo_time, n_ts)
othertimes = othertimes.tolist()
othertimes.append(times[-1] + (times[1] - times[0]))
ref_evo = mesolve(L0, newr, tlist=othertimes).states
zerolist = []
onelist = []

for state in ref_evo:
    zerolist.append(abs(state.full()[0][0]))
    onelist.append(abs(state.full()[1][1]))

plt.plot(othertimes, zerolist, color='red',  linestyle='dashed', markersize=3)
plt.plot(othertimes, onelist, color='blue',  linestyle='dashed', markersize=3)

time_list = np.linspace(0, evo_time, n_ts)[:-1]

# ctrls = torch.load('databigrun.pth')['ctrls']
ctrls = None
evolution = create_evolution(L0, H_con, Ham_list, rho0, rhotar, ref_evo, time_list, glob_dim, ctrls, measurements)
ctrls = evolution.optimize(10000, 0.0000005, 0, 0)

print("lam:")
print(evolution.lam)
print(evolution.lam2)
print("ent_err")
print(evolution.ent_error_list)
print("fid_err")
print(evolution.fid_err_list)
print("ctrls")
#print(evolution.ctrls_list[-2])
print(evolution.ctrls_real)
print(evolution.ctrls_im)
print("ent_grad")
print(evolution.ent_grad)
print("fid_grad")
print("fwd_evo")
print(_num_vec_to_op(evolution.fwd_evo[-1].detach(), glob_dim))
print("fid")
print(evolution.fid)
#print("START")
#print(evolution.fid_grad[0])
#print(evolution.fid_grad[1])
print("END")
print(evolution.fid_grad[-2])
print(evolution.fid_grad[-1])
print("Energy")
print(evolution.energy_list)
times = times.tolist()
times.append(times[-1] + (times[1] - times[0]))
zerolist = []
onelist = []
new_fwd_evo = [_num_vec_to_op(state.detach().numpy(), glob_dim) for state in evolution.fwd_evo]




dictionary = {'ctrls' : ctrls,
              'ctrls_real': evolution.ctrls_real,
              'ctrls_im': evolution.ctrls_im,
              'energy_list':evolution.energy_list,
              'fid_err_list':evolution.fid_err_list,
              'gradients':evolution.ent_grad,
              'lamlist':[evolution.lam, evolution.lam2],
              'ent_err_list':evolution.ent_error_list}

torch.save(dictionary, 'databigrun.pth')

for state in new_fwd_evo:
    zerolist.append(state[0][0])
    onelist.append(state[1][1])

time_list = np.linspace(0, evo_time, n_ts)
evolution_evo_list = evolution.evo_list
    
norm = len(evolution_evo_list)
for i in range(norm):
    evo = evolution.evo_list[i]
    new_evo = [_num_vec_to_op(state, glob_dim) for state in evo]
    zerolist = []
    onelist = []
    for state in new_evo:
        zerolist.append(abs(state[0][0]))
        onelist.append(abs(state[1][1]))
    
    if i == norm-1:
        plt.plot(time_list, zerolist, color = 'red', label = '$|0>$')
        plt.plot(time_list, onelist, color = 'blue', label = '$|1>$')
    else:
        plt.plot(time_list, zerolist, alpha = (0.7*(i+1)/norm), color = 'red')
        plt.plot(time_list, onelist, alpha = (0.7*(i+1)/norm), color = 'blue')
        
plt.xlabel("Time")
plt.ylabel("Population")
plt.legend()

ax = plt.gca()
ax.set_xlim([0, 1.02])
ax.set_ylim([-0.05, 1.05])

ax2 = ax.twinx()
ax2.set_ylim(-0.05, 1.05)
ax2.set_yticks([0, 0.1, 0.9, 1])
ax2.yaxis.set_tick_params(labelright=True)

plt.show()