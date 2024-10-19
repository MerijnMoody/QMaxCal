import matplotlib.pyplot as plt 
import numpy as np
from qutip import *
from TorchIt import *

def _num_vec_to_op(rho, dim):
    #print(rho)
    return np.reshape(rho, (dim, dim))

glob_dim = 2
vac = basis(glob_dim, 0)
a = destroy(glob_dim)
#con = a + a.dag()
#con1 = Qobj([[0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0]])
#con2 = Qobj([[0, 0, 0, 0], [0, 0.5, 0, 0], [0, 0, 1, 0], [0,0,0, 1.5]])
#con3 = Qobj([[0, 0, 2 + 1j], [0, 0, 0], [2-1j, 0, 0]])

con2 = Qobj([[0,0], [0, 1]])

#con3a = Qobj([[0, 0, 1], [0, 0, 0], [0, 0, 0]])
#con3b = Qobj([[0, 0, 0], [0, 0, 0], [1, 0, 0]])
#print(con1)
#print(con2)
#print(con3)
H_sys = a.dag()*a

#print(H_sys)
#print(a + a.dag())
#H_con = [liouvillian(a + a.dag()), liouvillian(H_sys), liouvillian(con2), liouvillian(con3)]
#H_con = [[liouvillian(a), liouvillian(a.dag())], [liouvillian(a*a), liouvillian(a.dag()*a.dag())], [liouvillian(con2), liouvillian(con2)]]
#liouvillian(a*a.dag())
H_con = [[liouvillian(a), liouvillian(a.dag())], [liouvillian(H_sys), liouvillian(H_sys.dag())]]

# Number of time slots
n_ts = 40
# Time allowed for the evolution 
evo_time = 1

times = np.linspace(0,evo_time,n_ts)
c1 = -2
c2 = -3
#c2 = 0

L0 = liouvillian(H_sys, c_ops=[c1*a, c2*a*a.dag()])

rho0 = operator_to_vector(Qobj([[0.8,0],[0,0.2]]))
#rho0 = operator_to_vector(Qobj([[0.9,0,0],[0,0.1,0], [0,0,0]]))
#rho0 = operator_to_vector(Qobj([[0.9,0,0,0],[0,0.05,0,0],[0,0,0.03,0],[0,0,0,0.02]]))
newr = vector_to_operator(rho0)
rhotar = operator_to_vector(Qobj([[0.1,0],[0,0.9]]))
#rhotar = operator_to_vector(Qobj([[0.5,0,0.5], [0,0,0], [0.5,0,0.5]]))
#rhotar = operator_to_vector(Qobj([[0.4,0,0,0.4], [0,0,0,0], [0,0,0.2,0], [0.4,0,0,0.4]]))


othertimes = np.linspace(0,evo_time,n_ts)
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

#plt.legend()
#input()

time_list = np.linspace(0, evo_time, n_ts)[:-1]

evolution = create_evolution(L0, H_con, rho0, rhotar, ref_evo, time_list, glob_dim)
evolution.optimize(2000, 0.001, 0, 0)
print("lam:")
print(evolution.lam)
print("ent_err")
print(evolution.ent_error_list)
print("fid_err")
print(evolution.fid_err_list)
print("ctrls")
print(evolution.ctrls_list[-2])
print(evolution.ctrls_real)
print(evolution.ctrls_im)
print("ent_grad")
print(evolution.ent_grad)
print("fid_grad")
#print(evolution.fid_grad)
print("fid_grad_approx")
print(evolution.fid_grad_approx)
print("ent_grad_approx")
print(evolution.ent_grad_approx)
print("fwd_evo")
print(_num_vec_to_op(evolution.fwd_evo[-1].detach(), glob_dim))
print("fid")
print("START")
print(evolution.fid)
print(evolution.fid_grad[0])
print(evolution.fid_grad[1])
print("END")
print(evolution.fid_grad[-2])
print(evolution.fid_grad[-1])
print("Jump")
print(evolution.jump)
times = times.tolist()
times.append(times[-1] + (times[1] - times[0]))
zerolist = []
onelist = []

new_fwd_evo = [_num_vec_to_op(state.detach().numpy(), glob_dim) for state in evolution.fwd_evo]

for state in new_fwd_evo:
    #print(state)
    zerolist.append(state[0][0])
    onelist.append(state[1][1])

#plt.plot(times, zerolist, label='0list')
#plt.plot(times, onelist, label='1list')
time_list = np.linspace(0, evo_time, n_ts)

evolution_evo_list = evolution.evo_list
#print(evolution_evo_list)
with open('test.npy', 'wb') as f:
    np.save(f, evolution_evo_list)
    
norm = len(evolution_evo_list)
for i in range(norm):
    evo = evolution.evo_list[i]
    new_evo = [_num_vec_to_op(state, glob_dim) for state in evo]
    zerolist = []
    onelist = []
    twolist = []
    threelist = []
    for state in new_evo:
        #print(state)
        zerolist.append(abs(state[0][0]))
        onelist.append(abs(state[1][1]))
    
    if i == norm-1:
        plt.plot(time_list, zerolist, color = 'red', label = '$|0>$')
        plt.plot(time_list, onelist, color = 'blue', label = '$|1>$')
    else:
        plt.plot(time_list, zerolist, alpha = (0.7*(i+1)/norm), color = 'red')
        plt.plot(time_list, onelist, alpha = (0.7*(i+1)/norm), color = 'blue')
    #plt.plot(times, twolist, alpha = (0.5*(i+1)/norm), color = 'green')
    #plt.plot(times, threelist, alpha = (0.5*(i+1)/norm), color = 'orange')
    
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

#res = evolution.optimize()
print(res)
args = res.x
print(res.fun)
print(evolution.grads)
cons = args[:-1]
print(cons)
print(args[-1])
#print(evolution.fid_err_list)
#print(evolution.fwd_evo)