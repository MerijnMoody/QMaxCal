import numpy as np
from qutip import *
from lindblad import *

vac = basis(2, 0)
a = destroy(2)
con = sigmax()

H_sys = a.dag()*a
H_con = [liouvillian(con)]
#liouvillian(a*a.dag())

# Number of time slots
n_ts = 30
# Time allowed for the evolution 
evo_time = 0.5

times = np.linspace(0,evo_time,n_ts)
c1 = -0.6
c2 = -1

L0 = liouvillian(H_sys, c_ops=[c1*a, c2*a*a.dag()])

rho0 = operator_to_vector(Qobj([[0.8,0],[0,0.2]]))
newr = vector_to_operator(rho0)
rhotar = operator_to_vector(Qobj([[0.3,0], [0,0.7]]))

othertimes = np.linspace(0,evo_time,n_ts)
othertimes = othertimes.tolist()
othertimes.append(times[-1] + (times[1] - times[0]))
ref_evo = mesolve(L0, newr, tlist=othertimes).states
zerolist = []
onelist = []

for state in ref_evo:
    zerolist.append(abs(state.full()[0][0]))
    onelist.append(abs(state.full()[1][1]))

#plt.plot(times, zerolist, label='0list')
#plt.plot(times, onelist, label='1list')
#plt.legend()

time_list = np.linspace(0, evo_time, n_ts)

evolution = create_evolution(L0, H_con, rho0, rhotar, ref_evo, time_list, 2)
evolution.ascent(1000, 0.05, 0, 0)
print("lam:")
print(evolution.lam)
print("ent_err")
print(evolution.ent_error_list)
print("fid_err")
print(evolution.fid_err_list)
print("ctrls")
print(evolution.ctrls)
print("ent_grad")
print(evolution.ent_grad)
print("fid_grad")
print(evolution.fid_grad)
print("fid_grad_approx")
print(evolution.fid_grad_approx)
print("ent_grad_approx")
print(evolution.ent_grad_approx)
print("fwd_evo")
print(evolution.fwd_evo)
times = times.tolist()
times.append(times[-1] + (times[1] - times[0]))
zerolist = []
onelist = []
for state in evolution.fwd_evo:
    #print(state)
    zerolist.append(abs(vector_to_operator(state).full()[0][0]))
    onelist.append(abs(vector_to_operator(state).full()[1][1]))

#plt.plot(times, zerolist, label='0list')
#plt.plot(times, onelist, label='1list')

norm = len(evolution.evo_list)
for i in range(norm):
    evo = evolution.evo_list[i]
    zerolist = []
    onelist = []
    for state in evo:
        zerolist.append(abs(vector_to_operator(state).full()[0][0]))
        onelist.append(abs(vector_to_operator(state).full()[1][1]))
    plt.plot(times, zerolist, alpha = (0.5*(i+1)/norm), color = 'red')
    plt.plot(times, onelist, alpha = (0.5*(i+1)/norm), color = 'blue')
    
plt.legend()
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