
import matplotlib.pyplot as plt 
import numpy as np
from qutip import *
from TorchIt2 import *

def _num_vec_to_op(rho, dim):
    return np.reshape(rho, (dim, dim))

glob_dim = 2
vac = basis(glob_dim, 0)
a = destroy(glob_dim)

con2 = Qobj([[0,0], [0, 1]])

H_sys = a.dag()*a

H_con = [[liouvillian(a), liouvillian(a.dag())]]
Ham_list = [[0.5 * H_sys.full(), 0.5 * H_sys.full()], [a.full(), a.dag().full()]]

n_ts = 45
evo_time = 1

times = np.linspace(0,evo_time,n_ts)
c1 = -2
c2 = -3
c3 = 0.2

L0 = liouvillian(H_sys, c_ops=[c1*a, c2*a*a.dag(), c3*(a+a.dag())])

rho0 = operator_to_vector(Qobj([[0.8,0],[0,0.2]]))
newr = vector_to_operator(rho0)
rhotar = operator_to_vector(Qobj([[0.1,0],[0,0.9]]))
measurements = [[to_super(Qobj([[1, 0],[0, 0]])).full(), to_super(Qobj([[0,0],[0,1]])).full()] ,[1, 15, 30]]

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

# = torch.load('databigrun.pth')['ctrls']
ctrls = None
evolution = create_evolution(L0, H_con, Ham_list, rho0, rhotar, ref_evo, time_list, glob_dim, ctrls, measurements,10)
ctrls = evolution.optimize(30000, 0.0005,0,0)
ctrls_real = ctrls[0]
ctrls_im = ctrls[1]

print("Parameters")
print(evolution.params_list)
print("lam:")
print(evolution.lam)
print(evolution.lam2)
print("ent_err")
print(evolution.ent_error_list)
print("fid_err")
print(evolution.fid_err_list)
print("ctrls")
print(ctrls_real)
print(ctrls_im)
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
# print(evolution.fid_grad[-2])
# print(evolution.fid_grad[-1])
print("Energy")
print(evolution.energy_list)


times = times.tolist()
times.append(times[-1] + (times[1] - times[0]))
zerolist = []
onelist = []
new_fwd_evo = [_num_vec_to_op(state.detach().numpy(), glob_dim) for state in evolution.fwd_evo]

print("Path Evo")
print(evolution.PathDist_Evo)
print("Path Ref")
print(evolution.PathDist_ref)


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

# After optimization in EVAL_spline_Example.py
import matplotlib.pyplot as plt

# Convert tensors to numpy arrays
fid_errors = [err.detach().numpy() if torch.is_tensor(err) else err 
              for err in evolution.fid_err_list]
ent_errors = [err.detach().numpy() if torch.is_tensor(err) else err 
              for err in evolution.ent_error_list]

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Plot fidelity and entropy errors on the left subplot
iterations = range(len(fid_errors))
ax1.plot(iterations, fid_errors, 'b-', label='Fidelity Error')
ax1.set_xlabel('Iteration')
ax1.set_ylabel('Fidelity Error', color='b')
ax1.tick_params(axis='y', labelcolor='b')

# Create second y-axis for entropy on the left subplot
ax1_2 = ax1.twinx()
ax1_2.plot(iterations, ent_errors, 'r-', label='Entropy Error')
ax1_2.set_ylabel('Entropy Error', color='r')
ax1_2.tick_params(axis='y', labelcolor='r')

# Add title and legend to the left subplot
ax1.set_title('Fidelity and Entropy Errors')
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax1_2.get_legend_handles_labels()
ax1_2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
ax1.grid(True)

# Plot energy + 5 on the right subplot
energy_minus_5 = [(energy.detach().numpy() + 5) if torch.is_tensor(energy) else (energy + 5) for energy in evolution.energy_list]
ax2.plot(iterations, energy_minus_5, 'g-', label='Energy + 5')
ax2.set_xlabel('Iteration')
ax2.set_ylabel('Energy + 5', color='g')
ax2.tick_params(axis='y', labelcolor='g')

# Add title and legend to the right subplot
ax2.set_title('Energy + 5')
ax2.legend(loc='upper right')
ax2.grid(True)

plt.tight_layout()
plt.show()


# Add after optimization in EVAL_spline_Example.py
import matplotlib.pyplot as plt

# Get time points
time_points = time_list[:-1]

# Plot controls
plt.figure(figsize=(12, 6))

# Real controls
plt.subplot(1, 2, 1)
for i in range(ctrls_real.shape[1]):
    plt.plot(time_points, ctrls_real[:, i].detach().numpy(), 
             label=f'Control {i+1} (Real)')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.title('Real Controls')
plt.legend()
plt.grid(True)

# Imaginary controls
plt.subplot(1, 2, 2)
for i in range(ctrls_im.shape[1]):
    plt.plot(time_points, ctrls_im[:, i].detach().numpy(), 
             label=f'Control {i+1} (Imaginary)')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.title('Imaginary Controls')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
