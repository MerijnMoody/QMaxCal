import scipy.linalg as la

import numpy as np
from numpy import e, real, sort, sqrt, imag, inner, conj, inf
from scipy import *
from scipy.linalg import logm
from scipy.optimize import minimize, root, check_grad, approx_fprime

# qutip
from qutip import *
from qutip import Qobj
from qutip.entropy import entropy_relative
from qutip.core.superoperator import liouvillian, vector_to_operator, operator_to_vector
from qutip.core import data as _data

import matplotlib.pyplot as plt

def _num_vec_to_op(rho):
    #print(rho)
    return np.array([[rho[0][0], rho[1][0]], [rho[2][0], rho[3][0]]])

def _grad_ent(rho, sigma, gamma, base=e, sparse=False):
    #print("rho" + str(rho))
    #print("sigma" + str(sigma))
    #print("gamma" + str(gamma))
    rho = np.array(rho.full())
    sigma = np.array(sigma.full())
    gamma = np.array(gamma.full())
    try:
        out = np.trace(gamma @ (logm(sigma) - logm(rho)))
        return out
    except:
        #raise TypeError(str(gamma), str(sigma), str(rho))
        return 100


class LindBladEvolve(object):
    def __init__(self):
        self.fid_err_list = []
        self.time_list = None
        self.dt = None
        self.n_ts = None
        self.initial = None
        self.target = None
        self.ref_evo = None
        self.onwd_evo = None
        self.evo_list = []
        self.ent_error_list = [9999999999999999]
        self.fwd_evo = None
        self.ctrls = None
        self.ctrl_gen = None
        self.dyn_gen = None
        self.dims = None
        self.n_ctrls = None
        self.prop = None
        self.prop_grad = None
        self.scale = 10
        self.ent_grad = None
        self.fid_grad = None
        self.fid_grad_approx = None
        self.ent_grad_approx = None

    def _get_dyn_gen(self, k):
        dyn_gen = self.dyn_gen
        for j in range(self.n_ctrls):
            dyn_gen = dyn_gen + self._get_ctrl_gen(k,j)
        return dyn_gen

    def _get_ctrl_gen(self, k,j):
        return self.ctrls[k][j]*self.ctrl_gen[j]

    def _compute_prop_grad(self, k, j, compute_prop=True):
        """
        Calculate the gradient of propagator wrt the control amplitude
        in the timeslot using the expm_frechet method
        The propagtor is calculated (almost) for 'free' in this method
        and hence it is returned if compute_prop==True
        Returns:
            [prop], prop_grad
        """
        A = self._get_dyn_gen(k).full() * self.dt
        E = self._get_ctrl_gen(k, j).full() * self.dt
        #raise TypeError(str(self.time_list))
        if compute_prop:
            prop_dense, prop_grad_dense = la.expm_frechet(A, E)
            prop = Qobj(prop_dense, dims=self.dyn_gen.dims)
            prop_grad = Qobj(prop_grad_dense, dims=self.dyn_gen.dims)
            return prop, prop_grad
        else:
            prop_grad_dense = la.expm_frechet(A, E, compute_expm=False)
            prop_grad = Qobj(prop_grad_dense, dims=self.dyn_gen.dims)
            return prop_grad

    def _evo(self):
        n_ts = self.n_ts
        n_ctrls = self.n_ctrls

        self.prop_grad = [[0 for _ in range(n_ctrls)] for _ in range(n_ts)]
        self.prop = [0 for _ in range(n_ts)]

        for k in range(n_ts):
            for j in range(n_ctrls):
                if j == 0:
                    (
                        self.prop[k],
                        self.prop_grad[k][j],
                    ) = self._compute_prop_grad(k, j)
                else:
                    self.prop_grad[k][j] = self._compute_prop_grad(k,j, compute_prop=False)

        self.fwd_evo = [0 for _ in range(n_ts+1)]
        self.fwd_evo[0] = self.initial
        for k in range(n_ts):
            self.fwd_evo[k+1] = self.prop[k] * self.fwd_evo[k]

        R = range(n_ts - 2, -1, -1)
        self.onwd_evo = [0 for _ in range(n_ts)]
        self.onwd_evo[n_ts - 1] = self.prop[n_ts - 1]
        for k in R:
            self.onwd_evo[k] = self.onwd_evo[k + 1] * self.prop[k]
            
    def _evo_ctrls(self, ctrls):
        self.ctrls = np.reshape(ctrls, (self.n_ts, self.n_ctrls))
        #self.ctrls = torch.reshape(ctrls, (self.n_ts, self.n_ctrls))
        self._evo()


    def _get_error(self, ctrls_list):
        ent_error = self._get_ent_err()
        fid_error = self._get_fid()

        return [ent_error, fid_error]

    def _get_ent_err(self):
        ent_error = 0
        ref_evo = self.ref_evo
        n_ts = self.n_ts
        #raise TypeError(str(self.fwd_evo) + " " + str(ref_evo) + " " + str(len(self.fwd_evo)) + " " + str(len(ref_evo)))
        for i in range(n_ts+1):
            ent_error += entropy_relative(vector_to_operator(Qobj(self.fwd_evo[i], dims=self.initial.dims)), ref_evo[i])

        return ent_error

    def _get_fid(self):
        #evo_final = _num_vec_to_op(self.fwd_evo[-1].full())
        #evo_f_diff = _num_vec_to_op(self.target.full()) - evo_final
        evo_final = self.fwd_evo[-1].full()
        evo_f_diff = self.target.full() - evo_final
        #raise TypeError(str(evo_f_diff.conj().T.dot(evo_f_diff)))
        #raise TypeError(str(self.target.full()) + " " + str(evo_final) + " " + str(evo_f_diff) + " " + str(evo_f_diff.conj().T.dot(evo_f_diff)))

        return np.real(np.trace(evo_f_diff.conj().T.dot(evo_f_diff)))

    def _compute_fid_err_grad(self):
        """
        Calculate exact gradient of the fidelity error function
        wrt to each timeslot control amplitudes.
        Uses the trace difference norm fidelity
        These are returned as a (nTimeslots x n_ctrls) array
        """
        n_ctrls = self.n_ctrls
        n_ts = self.n_ts

        # create n_ts x n_ctrls zero array for grad start point
        fid_grad = np.zeros([n_ts, n_ctrls])
        ent_grad = np.zeros([n_ts, n_ctrls])

        evo_final = self.fwd_evo[-1]
        evo_f_diff = self.target - evo_final
        for j in range(n_ctrls):
            for k in range(n_ts):
                entg = 0
                #raise TypeError(str(evo_grad))
                evo_grad_ent = self.prop_grad[k][j] * self.fwd_evo[k]
                for k2 in range(k+1, n_ts):
                    entg += (_grad_ent(vector_to_operator(self.fwd_evo[k2]),
                                       self.ref_evo[k2],
                                       vector_to_operator(evo_grad_ent))
                             + vector_to_operator(evo_grad_ent).tr()
                             )
                    evo_grad_ent = self.prop[k2] * evo_grad_ent
                
                fwd_evo = self.fwd_evo[k]
                evo_grad = self.prop_grad[k][j] * fwd_evo
                if k + 1 < n_ts:
                    evo_grad = self.onwd_evo[k+1] * evo_grad
                    # Note that the value should have not imagnary part, so
                    # using np.real, just avoids the complex casting warning
                    #raise TypeError(str(evo_grad) + str(self.onwd_evo[k+1] * self.initial) + str(self.fwd_evo[-1]))
                    g = (
                        -2
                        * np.real((evo_f_diff).dag() * evo_grad)
                    )

                if np.isnan(g):
                    g = np.Inf

                fid_grad[k][j] = g
                ent_grad[k][j] = np.real(entg)

        #raise TypeError(str(ent_grad) + str(fid_grad))
        #return [np.reshape(ent_grad, (self.n_ts * self.n_ctrls)), np.reshape(fid_grad, (self.n_ts * self.n_ctrls))]
                
        
        return [ent_grad, fid_grad]

    def _get_grad(self, ctrls):
        self.ctrls = np.reshape(ctrls, (self.n_ts, self.n_ctrls))
        self._evo()

        gradlist = self._compute_fid_err_grad()
        return gradlist[1]

    def _get_fid_ctrls(self, ctrls):
        self.ctrls = np.reshape(ctrls, (self.n_ts, self.n_ctrls))
        self._evo()
        return self._get_fid()

    def _get_ent_ctrls(self, ctrls):
        self.ctrls = np.reshape(ctrls, (self.n_ts, self.n_ctrls))
        self._evo()
        return self._get_ent_err()

    def H(self, args):
        ctrls = args[:-1]
        #ctrls = args
        self.ctrls = np.reshape(ctrls, (self.n_ts, self.n_ctrls))
        self._evo()

        lam = args[-1]
        error_list = self._get_error(ctrls)
        fid_error = error_list[1]
        ent_error = error_list[0]


        #raise TypeError(str(fid_error))
        #self.fid_err_list.append((fid_error, self.fwd_evo[-1]))
        grad_list = self._compute_fid_err_grad()
        #ent_grad = np.reshape(grad_list[0], (self.n_ts * self.n_ctrls))
        #fid_grad = np.reshape(grad_list[1], (self.n_ts * self.n_ctrls))
        #self.grads = [ent_grad, fid_grad]

        ent_grad = self.approximated_ent_grad(ctrls)
        fid_grad = self.approximated_fid_grad(ctrls)
        #return np.append(100*(ent_grad + abs(lam) * fid_grad), 100 * fid_error)
        return la.norm(np.append((ent_grad + 0*abs(lam) * fid_grad), 0*fid_error))
        #return ent_error-abs(lam)*(fid_error)
        #return np.append(np.zeros(20), self.scale*fid_error)
        #return fid_error

    def lagrange_grad(args):
        ctrls = args[:-1]
        lam = args[-1]


    def approximated_fid_grad(self, ctrls):
        #raise TypeError(ctrls)
        return approx_fprime(ctrls, self._get_fid_ctrls)

    def approximated_ent_grad(self, ctrls):
        #raise TypeError(ctrls)
        return approx_fprime(ctrls, self._get_ent_ctrls)
    
    def ascent(self, i_num, eps, c, fid_tar):
        ctrls = np.random.rand(self.n_ts * self.n_ctrls)
        self.lam = 10
        
        self.ctrls = np.reshape(ctrls, (self.n_ts, self.n_ctrls))
        self._evo()
        #for t in range(10):
        #    self.ctrls[t][0] = 0
        
        for i in range(i_num):
            error_list = self._get_error(ctrls)
            fid_error = error_list[1] - fid_tar
            ent_error = error_list[0]
            
            grad_list = self._compute_fid_err_grad()
            ent_grad = -grad_list[0]
            self.ent_grad = ent_grad
            fid_grad = grad_list[1]
            self.fid_grad = fid_grad
            
            #fid_grad = self.approximated_fid_grad(np.reshape(self.ctrls, (self.n_ts * self.n_ctrls)))
            #self.fid_grad_approx = fid_grad
            
            #ent_grad = 0*self.approximated_ent_grad(np.reshape(self.ctrls, (self.n_ts * self.n_ctrls)))
            #self.ent_grad_approx = ent_grad
            
            #raise TypeError(str(fid_grad) + str(grad_list[1]))
            self.lam += fid_error * eps
            for k in range(self.n_ts):
                for j in range(self.n_ctrls):
                    self.ctrls[k][j] -= ent_grad[k][j] + abs(self.lam) * fid_grad[k][j] + c * fid_error * fid_grad[k][j]
            
            #raise TypeError(str(np.array(ent_grado)/ent_grad) + str(np.array(fid_grado)/fid_grad))
            #for t in range(10):
            #    self.ctrls[t][0] = 0
            
            if i % 100 == 0:
                #self.ctrls += np.reshape(4*np.random.rand(self.n_ts * self.n_ctrls), (self.n_ts, self.n_ctrls))
                ent_err = self._get_ent_err()
                self.fid_err_list.append(self._get_fid())
                self.ent_error_list.append(self._get_ent_err())
                self.evo_list.append(self.fwd_evo)
            
            self._evo()
        
        fid_grad = self.approximated_fid_grad(np.reshape(self.ctrls, (self.n_ts * self.n_ctrls)))
        self.fid_grad_approx = fid_grad
        
        ent_grad = self.approximated_ent_grad(np.reshape(self.ctrls, (self.n_ts * self.n_ctrls)))
        self.ent_grad_approx = ent_grad
        
        
            
            

    def optimize(self):
        x0 = 0*np.random.rand(self.n_ts * self.n_ctrls+1)

        #return check_grad(self.H, self.approximated_grad, x0)
        #x0 = np.zeros(self.n_ts * self.n_ctrls + 1)
        res = minimize(self.H, x0=x0, method='nelder-mead', options={'maxfev':500})
        return res

def create_evolution(dyn_gen, H_con, rho0, rhotar, ref_evo, time_list, dims):
    evolution = LindBladEvolve()
    evolution.time_list = time_list
    evolution.n_ts = len(time_list)
    evolution.initial = rho0
    evolution.target = rhotar
    evolution.ref_evo = ref_evo
    evolution.ctrl_gen = H_con
    evolution.dyn_gen = dyn_gen
    evolution.n_ctrls = len(H_con)
    evolution.dims = dims
    evolution.dt = time_list[1] - time_list[0]

    return evolution
