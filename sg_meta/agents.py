"""Implementations of Leader, Follower, and Meta-Learning classes."""
import numpy as np
from scipy.optimize import minimize, LinearConstraint, Bounds


class Leader:
    def __init__(self, param, env) -> None:
        self.seed = param['seed']
        self.rng = np.random.default_rng(self.seed)

        self.dimx, self.dimy, self.dimv = param['dimx'], param['dimy'], param['dimv']
        self.dims = self.dimx * self.dimy * self.dimv
        self.dimua, self.dimub = param['dimua'], param['dimub']
        self.dimT, self.dt = param['dimT'], param['dt']

        self.c1, self.c2 = param['l_reward']['c1'], param['l_reward']['c2'] 
        self.c3, self.c4 = param['l_reward']['c3'], param['l_reward']['c4']

        self.p = env.transition_matrix()
        self.ga = env.stage_reward(self.c1, self.c2, self.c3, self.c4)
        self.gaf = env.terminal_reward() 
        self.ga_old = self.ga       # in case of changing utilities
        self.gam, self.lam = param['gam'], param['lam']
        self.sigma = param['sigma']
        
        self.meta = None
    

    def set_utility(self, g):
        """Set the leader's utility externally."""
        self.ga = g
    

    def forward_predict(self, sid_init):
        """
        This function predicts all possible states given the initial state for the use of DP.
        Note that it is not the same forward_predict() in the follower's utility, where the leader's strategy is given
        """
        S_set = [ [sid_init] ]      # S_0 = [sid_init]
        for t in range(self.dimT):
            S_t = S_set[-1]
            S_tp1 = []
            for i in S_t:
                # predict for all possible actions
                for j in range(self.dimua):
                    for k in range(self.dimub):
                        #S_tp1.append( self.dynamics(i,j,k) )    ##
                        S_tp1.append( self.p[:, i,j,k].argmax() )
            S_tp1 = list(set(S_tp1))    # get unique S_tp1
            S_set.append(S_tp1)
        return S_set
    

    def get_policy_dp(self, S_set, gb):
        """
        This function performs DP to get the optiaml and sampled follower's policy for s_t in S_t based on the given gb.
        S_set = [S_0, S_1, ..., S_T]: collection of statesm which can performs complete DP based on the dynamics.
        We can set S_t = all s to find complete DP policy if necessary.
        Output: (for the ease of storage and index)
        - ua[t,s, :] = prob(.|s): leader's policy at time t and state s.    0 for s_t not in S_t
        - ub[t,s, :] = prob(.|s): follower's policy at time t and state s.  0 for s_t not in S_t
        - ub_samp[t,s, :] = e(.|s): follower's sampled action at time t and state s. 0 for s_t not in S_t
        """
        ua, ub = np.zeros((self.dimT,self.dims, self.dimua)), np.zeros((self.dimT,self.dims, self.dimub))
        ub_samp = np.zeros_like(ub)
        # assign terminal value if necessary
        VA = np.zeros((self.dimT+1, self.dims))
        VB = np.zeros((self.dimT+1, self.dims))
        VA[-1, :], VB[-1, :] = self.gaf, self.gaf
        for t in reversed(range(self.dimT)):
            if self.sigma[t]:
                # perform one-step meta-learning for s in the training data set (can be implemented in parallel)
                for i in S_set[t]:
                    # compute composite utility ga_comb and gb_comp
                    ga_comp = self.ga[i, :] + self.gam * np.tensordot(self.p[:,i,:,:], VA[t+1,:], axes=(0,0))
                    gb_comp = gb[i, :] + self.gam * np.tensordot(self.p[:,i,:,:], VB[t+1,:], axes=(0,0))
                    #if i == 82:     # DEBUG
                    #    tmp = 1
                    # update leader's value function
                    VA[t, i], x_opt = self.meta.update_leader_t_s(ga_comp, gb_comp)
                    # update follower's value function
                    y_opt = np.exp(self.lam*x_opt @ gb_comp) / np.exp(self.lam*x_opt @ gb_comp).sum()     # follower's QR
                    VB[t, i] = x_opt @ gb_comp @ y_opt - y_opt @ np.log(y_opt) / self.lam
                    # record trajectory
                    ua[t,i, :], ub[t,i, :] = x_opt, y_opt
                    idx = self.rng.choice(np.arange(self.dimub), p=y_opt)   # sample from y, QR response = mix strategy
                    idx = np.argmax(y_opt)     # or use the one with the max probability
                    ub_samp[t,i, idx] = 1
            else:
                # follower always select 0, update value function for s in the training data set
                for i in S_set[t]:
                    # update leader's value function
                    ga_comp = self.ga[i, :, 0].copy()
                    for j in range(self.dimua):
                        #sid = self.dynamics(i, j, 0)        ##
                        sid = self.p[:, i, j, 0].argmax()
                        ga_comp[j] += self.gam * VA[t+1, sid]
                    VA[t, i], ua_opt = np.max(ga_comp), np.argmax(ga_comp)
                    # update follower's value function
                    #sid_tp1 = self.dynamics(i, ua_opt, 0)   ##
                    sid_tp1 = self.p[:, i, ua_opt, 0].argmax()
                    VB[t, i] = gb[i, ua_opt, 0] + self.gam * VB[t+1, sid_tp1]
                    # record trajectory
                    ua[t,i, ua_opt] = 1
                    ub[t,i, 0] = 1     
                    ub_samp[t,i, 0] = 1
        return ua, ub, ub_samp



class Follower:
    def __init__(self, param, env, theta) -> None:
        self.seed = param['seed']
        self.rng = np.random.default_rng(self.seed)
        self.theta = theta

        self.dimx, self.dimy, self.dimv = param['dimx'], param['dimy'], param['dimv']
        self.dims = self.dimx * self.dimy * self.dimv
        self.dimua, self.dimub = param['dimua'], param['dimub']
        self.dimT, self.dt = param['dimT'], param['dt']

        self.c1, self.c2 = param['f_reward'][theta]['c1'], param['f_reward'][theta]['c2']
        self.c3, self.c4 = param['f_reward'][theta]['c3'], param['f_reward'][theta]['c4']
        
        self.p = env.transition_matrix()
        self.gb = env.stage_reward(self.c1, self.c2, self.c3, self.c4)
        self.gbf = env.terminal_reward() 
        self.gam, self.lam = param['gam'], param['lam']
        self.sigma = param['sigma']
    

    def set_utility(self, g):
        """
        This function sets the follower's utility externally.
        """
        self.gb = g
    

    def forward_predict(self, ua, sid_init):
        """
        This function predicts all possible states given the leader's policy for the use of DP.
        ua[t,s,:] = prob(.|s): leader's policy at time t and state s.
        """
        S_set = [ [sid_init] ]          # S_0 = [sid_init]
        for t in range(self.dimT):
            S_t = S_set[-1]
            S_tp1 = []
            for i in S_t:
                x = ua[t, i,:]
                # predict for follower's all possible actions
                for j in range(self.dimub):
                    prob_tp1 = np.tensordot(self.p[:,i,:,j], x, axes=(1, 0))
                    S_tp1 += np.nonzero(prob_tp1)[0].tolist()
            S_tp1 = list(set(S_tp1))    # get unique S_tp1
            S_set.append(S_tp1)
        return S_set


    def get_br_dp(self, ua, S_set):
        """
        This function performs DP to get the optimal and sampled follower's policy for s_t in S_t.
        ua[t,s,:] = prob(.|s): leader's policy at time t and state s.
        S_set = [S_0, S_1, ..., S_T]: collection of statesm which can performs complete DP based on the dynamics.
        We can set S_t = all s to find complete DP policy if necessary.
        """
        # assign terminal value if necessary. Othewise, V_T = 0 for all s
        V = np.zeros((self.dimT+1, self.dims))
        V[-1, :] = self.gbf
        #ub, ub_samp = [], []
        ub = np.zeros((self.dimT, self.dims, self.dimub))
        ub_samp = np.zeros_like(ub)
        for t in reversed(range(self.dimT)):
            S_t = S_set[t]
            #ub_t, ub_samp_t = [], []
            if self.sigma[t]:
                # perform DP to update value and policy
                for i in S_t:
                    x = ua[t,i,:]
                    gb_comp = self.gb[i,:] + self.gam * np.tensordot(self.p[:,i,:,:], V[t+1,:], axes=(0,0))
                    y = np.exp(self.lam*x @ gb_comp) / np.exp(self.lam*x @ gb_comp).sum()       # follower's QR
                    V[t, i] = x @ gb_comp @ y - y @ np.log(y) / self.lam
                    #ub_t.append( [i] + y.tolist() )
                    ub[t, i, :] = y
                    idx = self.rng.choice(range(self.dimub), p=y)      # sample from y, QR response = mix strategy
                    idx = np.argmax(y)
                    #ysamp = np.zeros(self.dimub)
                    #ysamp[idx] = 1
                    #ub_samp_t.append( [i] + ysamp.tolist() )
                    ub_samp[t, i, idx] = 1
            else:
                for i in S_t:
                    x = ua[t,i,:]
                    V[t, i] = x @ ( self.gb[i,:, 0] + self.gam * np.tensordot(self.p[:,i,:,0], V[t+1,:], axes=(0,0)) )
                    #y = np.zeros(self.dimub)
                    #y[0] = 1
                    #ub_t.append( [i] + y.tolist() )     # always choose action 0
                    #ub_t.ub_samp_t( [i] + y.tolist() )
                    ub[t, i, 0] = 1     
                    ub_samp[t, i, 0] = 1
            #ub.insert(0, np.array(ub_t))    # time is reversed in DP, insert instead of prepend
            #ub_samp.insert(0, np.array(ub_samp_t))
        return ub, ub_samp


    def sample_br_traj(self, ua, sid_init):
        """
        This function samples a BR trajectory (a decision tree) starting from sid_init as one data point, 
        given leader's policy and sampled follower's QR response
        ua[t,s,:] = prob(.|s) contains leader's strategies for all possible states s.
        S_t and A_t have the same size, which record possible states at time t and the corresponding policy trajectory.
        """
        S_set = self.forward_predict(ua, sid_init)
        _, ub_samp = self.get_br_dp(ua, S_set)
        # predict states using sampled QR response and record the decision tree
        S_t = [sid_init]
        data = [ np.concatenate( (np.array([0,sid_init]), ua[0,sid_init,:], ub_samp[0,sid_init,:])) ]
        for t in range(self.dimT-1):    # only need to compute policies up to T-1
            S_tp1 = []
            for s_t in S_t:
                # predict S_tp1 and append to S_tp1
                x, y = ua[t, s_t, :], ub_samp[t, s_t, :]
                if y.sum() == 0:    # DEBGU
                    tmp = 1     # for testing, make sure the correct follower's response is used
                prob_tp1 = np.tensordot( np.tensordot(self.p[:,s_t,:,:], x, axes=(1,0)), y, axes=(1,0) )
                S_tp1 += np.nonzero(prob_tp1)[0].tolist()
            S_tp1 = list(set(S_tp1))    # prune S_tp1
            # add policy according to S_tp1
            for s_tp1 in S_tp1:
                data.append( np.concatenate( (np.array([t+1,s_tp1]), ua[t+1,s_tp1,:], ub_samp[t+1,s_tp1,:]) ) )
            S_t = S_tp1
        return np.array(data)


    def sample_wapper(self):
        def generate_random_ua():
            ua = self.rng.random((self.dimT, self.dims, self.dimua))
            for t in range(self.dimT):
                b = np.sum(ua[t, :], axis=1)
                ua[t, :] = np.diag(1/b) @ ua[t, :]
            return ua
        
        data = []
        # sample ITER*5 data entries. reuse leader's policy for different initial states.
        ITER = 50
        for iter in range(ITER):
            S_init = self.rng.choice(self.dims, 5)
            ua = generate_random_ua()
            for s in S_init:
                data.append( self.sample_br_traj(ua, s) )
        for i in range(len(data)):
            # add data index for each BR trajectory
            idx = (i+1) * np.ones((data[i].shape[0], 1))
            data[i] = np.hstack( (idx, data[i]) )
        return np.vstack(data)



class Meta(Leader):
    def __init__(self, param, env) -> None:
        super().__init__(param, env)
        self.total_type, self.type_pdf = param['total_type'], param['type_pdf']
        self.alp, self.beta = param['alp'], param['beta']
    

    def sample_tasks(self, N):
        """
        This function samples a batch of tasks with size N from type_pdf mu.
        """
        return self.rng.choice(np.arange(self.total_type), N, p=self.type_pdf)
    

    def sample_task_theta(self, theta, data, N):
        """
        This function samples N data entry of task theta from given data. Each data entry is a complete decision tree.
        N should be smaller than the totoal number of decision trees.
        data[theta] structure: [data#, t, s, piA, piB]
        """
        D_theta = data[theta]
        idx = self.rng.choice(np.unique(D_theta[:, 0]), N, replace=False)  # sample N decision trees using idx
        D_sample = []
        for i in range(idx.shape[0]):
            D_sample.append( D_theta[D_theta[:,0] == idx[i]] )
        return np.vstack(D_sample)
    

    def extract_S_set(self, data):
        """
        This function recovers S_t from given data. Only up to T-1 since there is no S_T in the data.
        data is a list, data[i] structure: [data#, t, s, piA, piB]
        """
        S_set = []
        for t in range(self.dimT):
            S_t = []
            for i in range(len(data)):
                D = data[i]
                idx = np.argwhere(D[:, 1] == t)[:, 0]
                S_t += D[idx, 2].astype('int64').tolist()
            S_t = list(set(S_t))    # get unique S_t
            S_set.append(S_t)
        return S_set


    def extract_policy_theta_t_s(self, t, sid, data):
        """
        This function extacts the policies from the given data for a fixed time t and state s.
        data structure: [data#, t, s, piA, piB]
        return a list [piA, piB]
        """
        D_theta = data[data[:, 1] == t]     # find time, must be non-empyt
        D_theta = D_theta[D_theta[:, 2] == sid]     # find state, can be empty because of sampling
        return [D_theta[:, 3:self.dimua+3], D_theta[:, self.dimua+3:]]


    def meta_dp(self, task_batch, data, g_meta):
        """
        This function performs one iteration of meta-learning using DP
        """
        # store meta cost
        cost_s_t_batch = np.zeros((self.dimT, self.dims, len(task_batch)))
        # sample training and testing data for each type of follower
        D_train, D_test = [], []    # D_train, D_test is ordered according to task_batch
        for theta in task_batch:
            D_train.append( self.sample_task_theta(theta, data, N=10) )
            D_test.append( self.sample_task_theta(theta, data, N=5) ) 
        S_set = self.extract_S_set(D_train)

        # assign terminal value if necessary
        VA = np.zeros((self.dimT+1, self.dims))
        VB = np.zeros((self.dimT+1, self.dims))
        VA[-1, :], VB[-1, :] = self.gaf, self.gaf
        for t in reversed(range(self.dimT)):
            if self.sigma[t]:
                # perform one-step meta-learning for s in the training data set (can be implemented in parallel)
                for i in S_set[t]:
                    # compute composite utility gb_comp as the initial value for meta-learning
                    tmp = self.gam * np.tensordot(self.p[:,i,:,:], VB[t+1,:], axes=(0,0))   # future value matrix
                    gb_comp = g_meta[i, :] + tmp
                    gb_comp, cost_s_t_batch[t,i,:] = self.meta_t_s(task_batch, t, i, gb_comp, D_train, D_test)    # meta-learning to find new composite utility
                    g_meta[i, :] = gb_comp - tmp    #subtract composite utility to obtain new estimated follower's utility
                    # update leader's value function
                    ga_comp = self.ga[i, :] + self.gam * np.tensordot(self.p[:,i,:,:], VA[t+1,:], axes=(0,0))
                    VA[t, i], x_opt = self.update_leader_t_s(ga_comp, gb_comp)
                    # update follower's value function
                    y_opt = np.exp(self.lam*x_opt @ gb_comp) / np.exp(self.lam*x_opt @ gb_comp).sum()     # follower's QR
                    VB[t, i] = x_opt @ gb_comp @ y_opt - y_opt @ np.log(y_opt) / self.lam
            else:
                # follower always select 0, update value function for s in the training data set
                for i in S_set[t]:
                    # update leader's value function
                    ga_comp = self.ga[i, :, 0].copy()
                    for j in range(self.dimua):
                        #sid = self.dynamics(i, j, 0)    ##
                        sid = self.p[:, i, j, 0].argmax()
                        ga_comp[j] += self.gam * VA[t+1, sid]
                    VA[t, i], ua_opt = np.max(ga_comp), np.argmax(ga_comp)
                    # update follower's value function
                    #sid_tp1 = self.dynamics(i, ua_opt, 0)   ##
                    sid_tp1 = self.p[:, i, ua_opt, 0].argmax()
                    VB[t, i] = g_meta[i, ua_opt, 0] + self.gam * VB[t+1, sid_tp1]
        return g_meta, cost_s_t_batch


    def meta_t_s(self, task_batch, t, sid, gb, D_train, D_test):
        """
        This function performs one-step meta-learning for a particular time t and state s.
        Note: gb is the follower's composite utility, not the follower's utility.
        """
        gb_mid = []
        D_train_t_s, D_test_t_s = [], []
        # inner loop for each type theta
        for i in range(len(task_batch)):
            D_train_t_s.append( self.extract_policy_theta_t_s(t, sid, D_train[i]) )    # sample training data for state s and time t
            grad_theta = self.compute_loss_grad(D_train_t_s[i], gb)
            gb_mid.append(gb - self.alp * grad_theta)
            D_test_t_s.append( self.extract_policy_theta_t_s(t, sid, D_test[i]) )     # sample testing data for outer loop
        # outer loop for meta
        grad_meta = np.zeros_like(gb)
        for i in range(len(task_batch)):
            grad_meta += self.compute_loss_hessprod(D_train_t_s[i], D_test_t_s[i], gb, gb_mid[i])
        gb_new = gb - self.beta / len(task_batch) * grad_meta
        
        # compute extract quantity for plotting and printing
        #cost_s_t_batch = np.zeros((len(task_batch,2)))     # record meta_cost at each s and t and task batch.
        cost_s_t_batch = np.zeros(len(task_batch))          # record meta_cost at each s and t.
        for i in range(len(task_batch)):
            #cost_s_t_batch[i, 0] = self.compute_loss(D_train_t_s[i], D_test_t_s[i], gb)
            #cost_s_t_batch[i, 1] = task_batch[i]
            cost_s_t_batch[i] = self.compute_loss(D_train_t_s[i], D_test_t_s[i], gb)

        return gb_new, cost_s_t_batch


    def compute_loss(self, D_train, D_test, gb):
        """
        This function computes the meta cost using testing data set. D_train is used if D_test is empty.
        We can also set D_test = D_train to compute loss for any gb and data.
        cost = cross entropy = - 1/N \sum_{i} [y]_i log([qr]_i), i is data entry index.
        """
        if D_train[0].size == 0:
            return 0
        
        if D_test[0].size == 0:
            D_test = D_train
        piA, piB = D_test[0], D_test[1]
        N = piA.shape[0]
        cost = 0
        for i in range(N):
            x, y = piA[i, :], piB[i, :]
            qr = np.exp(self.lam*x @ gb) / np.exp(self.lam*x @ gb).sum()    # follower's QR
            cost += (- y @ np.log(qr))  
        cost /= N
        return cost


    def compute_loss_grad(self, D_train, gb):
        """
        This function computes the gradient of negative cross entropy loss given the data piA, piB.
        grad is in a matrix form.
        """
        if D_train[0].size == 0:
            return np.zeros_like(gb)    # no update
        piA, piB = D_train[0], D_train[1]
        N = piA.shape[0]
        grad = np.zeros_like(gb)
        for i in range(N):
            x, y = piA[i, :], piB[i, :]
            qr = np.exp(self.lam*x @ gb) / np.exp(self.lam*x @ gb).sum()    # follower's QR
            grad += self.lam*x[:, None] @ (qr-y)[None, :]  # add new axis to obtain a 2d array.
        grad /= N
        return grad
    

    def compute_loss_hessprod(self, D_train, D_test, gb, gb_mid):
        """
        This function computes the hessian product ( I-alp*hess(gb; Dtrain) ) * grad(gb_mid, Dtest).
        hessprod is in a matrix form.
        """
        if D_train[0].size == 0:
            return np.zeros_like(gb)    # no update
        
        if D_test[0].size == 0:
            grad = self.compute_loss_grad(D_train, gb_mid)      # keep inner and outer update consistent
        else:
            grad = self.compute_loss_grad(D_test, gb_mid)
        # d^2f/dVi^2 = lam^2 * x@x.T [exp/sum(exp) - exp^2/sum(exp)^2]
        piA, piB = D_train[0], D_train[1]
        N = piA.shape[0]
        hess = np.zeros((self.dimub, self.dimua,self.dimua))    # hess[i] is for d^f/dV_i, a dimua x dimua matrix
        for i in range(N):
            x, y = piA[i, :], piB[i, :]
            qr = np.exp(self.lam*x @ gb) / np.exp(self.lam*x @ gb).sum()    # follower's QR
            for j in range(self.dimub):
                hess[j, :] += self.lam**2 * x[:, None] @ x[None, :] * (qr[j] - qr[j]**2) / N
        # compute hessprod
        hessprod = np.zeros_like(gb)
        for i in range(self.dimub):
            hessprod[:, i] = ( np.eye(self.dimua) - self.alp * hess[i] ) @ grad[:, i]
        return hessprod


    def update_leader_t_s(self, ga_comp, gb_comp):
        """
        This function updates leader's value function given new gb. follower's QR is known to the leader given gb.
        Note we need to max objective, add "-" before obj and jac. But return positive obj, add "-" before res.fun.
        check myjac computation.
        """
        def myobj(x, ga_comp, gb_comp):
            y = np.exp(self.lam*x @ gb_comp) / np.exp(self.lam*x @ gb_comp).sum()   # follower's QR
            return - x @ ga_comp @ y

        def myjac(x, ga_comp, gb_comp):
            y = np.exp(self.lam*x @ gb_comp) / np.exp(self.lam*x @ gb_comp).sum()
            dydx = self.lam * (gb_comp - (gb_comp@y)[:,None] @ np.ones((1, self.dimub))) @ np.diag(y)   # gradient layout not jac
            #dy = np.zeros((self.dimub, self.dimua))
            #for i in range(self.dimub):
            #    dy[i, :] = y[i] * (self.lam*gb_comp[:,i]) + y[i] * (self.lam*gb_comp @ y)
            return - ga_comp @ y - (x @ ga_comp) @ dydx.T
        
        def myobj1(x, ga_comp, gb_comp):
            y = np.exp(self.lam*x @ gb_comp) / np.exp(self.lam*x @ gb_comp).sum()   # follower's QR
            return x @ ga_comp @ y
        def myjac1(x, ga_comp, gb_comp):
            y = np.exp(self.lam*x @ gb_comp) / np.exp(self.lam*x @ gb_comp).sum()
            dydx = self.lam * (gb_comp - (gb_comp@y)[:,None] @ np.ones((1, self.dimub))) @ np.diag(y)   # gradient layout not jac
            return ga_comp @ y + (x @ ga_comp) @ dydx.T
  
        mybound = Bounds(np.zeros(self.dimua), np.ones(self.dimua))
        myconstr = [LinearConstraint(A=np.ones((1, self.dimua)), lb=1, ub=1)]
        #x0 = np.random.rand(self.dimua)
        #x0 /= x0.sum()
        #res = minimize(myobj, x0, args=(ga_comp,gb_comp), jac=myjac, bounds=mybound, constraints=myconstr)
        #print(res.status, res.message)
        #return -res.fun, res.x  # return positive value, add "-" before 
        # run multiple times with different initial conditions to find the optimal solution
        ITER = 150   #100
        f_list, sol_list = np.zeros(ITER), np.zeros((ITER, self.dimua))
        for i in range(ITER):
            x0 = np.random.rand(self.dimua)
            x0 /= x0.sum()
            res = minimize(myobj, x0, args=(ga_comp,gb_comp), jac=myjac, bounds=mybound, constraints=myconstr)
            f_list[i] = res.fun
            sol_list[i, :] = res.x / res.x.sum()
            #f_list.append(res.fun)
            #sol_list.append(res.x / res.x.sum())
        idx = np.argmin(f_list)
        return -f_list[idx], sol_list[idx, :]  # return positive value, add "-" before 
        