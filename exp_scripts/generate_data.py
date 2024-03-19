"""This scripts generates the learning data for all types of followers."""
import sys, os
sys.path.insert(1, os.path.realpath('.'))

import json
import numpy as np
from sg_meta.utils import Environment
from sg_meta.agents import Leader, Follower, Meta


def generate_data():
    """
    This function generates the data set for all types of followers for meta learning algorithms.
    Generated data format: D[i, t] = [s, piA, piB], which has the shape N x K x (1+m+n)
    """
    param = json.load(open('parameters.json'))
    env = Environment(param)
    leader = Leader(param, env)
    leader.meta = Meta(param, env)
    follower = [Follower(param, env, theta) for theta in range(leader.meta.total_type)]
    
    if not os.path.exists('data'):
        os.mkdir('data')
    
    # generate leader and follower ground truth utility
    if not os.path.exists('data/ga.npy'):
        np.save('data/ga.npy', leader.ga)
    for theta in range(leader.meta.total_type):
        if not os.path.exists(f'data/gb{theta}.npy'):
            np.save(f'data/gb{theta}.npy', follower[theta].gb)
    
    # generate learning data for each type of follower, grouped by list.
    data = []
    for theta in range(leader.meta.total_type):
        fname = f'data/f{theta}_data.npy'
        if os.path.exists(fname):
            data.append( np.load(fname) )
        else:
            ub_traj = sample_wapper(follower[theta])
            np.save(fname, ub_traj)
            data.append( ub_traj )
            #print(ub_traj)
    return data


def sample_wapper(f):
    """wrapper to generate type theta follower's data."""
    def generate_random_ua():
        """generate random leader's control ua."""
        ua = f.rng.random((f.dimT, f.dims, f.dimua))
        for t in range(f.dimT):
            b = np.sum(ua[t, :], axis=1)
            ua[t, :] = np.diag(1/b) @ ua[t, :]
        return ua
    
    # generate N data entries
    data = []
    N = 250
    for i in range(N):
        sid_init = f.rng.choice(f.dims)             # choose one initial state
        ua = generate_random_ua()
        data_br = f.sample_br_traj(ua, sid_init)    # generate BR trjactory, a matrix.

        # add data index for each BR trajectory
        idx = (i+1) * np.ones((data_br.shape[0], 1))
        data.append( np.hstack( (idx, data_br) ) )
    return np.vstack(data)



if __name__ == '__main__':
    generate_data()
