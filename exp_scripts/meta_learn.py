"""This scripts implementation meta-learning and adaptation algorithms."""
import sys, os
sys.path.insert(1, os.path.realpath('.'))

import json
import numpy as np
from sg_meta.utils import Environment, Utilities
from sg_meta.agents import Leader, Follower, Meta


def sg_meta():
    """
    This function performs meta-learning. Task represents follower type.
    """
    param = json.load(open('parameters.json'))
    env = Environment(param)
    U = Utilities()

    leader = Leader(param, env)
    leader.meta = Meta(param, env)
    data = U.load_data()
    U.print_key_parameters(param)   # comment if not necessary 
    
    # Run meta-learning
    iter, ITER_META = 0, 1500
    N_samp_task = 5
    g_meta = 10*leader.rng.random((leader.dims, leader.dimua, leader.dimub)) + leader.ga    # initial value
    print('Start meta learning...\n')
    for iter in range(ITER_META):
        task_batch = leader.meta.sample_tasks(N=N_samp_task)
        g_meta, cost_s_t_batch = leader.meta.meta_dp(task_batch, data, g_meta)
        
        print(f'meta iter: {iter+1}')
        print(f'--task batch: {task_batch}')
        print(f'--meta cost: {cost_s_t_batch.reshape(cost_s_t_batch.size).tolist()}')
    print('\nmeta learning complete.')
    print(f'meta utility gb_meta: {g_meta.reshape(g_meta.size).tolist()}\n')
    
    return g_meta


def sg_adapt(theta, g_meta):
    """
    This function performs adaptation for a specific follower with type theta.
    """
    param = json.load(open('parameters.json'))
    env = Environment(param)
    U = Utilities()

    leader = Leader(param, env)
    leader.meta = Meta(param, env)
    data = U.load_data()

    iter, ITER_APAPT = 0, 30
    print(f'Start adaptation type {theta} follower...\n')
    for iter in range(ITER_APAPT):
        task_batch = [theta]
        g_adapt, _ = leader.meta.meta_dp(task_batch, data, g_meta)

        print(f'adaptation iter: {iter+1}')
    print('\nadaptation complete.')
    print(f'adapted utility gb_adapt_{theta}: {g_adapt.reshape(g_adapt.size).tolist()}')
    
    return g_adapt


if __name__ == '__main__':
    # perform meta-learning
    g_meta = sg_meta()
    np.save('data/gb_meta.npy', g_meta)

    # perform adaptation for type theta follower
    theta = 1   # [0,1,2,3,4]
    g_adapt = sg_adapt(theta, g_meta)
    np.save(f'data/gb_adapt_{theta}.npy', g_adapt)
