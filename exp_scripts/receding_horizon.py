"""This script implements receding horizon planning using the adapted model."""
import sys, os
sys.path.insert(1, os.path.realpath('.'))

import json
import numpy as np
from sg_meta.utils import Environment, Utilities
from sg_meta.agents import Leader, Follower, Meta


def receding_horizon(theta, s_init=[0,1,0]):
    """
    This function implements receding horizon control for a specific type of follower, starting from s_init.
    Output:
        - x_traj: ITER+1 x dims, x_traj[t, :] is s_t
        - ua_traj: ITER x dimua, ua_traj[t, :] = [0,0,1,0,0], one-hot vector
        - ub_traj: ITER x dimub, ub_traj[t, :] = [0,0,1,0,0], one-hot vector
    """
    param = json.load(open('parameters.json'))
    env = Environment(param)
    U = Utilities()

    leader = Leader(param, env)
    leader.meta = Meta(param, env)
    follower = Follower(param, env, theta)
    gb_adapt = U.load_adapted_utility(theta)
    U.print_key_parameters(param)   # comment if not necessary 

    leader.set_utility(gb_adapt)        # set leader's utility to gb_adapt to help the planning

    iter, ITER_PLAN = 0, 15
    s_traj, ua_traj, ub_traj = [], [], []
    s_traj.append( s_init )
    for iter in range(ITER_PLAN):
        sid = env.from_s_to_sid(s_traj[-1])
        # leader observes s, predict S_set, and do DP over S_set
        S_set = leader.forward_predict(sid)
        ua, ub_l, ub_samp_l = leader.get_policy_dp(S_set, gb_adapt)
        
        # follower observes s and ua_traj, predict S_set', and do DP over S_set'
        S_set = follower.forward_predict(ua, sid)
        ub_f, ub_samp_f = follower.get_br_dp(ua, S_set)
        
        # take the first strategy. leader samples from mixed strategy.
        x = ua[0, sid, :]
        y = ub_samp_f[0, sid, :]    # or y = ub_samp_l, follower follow the leader's instruction
        # speficy leader follower actions
        a = np.random.choice(np.arange(x.shape[0]), p=x)
        a = np.argmax(x)
        b = np.nonzero(y)[0][0]
        sid_new = env.from_s_to_sid( env.dynamics(env.from_sid_to_s(sid), a, b) )

        # save trajectory
        s_traj.append( env.from_sid_to_s(sid_new) )
        ub_traj.append(ub_samp_f[0, sid, :])
        tmp = np.zeros_like(x)
        tmp[a] = 1
        ua_traj.append(tmp)
        
        print(f'planning iter: {iter+1}')

        if env.is_destination(env.from_sid_to_s(sid_new)):
            break
    print('s_traj:', s_traj)
    #print('ua_traj:', ua_traj)
    #print('ub_traj:', ub_traj)
    return np.array(s_traj), np.array(ua_traj), np.array(ub_traj)


if __name__ == '__main__':
    theta = 1      # [0,1,2,3,4]
    s_init = [0, 1, 0]      # specify initial state of the car
    s, ua, ub = receding_horizon(theta, s_init)

    # save if necessary
    np.save(f'data/s_traj_{theta}.npy', s)
    np.save(f'data/ua_traj_{theta}.npy', ua)
    np.save(f'data/ub_traj_{theta}.npy', ub)
