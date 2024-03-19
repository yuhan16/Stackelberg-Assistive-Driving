# Stackelberg Meta-Learning for Assistive Driving
This repo is for Stackelberg meta-learning assistive driving project.


## Requirements
- Python 3.11 or higher


## Running Scripts
1. Create a virtual environment with Python 3.11 and source the virtual environment:
```bash
$ python3.11 -m venv <your-virtual-env-name>
$ source /path-to-venv/bin/activate
```
2. `pip` install the requirements:
```bash
(venv-name)$ pip install -r requirements.txt
```
3. In the project directory, run scripts:
```bash
(venv-name)$ python experiments/generate_data.py    # generate_data.py as an example
```

**Note:** Run `generate_data.py`, `meta_learn.py`, and `receding_horizon.py` in turns to first complete learning and then perform planning.


## File Structures
- `requirements.txt`: necessary packages used in the project (installed by pip).
- `parameters.json`: configurations and parameters.
- `sg_meta/`: algorithm implementations.
- `exp_scripts/`: examples for running the algorithm.
- `logs/`: simulation log directory.
- `data/`: data directory. Sotre necessary data including plots and key training data for reference.

Class definitions:
- `Leader`: compute leader's utility and implement DP.
- `Follower`: compute follower's utility, generate BR response and trajectory.
- `Meta`: meta-learning related, including task and data sampling, and backward propogation implementation.
- `Environment`: implement environment dynamics and define rewards.
- `Utilities`: miscellaneous helper functions.


### Log Specifications
Use command line to redirect output and generate logs.
```bash
$ mkdir logs    # create logs directory
$ python3.11 exp_scripts/meta_learn.py > logs/meta.log
```
Change `print` functions in the code for customized log output.


### Coding Specifications
- State `s = [x,y,v]` is a 3d vector. `x=[0,...,10], y=[0,1,2], v=[0,1,2]`. 
  - Linear index for `s`:  `v` first increase, then `y`, then `x`. We assign a unique index to the state `s` starting from 0 to `|X|*|Y|*|V|`. `sid = x*|X| + y*|Y| + v*|V|`.
- Conditional probability as 1d numpy array. `prob(.|s) = [sid, piA]`$\in\mathbb{R}^{m+1}$, `piA`$\in\mathbb{R}^m$.
  - Leader and follower's strategies are time- and state-dependent policies. Use 1d numpy array to store the policy, e.g., `piA[t,:] = prob(.|s)`.
- BR data `D` are tree-like trajectories. 
  - A single data entry, `D[i] = [data#, t, s, piA, piB]`$\in\mathbb{R}^{m+n+3}$, represents a branch of a decision tree. `data#` is the tree index.
  - Each decision tree has multiple branches. The index `i` is meaningless.
  - Use column to indetify or search related data.
  - For a fixed time `t`, the leader and follower's policy is conditioned on the same `s`. No need for store `D[i,t] = [probA(.|s), probB(.|s)]`.
  - When `sigma(t) = 0`, we set `D[i,t] = [0,0,0]`.
- Trajectory is stored in a 2d numpy array with axis0 as time index. `x_traj[t, :] = x_t`, `u_traj[t, :] = u_t`.
- Use lists to store type-related quantity. E.g., `data[i]` is the learning data for type i follower. `gb_adapt[i]` is the adapted parameter for type i follower.

**Notes in data generation:** We need data in a decision tree to perform DP. Suppose we have only trajectory `probA_0(.|s),..., probA_T(.|s)` for some simulated trajectory. Then, we can only compute `V_T` for state `s_T`. We cannot obtain `V_Tm1` because we only have one `s_T` and thus cannot compute Bellman equation. We need all possible `s_T` starting from `s_Tm1`. Therefore, each data cannot be index by either `i` or `t` or `s` if using numpy array. We use the following format to store each data tree: `[data#, t, s, piA, piB]`. Then we stack all data together (a 2d array) and search them using columns. 
