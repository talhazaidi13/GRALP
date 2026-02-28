import os
import collections
import numpy as np
import gym
import pdb

from contextlib import (
    contextmanager,
    redirect_stderr,
    redirect_stdout,
)

@contextmanager
def suppress_output():
    """
        A context manager that redirects stdout and stderr to devnull
        https://stackoverflow.com/a/52442331
    """
    with open(os.devnull, 'w') as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)

with suppress_output():
    ## d4rl prints out a variety of warnings
    import d4rl

#-----------------------------------------------------------------------------#
#-------------------------------- general api --------------------------------#
#-----------------------------------------------------------------------------#

def load_environment(name):
    if type(name) != str:
        ## name is already an environment
        return name
    with suppress_output():
        wrapped_env = gym.make(name)
    env = wrapped_env.unwrapped
    env.max_episode_steps = wrapped_env._max_episode_steps
    env.name = name
    return env

def get_dataset(env):
    dataset = env.get_dataset()

    return dataset

def load_dataset(path):
    keys = ['state', 'action', 'reward', 'not_done', 'next_state', ]
    path_list = [os.path.join(path, f'{key}.npy') for key in keys]
    dataset = dict()
    for key, path in zip(keys, path_list):
        dataset[key] = np.load(path).squeeze()
        if 'state' in key or 'next_state' in key:
            # For Fetch 3:6 is the achieved goal == -3:0
            dataset[key] = dataset[key][:,:-6]

    # rename keys
    dataset['observations'] = dataset.pop('state')
    dataset['actions'] = dataset.pop('action')
    dataset['rewards'] = dataset.pop('reward')
    dataset['terminals'] = 1-dataset.pop('not_done')
    dataset['next_observations'] = dataset.pop('next_state')
    return dataset


def sequence_dataset(env, preprocess_fn, load_path=None, reward_shaping=0, radius=0.15, min_length=9, 
                     discount_array=None, returns_scale=1000) :
    """
    Returns an iterator through trajectories.
    Args:
        env: An OfflineEnv object.
        dataset: An optional dataset to pass in for processing. If None,
            the dataset will default to env.get_dataset()
        **kwargs: Arguments to pass to env.get_dataset().
    Returns:
        An iterator through dictionaries with keys:
            observations
            actions
            rewards
            terminals
    """
    if load_path is None:
        dataset = get_dataset(env)
        dataset = preprocess_fn(dataset)
    else:
        dataset = load_dataset(load_path)
        print(f'Loaded dataset from {load_path}')
        # show keys
        print(dataset.keys())

    N = dataset['rewards'].shape[0]
    data_ = collections.defaultdict(list)

    # The newer version of the dataset adds an explicit
    # timeouts field. Keep old method for backwards compatability.
    use_timeouts = 'timeouts' in dataset

    episode_step = 0
    waiting_for_reset = False
    print ('env_name: ', env.name)
    print ('state_shape: ', dataset['observations'].shape)
    print ('action_shape: ', dataset['actions'].shape)
    print ('reward_max: ', np.max(dataset['rewards']))
    print ('reward_min: ', np.min(dataset['rewards']))
    print ('terminal_count: ', np.sum(dataset['terminals']))
    print ('timeouts_count: ', np.sum(dataset['timeouts']))
    print ('max_path_length: ', env.max_episode_steps)
    print ('all keys: ', dataset.keys())

    episode_nu = 0
    for i in range(N):
        done_bool = bool(dataset['terminals'][i])
        if use_timeouts:
            final_timestep = dataset['timeouts'][i]
        else:
            final_timestep = (episode_step == env.max_episode_steps - 1)

        if final_timestep:
            if episode_step < 3:
                continue
        for k in dataset:
            if 'metadata' in k: continue
            data_[k].append(dataset[k][i])    
        # #------------------------------------------------------------------------------#
        if done_bool or final_timestep:   
            episode_nu += 1         
            episode_step = 0
            episode_data = {}
            for k in data_:
                episode_data[k] = np.array(data_[k])
            episode_data = discounted_episodic_scaled_return(episode_data, discount_array, returns_scale)
            if len(np.array(episode_data['observations'])) > 1001  or len(np.array(episode_data['observations'])) < 10 :
                gg=1
            if 'maze2d' in env.name or 'pen' in env.name or 'antmaze' in env.name or 'kitchen' in env.name or 'hammer' in env.name or 'door' in env.name or 'relocate' in env.name:
                episode_data = process_maze2d_episode(episode_data)
            if ('maze2d' in env.name or 'antmaze' in env.name) and reward_shaping==1:
                episode_data = shape_rewards_for_episode(episode_data, 0.99, 1.0)
            if 'maze2d' in env.name:
                episode_data = distance_to_target_goal_maze2d(episode_data)
            yield episode_data
            data_ = collections.defaultdict(list)
        episode_step += 1


        ttt=0



def process_maze2d_episode(episode):
    assert 'next_observations' not in episode
    length = len(episode['observations'])
    next_observations = episode['observations'][1:].copy()
    for key, val in episode.items():
        if isinstance(val, (np.ndarray, list)):
            episode[key] = val[:-1]
        else:
            episode[key] = val 
    episode['next_observations'] = next_observations
    return episode

def distance_to_target_goal_maze2d(episode):
    target_goal = np.array([6, 6], dtype=np.float32)
    obs_unnormalized = episode['observations'][:, :2]
    distance = target_goal - obs_unnormalized
    episode['distance_to_target_goal'] = distance
    return episode

def discounted_episodic_scaled_return(episode_data, discount_array, returns_scale):
    r = np.asarray(episode_data['rewards'], dtype=np.float32)
    d = np.asarray(discount_array[:len(r)], dtype=np.float32).reshape(-1)   # [1, γ, γ^2, ...]
    rtg = np.convolve(r[::-1], d, mode='full')[:len(r)][::-1]
    episode_data['scaled_RTG'] = rtg / float(returns_scale)
    episode_data['RTG'] = rtg
    # rtg = np.empty_like(r, dtype=np.float32)
    # acc = 0.0
    # for t in range(len(r)-1, -1, -1):
    #     acc = r[t] + discount_array[1] * acc  # if discount_array[1] == γ
    #     rtg[t] = acc
    # rtg = rtg / returns_scale
    return episode_data

def shape_rewards_for_episode(episode_data, gamma, alpha=1.0):
    """
    Applies potential-based reward shaping to a single episode dictionary.
    """
    # In some datasets, the goal is in 'infos/goal'
    if 'infos/goal' in episode_data:
        goals = episode_data['infos/goal']
    else:
        # Fallback for environments without explicit goals (though shaping is less common here)
        return episode_data

    # --- 1. Calculate Potential (Φ) for this episode ---
    # Positions: prefer infos/qpos over observations
    if 'infos/qpos' in episode_data:
        positions = episode_data['infos/qpos'][:, :2]
    else:
        positions = episode_data['observations'][:, :2]
    target_goals = goals[:, :2]
    potentials = -alpha * np.linalg.norm(positions - target_goals, axis=1)

    # --- 2. Calculate the Change in Potential ---
    potentials_current = potentials[:-1]
    potentials_next = potentials[1:]
    potential_change = gamma * potentials_next - potentials_current
    
    # --- 3. Apply the Shaping ---
    # The last reward in the episode remains unchanged
    episode_data['rewards'][:-1] += potential_change
    
    return episode_data








