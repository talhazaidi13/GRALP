from collections import namedtuple
import numpy as np
import torch, pickle
import d4rl

from .preprocessing import get_preprocess_fn
from .d4rl import load_environment, sequence_dataset
from .normalization import DatasetNormalizer
from .buffer import ReplayBuffer

RewardBatch = namedtuple('Batch', 'observations next_observations actions rtg')
Batch = namedtuple('Batch', 'observations next_observations actions')
ValueBatch = namedtuple('ValueBatch', 'trajectories conditions values')
Batch_pos_neg = namedtuple('Batch_pos_neg', 'observations next_observations actions rtg positives negatives positive_rew negative_rew positive_actions negative_actions conditions')
Batch_no_cotrast = namedtuple('Batch_no_cotrast', 'observations next_observations actions rtg conditions')
Batch_no_cotrast_raw_reward = namedtuple('Batch_no_cotrast_raw_reward', 'observations next_observations actions raw_rewards rtg conditions')
Batch_no_cotrast_with_goal = namedtuple('Batch_no_cotrast_with_goal', 'observations next_observations actions rtg conditions goal')
from torch.utils.data import DataLoader, Subset
import numpy as np
from sklearn.model_selection import train_test_split
import gym, d4rl, os, sys, pickle
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

class SequenceDataset_v1(torch.utils.data.Dataset):
    def __init__(self, 
                env_name,
                normalizer='GaussianNormalizer',
                use_normalizer=True,
                preprocess_fns=[], 
                horizon=16, 
                max_n_episodes=20000, 
                termination_penalty=0, 
                use_padding=False, 
                discount=0.99, 
                returns_scale=1000, 
                include_returns=True,
                load_path=None,
                subbatchsize = 32,
                contrastive_data = False,
                sequence_contrastive_states= True,
                raw_rewards = False,
                normed_actions = False,
                goal_flag = False,
                reward_shaping=0,
                radius=0.15,):
        self.contrastive_data = contrastive_data
        self.subbatchsize = subbatchsize # forcontrastiveloss, 8*horizon=16 = 128
        self.preprocess_fn = get_preprocess_fn(preprocess_fns, env_name)
        self.env_name = env_name
        self.env = env = load_environment(env_name)
        # get the max length of env
        self.max_path_length =  1001 if 'antmaze' in env_name else 1000
        self.horizon = horizon
        self.returns_scale = returns_scale
        self.discount = discount
        self.discounts = self.discount ** np.arange(self.max_path_length, dtype=np.float32)[:, None]
        self.use_padding = use_padding
        self.include_returns = include_returns
        self.sequence_contrastive_states= sequence_contrastive_states
        self.raw_rewards = raw_rewards
        self.radius = radius
        self.normed_actions = normed_actions
        self.reward_shaping = reward_shaping
        itr = sequence_dataset(env, self.preprocess_fn, load_path=load_path, reward_shaping=reward_shaping, radius=self.radius, 
                               discount_array=self.discounts, returns_scale=self.returns_scale  )
        self.ep_returns_all, self.ep_returns_all_2 = [],[]
        self.goal_flag = goal_flag

        fields = ReplayBuffer(max_n_episodes, self.max_path_length, termination_penalty)
        for i, episode in enumerate(itr):
            fields.add_path(episode)
        fields.finalize()
        #------------------------------------------------------------------------------
        self.normalizer = DatasetNormalizer(fields, normalizer, path_lengths=fields['path_lengths'])
        self.fields = fields
        self.n_episodes = fields.n_episodes
        self.path_lengths = fields.path_lengths
        
        for i in range(fields.n_episodes):
            self.ep_returns_all.append(fields.rewards[i].sum())
            disc_rewards = self.fields.rewards[i][0:self.path_lengths[i]]    # end is the horizon end point
            discounts_test = self.discounts[:self.path_lengths[i]]
            returns_test = np.cumsum(discounts_test * disc_rewards[::-1])[::-1][:self.horizon]
            self.ep_returns_all_2.append(np.max(returns_test))

        if use_normalizer:
                self.normalize(keys=["observations", "next_observations", "actions","RTG", "rewards"])

        self.indices = self.make_indices_2(fields.path_lengths, horizon)
        self.total_nums = len(self.indices)

        self.observation_dim = fields.observations.shape[-1]
        self.action_dim = fields.actions.shape[-1]

        print(fields)


    def re_make_indices(self):
        self.positive_indices, self.indices, self.negative_indices = self.make_indices(self.fields.path_lengths, self.horizon)

    def normalize(self, keys):
        for key in keys:
            # print(f"{key}: Means={self.normalizer.normalizers[key].means}, Stds={self.normalizer.normalizers[key].stds}") 
            array = self.fields[key].reshape(self.n_episodes*self.max_path_length, -1)
            normed = self.normalizer(array, key)
            self.fields[f'normed_{key}'] = normed.reshape(self.n_episodes, self.max_path_length, -1)

    def sort_by_values(self, to_sort, values):
        inds = np.argsort(values)[::-1]
        to_sort = to_sort[inds]
        values = values[inds]
        return to_sort, values
    
    def make_indices_2(self, path_lengths, horizon):
        '''
            makes indices for sampling from dataset;
            each index maps to a datapoint
        '''
        indices = []
        for i, path_length in enumerate(path_lengths):
            max_start = min(path_length - 1, self.max_path_length - horizon )
            # max_start = min(path_length - 1, self.max_path_length - horizon)
            if not self.use_padding:
                max_start = min(max_start, path_length - horizon)
            for start in range(max_start):
                end = start + horizon
                indices.append((i, start , end))
        return  np.array(indices)


    def get_conditions(self, observations):
        '''
            condition on current observation for planning
        '''
        return {0: observations[0]}

    def __len__(self):
        return self.indices.shape[0]

    def __getitem__(self, idx, eps=1e-4):
        path_ind, start, end = self.indices[idx]
        length = self.path_lengths[path_ind]
        observations = self.fields.normed_observations[path_ind, start:end]
        next_observations = self.fields.normed_next_observations[path_ind, start:end]
        if self.normed_actions:
            actions = self.fields.normed_actions[path_ind, start:end]
        else:
            actions = self.fields.actions[path_ind, start:end]
        conditions = self.get_conditions(observations) 
        reards_raw = self.fields.rewards[path_ind, start:end]
        goal = self.fields['normed_infos/goal'][path_ind, start:end] if (self.goal_flag  and 'infos/goal' in self.fields.keys) else []

        if self.include_returns:
            returns = self.fields['scaled_RTG'][path_ind, start:end]
            if self.raw_rewards:
                return Batch_no_cotrast_raw_reward(observations, next_observations, actions, reards_raw, returns, conditions)
            else:
                return Batch_no_cotrast_with_goal(observations, next_observations, actions, returns,conditions, goal)
        return Batch(observations, next_observations, actions, 0, conditions)    

    def build_transformer_sequences(self, skill_encoder, context_length, gamma, device='cuda', z_path=None, stride=1):
        all_rtg_sequences, all_state_sequences, all_skill_sequences, all_z = [], [], [], []
        skill_encoder.to(device)
        skill_encoder.eval()
        print(f"Processing {self.n_episodes} episodes into Transformer sequences...")
        final_dataset_list, final_dataset_list1 = [], []
        # Loop through each episode that has been parsed and stored in self.fields
        for i in range(self.n_episodes):
            # Use the actual length of the episode, not the max padded length
            ep_len = self.path_lengths[i]
            # Get the low-level data for this episode
            obs = self.fields.normed_observations[i, :ep_len]
            actions = self.fields.actions[i, :ep_len]
            rewards = self.fields.rewards[i, :ep_len]
            num_skills = ep_len // self.horizon
            stride = stride # self.horizon
            if ep_len <= self.horizon:
                starts = [0]
            else:
                starts = list(range(0, ep_len - self.horizon + 1, stride))  #[0, 12, 24, 36, 48]
                if (ep_len % self.horizon) != 0:
                    last_start = ep_len - self.horizon      # 65-12=53           # end-aligned last chunk
                    if len(starts) == 0 or starts[-1] != last_start:
                        starts.append(last_start)         # [0, 12, 24, 36, 48, 53]
            num_skills = len(starts)
            if num_skills == 0:
                continue #     
            # --- Build fixed-length chunks (no padding inside the encoder) ---
            obs_chunks    = np.stack([obs[s:s+self.horizon]     for s in starts], axis=0)       # (K, H, obs_dim)
            action_chunks = np.stack([actions[s:s+self.horizon] for s in starts], axis=0)       # (K, H, act_dim)
            reward_chunks = np.stack([rewards[s:s+self.horizon] for s in starts], axis=0)       # (K, H)
            high_level_states = np.stack([obs[s] for s in starts], axis=0)    
            # Encode actions into skills for this episode
            with torch.no_grad():
                obs_chunks_tensor = torch.from_numpy(obs_chunks).float().to(device)
                action_chunks_tensor = torch.from_numpy(action_chunks).float().to(device)
                z_means, _ = skill_encoder(obs_chunks_tensor, action_chunks_tensor)
                high_level_skills_z = z_means.squeeze(1).cpu().numpy()
                high_level_rtg = self.fields.normed_RTG[i, :ep_len][np.array(starts)].reshape(-1, 1)   # Z-normalized RTG
                high_level_rtg1 = self.fields.scaled_RTG[i, :ep_len][np.array(starts)].reshape(-1, 1)  # scaled RTG
            if self.goal_flag:
                if 'maze2d' in self.env_name:
                    goals_norm = self.fields['normed_infos/goal'][i, :ep_len]            # (T, goal_dim)
                goal_tokens = np.stack([goals_norm[s] for s in starts], axis=0)      # (K, goal_dim)
                high_level_states = np.concatenate([high_level_states, goal_tokens], axis=-1)  # (K, state_dim+goal_dim)
            # Create overlapping subsequences of length K from this episode
            if num_skills < context_length:
                # make exactly ONE right-padded window (the last suffix)
                real_len = num_skills
                pad_len  = context_length - real_len
                padded_rtgs   = np.concatenate([high_level_rtg[:real_len], np.zeros((pad_len, 1), dtype=np.float32)], axis=0)
                padded_states = np.concatenate([high_level_states[:real_len], np.zeros((pad_len, high_level_states.shape[1]), dtype=np.float32)], axis=0)
                padded_skills = np.concatenate([high_level_skills_z[:real_len], np.zeros((pad_len, high_level_skills_z.shape[1]), dtype=np.float32)],axis=0)
                attention_mask = np.concatenate([np.ones(real_len, dtype=np.int64), np.zeros(pad_len, dtype=np.int64)],axis=0)
                sequence_dict = {
                    'rtgs': padded_rtgs,
                    'states': padded_states,
                    'skills': padded_skills,
                    'attention_mask': attention_mask # Don't forget the mask!
                }
                final_dataset_list.append(sequence_dict)
            else:
                for j in range(num_skills - context_length + 1):
                    start, end = j, j + context_length  
                    attention_mask = np.ones(context_length)          
                    sequence_dict = {
                        'rtgs': high_level_rtg[start:end],  # rtg for each horizon
                        'states': high_level_states[start:end], # s0 for each horizon
                        'skills': high_level_skills_z[start:end], # z for each horizon
                        'attention_mask': attention_mask 
                        }
                    final_dataset_list.append(sequence_dict)
                    sequence_dict1 = {'rtgs': high_level_rtg1[start:end]}  # rtg for each horizon
                    final_dataset_list1.append(sequence_dict1)
            all_z.append(high_level_skills_z)
        if z_path is not None and len(all_z) > 0:
            z_all = np.concatenate(all_z, axis=0)  # (N_real, Dz) — no pads included
            mean_z = z_all.mean(axis=0)
            std_z  = z_all.std(axis=0)
            std_z  = np.clip(std_z, 1e-6, None)
            print(f"Computed skill stats: mean_z {mean_z}, std_z shape {std_z.shape}")
            stats_path = f'{z_path}/skill_stats.pkl'
            os.makedirs(os.path.dirname(stats_path), exist_ok=True)
            with open(stats_path, 'wb') as f:
                pickle.dump({'mean_z': mean_z, 'std_z': std_z}, f)
            print(f"Saved skill stats to {stats_path}")
        print("✅ Transformer dataset creation complete.")
        all_rtgs = np.concatenate([np.asarray(step['rtgs']).ravel()for step in final_dataset_list])
        print(f"Total number of training sequences: {len(final_dataset_list)}")
        print("*************************************************************.")
        print("REWARD SCALING INFO")
        print(f"REWARD SCALING: Z-NORM MAX REWARD SCALE: {np.max(all_rtgs)}")
        print(f"REWARD SCALING: Z-NORM MIN REWARD SCALE:{np.min(all_rtgs)}")
        print(f"REWARD SCALING: Z-NORM MEAN REWARD SCALE: {np.mean(all_rtgs)}")
        print(f"REWARD SCALING: Z-NORM STD REWARD SCALE: {np.std(all_rtgs)}")
        print("*************************************************************.")
        all_rtgs1 = np.concatenate([np.asarray(step['rtgs']).ravel()for step in final_dataset_list1])
        print(f"Total number of training sequences: {len(final_dataset_list1)}")
        print("*************************************************************.")
        print(f"REWARD SCALING INFO")
        print(f"REWARD SCALING: SCALED MAX NORMALIZED RTG: {np.max(all_rtgs1)}")
        print(f"REWARD SCALING: SCALED MIN NORMALIZED RTG:{np.min(all_rtgs1)}")
        print(f"REWARD SCALING: SCALED MEAN NORMALIZED RTG: {np.mean(all_rtgs1)}")
        print(f"REWARD SCALING: SCALED STD NORMALIZED RTG: {np.std(all_rtgs1)}")
        return final_dataset_list
    


class SplittableSequenceDataset(SequenceDataset_v1):
    def get_train_test_split(self, test_size=0.2, random_state=42):
        """
        Creates training and testing splits based on indices.
        
        Args:
            test_size: Proportion of the dataset to include in the test split
            random_state: Random seed for reproducibility
            
        Returns:
            train_dataset, test_dataset: Subset objects containing the split data
        """
        # Get all indices
        index_indices = np.arange(len(self.indices))
        
        # Split these meta-indices into train and test sets
        train_index_indices, test_index_indices = train_test_split(
            index_indices, test_size=test_size, random_state=random_state
        )
        
        # Create Subset objects
        # Create SequenceSubset objects
        train_dataset = SequenceSubset(self, train_index_indices)
        test_dataset = SequenceSubset(self, test_index_indices)
        
        return train_dataset, test_dataset



# Create train and test dataset classes that inherit from the base Dataset
class SequenceSubset(torch.utils.data.Dataset):
    def __init__(self, parent_dataset, subset_indices):
        """
        Creates a subset of the original dataset using specific row indices 
        from the parent dataset's self.indices array.
        
        Args:
            parent_dataset: The original SequenceDatasetV2 instance
            subset_indices: Indices to use from the parent_dataset.indices array
        """
        self.parent = parent_dataset
        self.subset_indices = subset_indices
        
    def __len__(self):
        return len(self.subset_indices)
    
    def __getitem__(self, idx):
        # Get the original index from our subset mapping
        original_idx = self.subset_indices[idx]
        # Use the parent's __getitem__ with the original index
        return self.parent.__getitem__(original_idx)



class EncodedDAEDataset(torch.utils.data.Dataset):
    def __init__(self, filepath, goal_flag=0):
        """
        Args:
            filepath (str): Path to the pickled file containing the list of dictionaries.
        """
        print(f"Loading encoded dataset from {filepath}...")
        with open(filepath, 'rb') as f:
            self.data_list_of_dicts = pickle.load(f)
        print(f"Loaded {len(self.data_list_of_dicts)} samples.")
        self.goal_flag = goal_flag
    def __len__(self):
        return len(self.data_list_of_dicts)

    def __getitem__(self, idx):
        item_dict = self.data_list_of_dicts[idx]
        
        s0 = torch.from_numpy(item_dict['s0']).float()
        z = torch.from_numpy(item_dict['z']).float()
        # Ensure 'return' is a scalar or a tensor with a consistent shape for batching
        # If item_dict['return'] is like np.array([value]), .item() makes it scalar
        # If it's already scalar, .item() might not be needed or could error if it's not a 1-element array
        rtg = torch.tensor(item_dict['return'].item() if item_dict['return'].ndim > 0 else item_dict['return'], dtype=torch.float32)
        sT = torch.from_numpy(item_dict['sT']).float()
        if self.goal_flag:
            goal = torch.from_numpy(item_dict['goal']).float()
            return s0, goal, z, rtg, sT
        else:
            return s0, z, rtg, sT


class EncodedTransformerDataset(torch.utils.data.Dataset):
    def __init__(self, filepath, goal_flag=0):
        """
        Args:
            filepath (str): Path to the pickled file containing the list of dictionaries.
        """
        print(f"Loading encoded dataset from {filepath}...")
        with open(filepath, 'rb') as f:
            self.data_list_of_dicts = pickle.load(f)
        print(f"Loaded {len(self.data_list_of_dicts)} samples.")
        self.goal_flag = goal_flag
    def __len__(self):
        return len(self.data_list_of_dicts)

    def __getitem__(self, idx):
        item_dict = self.data_list_of_dicts[idx]
        
        rtg = torch.from_numpy(item_dict['rtgs']).float()
        states = torch.from_numpy(item_dict['states']).float()
        z = torch.from_numpy(item_dict['skills']).float()
        attention_mask = torch.from_numpy(item_dict['attention_mask']).float()
        # Ensure 'return' is a scalar or a tensor with a consistent shape for batching
        # If item_dict['return'] is like np.array([value]), .item() makes it scalar
        # If it's already scalar, .item() might not be needed or could error if it's not a 1-element array
        return rtg, states, z, attention_mask

