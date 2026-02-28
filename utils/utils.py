#import d4rl
import gym
from torch.utils.data import TensorDataset
from torch.utils.data.dataloader import DataLoader
import torch.distributions.normal as Normal
import ipdb, random, pickle, torch, time, math, os
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.manifold import TSNE
import torch, torch.distributed as dist
from types import SimpleNamespace
import comet_ml
import socket, time, os
import os, tempfile, torch

def safe_create_experiment(api_key: str,
                           project: str,
                           rank: int,
                           world: int,
                           max_retry: int = 3):
    """
    Always returns a valid experiment-like object.
    Rank-0 tries to create the real Comet Experiment. Other ranks attach later.
    When Comet is unreachable we fall back to a dummy logger.
    """
    def _dummy():
        # minimal object with the comet API you actually use
        def _nop(*a, **k): return None
        return SimpleNamespace(
            log_metric=_nop, log_metrics=_nop, log_parameter=_nop,
            log_parameters=_nop, log_text=_nop, set_name=_nop,
            get_key=lambda: f"offline-{socket.gethostname()}-{int(time.time())}"
        )

    # ---- rank-0: try to open an online run ---------------------------------
    if rank == 0:
        for attempt in range(1, max_retry + 1):
            try:
                exp = comet_ml.Experiment(api_key=api_key,
                                          project_name=project,
                                          auto_output_logging="simple",
                                          log_env_details=True)
                key = exp.get_key()
                break
            except Exception as e:
                print(f"[rank-0] Comet connection failed (attempt {attempt}/{max_retry}): {e}")
                if attempt == max_retry:
                    exp = _dummy()
                    key = exp.get_key()
    else:
        exp, key = None, None

    # ---- broadcast the key so *all* ranks know it --------------------------
    key_list = [key]
    if world > 1:
        dist.broadcast_object_list(key_list, src=0)
    key = key_list[0]

    # ---- non-zero ranks attach (or create dummy if offline) ----------------
    if rank != 0:
        try:
            exp = comet_ml.ExistingExperiment(
                previous_experiment=key,
                api_key=api_key,
                project_name=project,
                auto_output_logging="simple"
            )
        except Exception:
            exp = _dummy()

    return exp
# ---------------------------------------------------------------------------

def all_gather_cat(t: torch.Tensor, ddp_active_flag: bool) -> torch.Tensor:
    """
    Gather a same-shaped tensor from every rank and concatenate on dim-0.
    Works on CPU or GPU tensors.
    """
    if not ddp_active_flag:
        return t
    if not torch.is_tensor(t):       # defensive guard
        raise TypeError("all_gather_cat expects a Tensor")
    bufs = [torch.zeros_like(t) for _ in range(dist.get_world_size())]
    dist.all_gather(bufs, t.contiguous())
    return torch.cat(bufs, dim=0)


# ---------------------------------------------------------------------------
def save_checkpoint(policy, path, epoch=None):
    # If using DistributedDataParallel, access the underlying module
    model = policy.module if hasattr(policy, 'module') else policy
    checkpoint = {
        'epoch': epoch,
        'ema_model': model.ema_model.state_dict(),
        'feasible_generator': model.feasible_generator.state_dict(),
        'feasible_generator_optimizer': model.feasible_generator_optimizer.state_dict(),
        'feasible_generator_lr_scheduler': model.feasible_generator_lr_scheduler.state_dict(),
        'planner': model.planner.state_dict(),
        'planner_optimizer': model.planner_optimizer.state_dict(),
        'planner_lr_scheduler': model.planner_lr_scheduler.state_dict(),
        'actor': model.actor.state_dict(),
        'actor_optimizer': model.actor_optimizer.state_dict(),
        'actor_lr_scheduler': model.actor_lr_scheduler.state_dict(),
        'critic': model.critic.state_dict(),
        'critic_optimizer': model.critic_optimizer.state_dict(),
        'critic_lr_scheduler': model.critic_lr_scheduler.state_dict(),
        'latentcritic': model.latentcritic.state_dict(),
        'latentcritic_optimizer': model.latentcritic_optimizer.state_dict(),
        'latentcritic_lr_scheduler': model.latentcritic_lr_scheduler.state_dict(),
        'ema_model': model.ema_model.state_dict(),
        'global_step': model.global_step,
        'latent_mix_lambda': model.latent_mix_lambda,
        'kl_block_streak': model.kl_block_streak,
        'kl_val': model.kl_val,
        # Add other necessary components if needed
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(checkpoint, path)
    print(f"Checkpoint saved to {path}")


def load_checkpoint(policy, path, device):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location=device)
    model = policy.module if hasattr(policy, 'module') else policy
    model.ema_model.load_state_dict(checkpoint['ema_model'])
    model.feasible_generator.load_state_dict(checkpoint['feasible_generator'])
    model.feasible_generator_optimizer.load_state_dict(checkpoint['feasible_generator_optimizer'])
    model.feasible_generator_lr_scheduler.load_state_dict(checkpoint['feasible_generator_lr_scheduler'])
    model.planner.load_state_dict(checkpoint['planner'])
    model.planner_optimizer.load_state_dict(checkpoint['planner_optimizer'])
    model.planner_lr_scheduler.load_state_dict(checkpoint['planner_lr_scheduler'])
    model.actor.load_state_dict(checkpoint['actor'])
    model.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
    model.actor_lr_scheduler.load_state_dict(checkpoint['actor_lr_scheduler'])
    model.critic.load_state_dict(checkpoint['critic'])
    model.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
    model.critic_lr_scheduler.load_state_dict(checkpoint['critic_lr_scheduler'])
    model.latentcritic.load_state_dict(checkpoint['latentcritic'])
    model.latentcritic_optimizer.load_state_dict(checkpoint['latentcritic_optimizer'])
    model.latentcritic_lr_scheduler.load_state_dict(checkpoint['latentcritic_lr_scheduler'])
    model.ema_model.load_state_dict(checkpoint['ema_model'])
    model.global_step = checkpoint.get('global_step', 0)
    model.latent_mix_lambda = checkpoint.get('latent_mix_lambda', 1.0)
    model.kl_block_streak = checkpoint.get('kl_block_streak', 0)
    model.kl_val = checkpoint.get('kl_val', torch.tensor(0.0))
    print(f"Checkpoint loaded from {path}")

def save(self, path=None):
        if path is None:
            path = "./model/checkpoint.pth"
        prefix = os.path.dirname(path)
        if not os.path.exists(prefix):
            os.makedirs(prefix)
        torch.save({
            'planner': self.planner.state_dict(),
            'ema_model': self.ema_model.state_dict(),
            'transformer': self.feasible_generator.state_dict(),
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
        }, path)

def load(self, path=None):
        if path is None:
            path = "./model/checkpoint.pth"
        checkpoint = torch.load(path, map_location=self.device)
        self.planner.load_state_dict(checkpoint['planner'])
        self.ema_model.load_state_dict(checkpoint['ema_model'])
        self.feasible_generator.load_state_dict(checkpoint['transformer'])
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])


# def save_transformer_checkpoint(transformer_model, optimizer, epoch_or_step, filepath):
#     os.makedirs(os.path.dirname(filepath), exist_ok=True)
#     checkpoint = {
#         'epoch_or_step': epoch_or_step,
#         'transformer_state_dict': transformer_model.state_dict(),
#         'optimizer_state_dict': optimizer.state_dict(),
#     }
#     torch.save(checkpoint, filepath)
#     print(f"           Checkpoint saved successfully to {filepath}")

# def load_transformer_checkpoint(filepath, transformer_model, optimizer, device='cuda:0'):
#     if not os.path.isfile(filepath):
#         raise FileNotFoundError(f"Checkpoint not found: {filepath}")
#     print(f"Loading checkpoint from {filepath}...")
#     checkpoint = torch.load(filepath, map_location=device)
#     transformer_model.load_state_dict(checkpoint['transformer_state_dict'])
#     try:
#         optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#     except:
#         print("Transformer Optimizer not loaded. Continuing...")
#         pass
#     start_epoch_or_step = checkpoint.get('epoch_or_step', 0)
#     print(f"Transformer Checkpoint loaded. Resuming from epoch/step {start_epoch_or_step + 1}")
#     return start_epoch_or_step + 1 # Return next epoch/step to start from    



def _filter_transformer_state_dict(sd):
    """Drop attached critics (Q1/Q2/V and any targets) from a transformer's state_dict."""
    drop_prefixes = ('Q1_targ.', 'Q2_targ.', 'V_targ.')
    return {k: v for k, v in sd.items() if not k.startswith(drop_prefixes)}
def save_transformer_checkpoint(transformer_model, optimizer, epoch_or_step, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    # filter out critics if they are attached as submodules
    raw_sd = transformer_model.state_dict()
    trf_sd = _filter_transformer_state_dict(raw_sd)
    checkpoint = {
        'epoch_or_step': epoch_or_step,
        'transformer_state_dict': trf_sd,
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(checkpoint, filepath)
    print(f"           Checkpoint saved successfully to {filepath}")
def load_transformer_checkpoint(filepath, transformer_model, optimizer, device='cuda:0', strict=False):
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Checkpoint not found: {filepath}")
    print(f"Loading checkpoint from {filepath}...")
    checkpoint = torch.load(filepath, map_location=device)
    # Load only what’s present; ignore missing critics etc.
    missing, unexpected = transformer_model.load_state_dict(
        checkpoint['transformer_state_dict'], strict=strict
    )
    if missing or unexpected:
        print(f"[load_transformer] missing keys: {missing}")
        print(f"[load_transformer] unexpected keys: {unexpected}")
    try:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    except Exception as e:
        print(f"Transformer Optimizer not loaded ({e}). Continuing...")

    start_epoch_or_step = checkpoint.get('epoch_or_step', 0)
    print(f"Transformer Checkpoint loaded. Resuming from epoch/step {start_epoch_or_step + 1}")
    return start_epoch_or_step + 1
# def _to_cpu(sd):  # tensor -> cpu
#     return {k: (v.cpu() if torch.is_tensor(v) else v) for k,v in sd.items()}
# def save_critics_min(filepath, Q1, Q2, V):
#     """Smallest useful critics file: only online Q1/Q2/V (CPU, legacy, atomic)."""
#     os.makedirs(os.path.dirname(filepath), exist_ok=True)
#     ckpt = {"Q1": Q1.state_dict(),
#             "Q2": Q2.state_dict(),
#             "V":  V.state_dict()}
#     dirp = os.path.dirname(filepath)
#     torch.save(ckpt, filepath)
# def load_critics_checkpoint(filepath, Q1, Q2, V, Q1_targ, Q2_targ, V_targ,
#                             optQ1=None, optQ2=None, optV=None, device='cuda:0'):
#     print(f"[critics] loading from {filepath}")
#     ckpt = torch.load(filepath, map_location=device)
#     Q1.load_state_dict(ckpt['Q1']);  Q2.load_state_dict(ckpt['Q2']);  V.load_state_dict(ckpt['V'])
#     print("[critics] loaded.")


def save_checkpoint_DAE(dae_model, optimizer, epoch_or_step, filepath):
    # Ensure the directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    checkpoint = {
        'epoch_or_step': epoch_or_step,
        'encoder_state_dict': dae_model.encoder.state_dict(),
        'decoder_state_dict': dae_model.decoder.state_dict(),  # This is your diff_planner
        'prior_state_dict': dae_model.prior.state_dict(),
        'ema_model_state_dict': dae_model.ema_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        # You can add other things like learning rate scheduler state if you use one
        # 'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
    }
    torch.save(checkpoint, filepath)
    print(f"           Checkpoint saved successfully to {filepath}")

def load_checkpoint_DAE(filepath, dae_model, optimizer=None, device='cuda:0'):
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Checkpoint not found: {filepath}")
    print(f"Loading checkpoint from {filepath}...")
    checkpoint = torch.load(filepath, map_location=device)
    dae_model.encoder.load_state_dict(checkpoint['encoder_state_dict'])
    dae_model.decoder.load_state_dict(checkpoint['decoder_state_dict']) # diff_planner
    dae_model.prior.load_state_dict(checkpoint['prior_state_dict'])
    dae_model.ema_model.load_state_dict(checkpoint['ema_model_state_dict'])
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("Optimizer state loaded.")
    start_epoch_or_step = checkpoint.get('epoch_or_step', 0)
    print(f"Checkpoint loaded. Resuming from epoch/step {start_epoch_or_step + 1}")
    return start_epoch_or_step + 1 # Return next epoch/step to start from    


import torch
import os

def save_q_a_checkpoint(q_a, optimizer, epoch_or_step, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    checkpoint = {
        'epoch_or_step': epoch_or_step,
        'q_a_state_dict': q_a.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(checkpoint, filepath)
    # print(f"              Q_A Checkpoint saved successfully to {filepath} at step {epoch_or_step}")

def load_q_a_checkpoint(filepath, q_a, optimizer, device='cuda:0'):
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Checkpoint not found: {filepath}")
    print(f"Loading Q_A checkpoint from {filepath}...")
    checkpoint = torch.load(filepath, map_location=device)  
    q_a.load_state_dict(checkpoint['q_a_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch_or_step = checkpoint.get('epoch_or_step', 0)
    print(f"Q_A Checkpoint loaded. Resuming Q_A training from step {start_epoch_or_step + 1}")
    return start_epoch_or_step + 1 # Return next epoch/step to start from





def save_iql_checkpoint(iql_trainer_instance, epoch_or_step, filepath):
    # Ensure the directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    checkpoint = {
        'epoch_or_step': epoch_or_step,   
        # Value Function
        'value_net_state_dict': iql_trainer_instance.value_net.state_dict(),
        'target_value_net_state_dict': iql_trainer_instance.target_value_net.state_dict(),
        'optim_v_state_dict': iql_trainer_instance.optim_v.state_dict(), 
        # Q Functions
        'q_net1_state_dict': iql_trainer_instance.q_net1.state_dict(),
        'q_net2_state_dict': iql_trainer_instance.q_net2.state_dict(),
        'target_q_net1_state_dict': iql_trainer_instance.target_q_net1.state_dict(),
        'target_q_net2_state_dict': iql_trainer_instance.target_q_net2.state_dict(),
        'optim_q_state_dict': iql_trainer_instance.optim_q.state_dict(), 
        # Policy Network
        'policy_net_state_dict': iql_trainer_instance.policy_net.state_dict(),
        'optim_pi_state_dict': iql_trainer_instance.optim_pi.state_dict(),
        # You can add other relevant info like hyperparameters if needed
        'iql_beta': iql_trainer_instance.iql_beta,
        'iql_tau_expectile': iql_trainer_instance.iql_tau_expectile,
    }
    torch.save(checkpoint, filepath)
    print(f"              IQL Checkpoint saved successfully to {filepath} at step {epoch_or_step}")


def load_iql_checkpoint(filepath, iql_trainer_instance, device='cuda:0'):
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Checkpoint not found: {filepath}")
    print(f"Loading IQL checkpoint from {filepath}...")
    checkpoint = torch.load(filepath, map_location=device)
    # Load Value Function
    iql_trainer_instance.value_net.load_state_dict(checkpoint['value_net_state_dict'])
    iql_trainer_instance.target_value_net.load_state_dict(checkpoint['target_value_net_state_dict'])
    iql_trainer_instance.optim_v.load_state_dict(checkpoint['optim_v_state_dict'])
    # Load Q Functions
    iql_trainer_instance.q_net1.load_state_dict(checkpoint['q_net1_state_dict'])
    iql_trainer_instance.q_net2.load_state_dict(checkpoint['q_net2_state_dict'])
    iql_trainer_instance.target_q_net1.load_state_dict(checkpoint['target_q_net1_state_dict'])
    iql_trainer_instance.target_q_net2.load_state_dict(checkpoint['target_q_net2_state_dict'])
    iql_trainer_instance.optim_q.load_state_dict(checkpoint['optim_q_state_dict'])
    # Load Policy Network
    iql_trainer_instance.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
    iql_trainer_instance.optim_pi.load_state_dict(checkpoint['optim_pi_state_dict'])
    start_epoch_or_step = checkpoint.get('epoch_or_step', 0)
    print(f"IQL Checkpoint loaded. Resuming IQL training from step {start_epoch_or_step + 1}")
    return start_epoch_or_step + 1 # Return next epoch/step to start from

    
    
###########################################################################################
#############   Policy Modules  ################

def alpha_schedule(epoch, start_epoch=0, end_epoch=10000, start_value=0.01, end_value=0.1):
    if epoch < start_epoch:
        return start_value
    elif epoch > end_epoch:
        return end_value
    else:
        progress = (epoch - start_epoch) / (end_epoch - start_epoch)
        return start_value + progress * (end_value - start_value)

def set_seed( seed):
    """Set seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def render_reference(renderer, trajectories, state_mean, state_std):
    '''
        renders training points
    '''
    normed_observations = trajectories
    observations = normed_observations * state_std + state_mean

    savepath = os.path.join("./reference", f'_sample-reference.png')
    if not os.path.exists("./reference"):
        os.makedirs("./reference")
    renderer.composite(savepath, observations)

def tensor_reshape(tensor, reference=None, flatten=True, add_dim=False):
    if tensor is None:
        return None            
    if flatten:
        return tensor.view(tensor.shape[0] * tensor.shape[1], *tensor.shape[2:])
    else:
        shape = [reference.shape[0], reference.shape[1]]
        if add_dim:
            shape.append(1)
        shape.append(-1)  # Infer the last dimension
        return tensor.view(*shape)

def calc_metrics(scores, env, Dones_all=[0]):
    normalized = [env.get_normalized_score(s)*100 for s in scores]
    normalized_avrg = env.get_normalized_score(np.mean(scores))*100
    return {
        'max': np.max(scores),
        'min': np.min(scores),
        'avg': np.mean(scores),
        'std': np.std(scores),
        'norm_max': np.max(normalized),
        'norm_min': np.min(normalized),
        'norm_avg': normalized_avrg,
        'norm_std': np.std(normalized),
        'normalized': normalized,
        'dones': np.sum(Dones_all),
        'scores_all': scores,
    }
    

###########################################################################################









# ---- TSNE -----------------------------------------------------------------------

@torch.no_grad()  
def collect_latents(encoder, dataloader, device, max_pts=2000):
    encoder.eval()
    latents, labels = [], []
    for nu, batch in enumerate(dataloader):
        batch = batch_to_device(batch, device)
        states=batch[0] ; actions=batch[2]; returns=batch[3]
        z_mu, _ = encoder(states, actions)  # [B,1,z_dim]
        latents.append(z_mu.squeeze(1).cpu())
        labels.append(returns.cpu())              # or behaviour IDs
        if sum(len(x) for x in latents) >= max_pts:
            break
    Z = torch.cat(latents, 0).float().cpu().numpy()              # [N,z_dim]
    y = torch.cat(labels, 0).cpu().numpy().astype(np.float32)              # [N]
    return Z, y

def tsne_2d(latents, seed=42, perp=30):
    # Deeper inspection of the array
    print(f"Latents shape: {latents.shape}")
    print(f"Latents type: {type(latents)}")
    print(f"Latents dtype: {latents.dtype}")
    
    # Check if the array has object dtype elements (which can contain mixed types)
    if latents.dtype == np.dtype('O'):
        # Sample the types of individual elements
        for i in range(min(5, len(latents))):
            for j in range(min(5, latents.shape[1])):
                print(f"Element [{i},{j}] type: {type(latents[i,j])}")
    
    # Try to force conversion to detect any problematic values
    try:
        test = np.array(latents, dtype=np.float32)
        print("Conversion to float32 successful")
    except Exception as e:
        print(f"Conversion failed: {e}")
    
    latents = np.ascontiguousarray(latents, dtype=np.float32)
    tsne = TSNE(n_components=2, perplexity=perp, learning_rate="auto",
                init="pca", random_state=seed)
    return tsne.fit_transform(latents)

def plot_tsne(X2, y, title):
    fig, ax = plt.subplots(figsize=(5,5), dpi=120)
    # Put labels into ≤10 discrete bins if they are continuous returns
    if y.dtype.kind in "f":                      # float → returns
        bins = np.linspace(y.min(), y.max(), 11)
        labels = np.digitize(y, bins)
    else:                                        # already categorical
        labels = y
    sc = ax.scatter(X2[:,0], X2[:,1], c=labels,
                    cmap="tab10", s=12, alpha=.8)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title)
    return fig
# ---------------------------------------------------------------------------


def batch_to_device(batch, device='cuda:0'):
    vals = [
        to_device(getattr(batch, field), device)
        for field in batch._fields
    ]
    return type(batch)(*vals)

def to_device(x, device='cuda:0'):
	if torch.is_tensor(x):
		return x.to(device)
	elif type(x) is dict:
		return {k: to_device(v, device) for k, v in x.items()}
	else:
		raise RuntimeError(f'Unrecognized type in `to_device`: {type(x)}')

def reparameterize(mean, std):
    eps = torch.normal(torch.zeros(mean.size()).cuda(), torch.ones(mean.size()).cuda())
    return mean + std*eps

def stable_weighted_log_sum_exp(x,w,sum_dim):
    a = torch.min(x)
    ipdb.set_trace()

    weighted_sum = torch.sum(w * torch.exp(x - a),sum_dim)

    return a + torch.log(weighted_sum)

def chunks(obs,actions,H,stride):
    '''
    obs is a N x 4 array
    goals is a N x 2 array
    H is length of chunck
    stride is how far we move between chunks.  So if stride=H, chunks are non-overlapping.  If stride < H, they overlap
    '''
    
    obs_chunks = []
    action_chunks = []
    N = obs.shape[0]
    for i in range(N//stride - H):
        start_ind = i*stride
        end_ind = start_ind + H
        
        obs_chunk = torch.tensor(obs[start_ind:end_ind,:],dtype=torch.float32)

        action_chunk = torch.tensor(actions[start_ind:end_ind,:],dtype=torch.float32)
        
        loc_deltas = obs_chunk[1:,:2] - obs_chunk[:-1,:2] #Franka or Maze2d
        
        norms = np.linalg.norm(loc_deltas,axis=-1)
        #USE VALUE FOR THRESHOLD CONDITION BASED ON ENVIRONMENT
        if np.all(norms <= 0.8): #Antmaze large 0.8 medium 0.67 / Franka 0.23 mixed/complete 0.25 partial / Maze2d 0.22
            obs_chunks.append(obs_chunk)
            action_chunks.append(action_chunk)
        else:
            pass

    print('len(obs_chunks): ',len(obs_chunks))
    print('len(action_chunks): ',len(action_chunks))
            
    return torch.stack(obs_chunks),torch.stack(action_chunks)


def get_dataset(env_name, horizon, stride, test_split=0.2, append_goals=False, get_rewards=False, separate_test_trajectories=False, cum_rewards=True):
    dataset_file = 'data/'+env_name+'.pkl'
    with open(dataset_file, "rb") as f:
        dataset = pickle.load(f)

    observations = []
    actions = []
    terminals = []
    rewards = []
    if get_rewards:
        rewards = []
    # goals = []

    if env_name == 'antmaze-large-diverse-v2' or env_name == 'antmaze-medium-diverse-v2':

        num_trajectories = np.where(dataset['timeouts'])[0].shape[0]
        assert num_trajectories == 999, 'Dataset has changed. Review the dataset extraction'

        if append_goals:
            dataset['observations'] = np.hstack([dataset['observations'],dataset['infos/goal']])
        print('Total trajectories: ', num_trajectories)

        for traj_idx in range(num_trajectories):
            start_idx = traj_idx * 1001
            end_idx = (traj_idx + 1) * 1001

            obs = dataset['observations'][start_idx : end_idx]
            act = dataset['actions'][start_idx : end_idx]
            if get_rewards:
                rew = np.expand_dims(dataset['rewards'][start_idx : end_idx],axis=1)
                
            # reward = dataset['rewards'][start_idx : end_idx]
            # goal = dataset['infos/goal'][start_idx : end_idx]

            num_observations = obs.shape[0]

            for chunk_idx in range(num_observations // stride - horizon):
                chunk_start_idx = chunk_idx * stride
                chunk_end_idx = chunk_start_idx + horizon

                observations.append(torch.tensor(obs[chunk_start_idx : chunk_end_idx], dtype=torch.float32))
                actions.append(torch.tensor(act[chunk_start_idx : chunk_end_idx], dtype=torch.float32))
                if get_rewards:
                    if np.sum(rew[chunk_start_idx : chunk_end_idx]>0):
                        rewards.append(torch.ones((chunk_end_idx-chunk_start_idx,1), dtype=torch.float32))
                        break
                    rewards.append(torch.tensor(rew[chunk_start_idx : chunk_end_idx], dtype=torch.float32))
                # goals.append(torch.tensor(goal[chunk_start_idx : chunk_end_idx], dtype=torch.float32))

        observations = torch.stack(observations)
        actions = torch.stack(actions)
        if get_rewards:
            rewards = torch.stack(rewards)
        # goals = torch.stack(goals)

        num_samples = observations.shape[0]
        # print(num_samples)
        # assert num_samples == 960039, 'Dataset has changed. Review the dataset extraction'

        print('Total data samples extracted: ', num_samples)
        num_test_samples = int(test_split * num_samples)

        if separate_test_trajectories:
            train_indices = np.arange(0, num_samples - num_test_samples)
            test_indices = np.arange(num_samples - num_test_samples, num_samples)
        else:
            test_indices = np.random.choice(np.arange(num_samples), num_test_samples, replace=False)
            train_indices = np.array(list(set(np.arange(num_samples)) - set(test_indices)))
        np.random.shuffle(train_indices)

        observations_train = observations[train_indices]
        actions_train = actions[train_indices]
        if get_rewards:
            rewards_train = rewards[train_indices]
        else:
            rewards_train = None
        # goals_train = goals[train_indices]

        observations_test = observations[test_indices]
        actions_test = actions[test_indices]
        if get_rewards:
            rewards_test = rewards[test_indices]
        else:
            rewards_test = None
        # goals_test = goals[test_indices]

        return dict(observations_train=observations_train,
                    actions_train=actions_train,
                    rewards_train=rewards_train,
                    # goals_train=goals_train,
                    observations_test=observations_test,
                    actions_test=actions_test,
                    rewards_test=rewards_test,
                    # goals_test=goals_test,
                    )

    elif 'kitchen' in env_name:

        num_trajectories = np.where(dataset['terminals'])[0].shape[0]

        print('Total trajectories: ', num_trajectories)

        terminals = np.where(dataset['terminals'])[0]
        terminals = np.append(-1, terminals)

        for traj_idx in range(1, len(terminals)):
            start_idx = terminals[traj_idx - 1] + 1
            end_idx = terminals[traj_idx] + 1

            obs = dataset['observations'][start_idx : end_idx]
            act = dataset['actions'][start_idx : end_idx]
            rew = np.expand_dims(dataset['rewards'][start_idx : end_idx],axis=1)

            # reward = dataset['rewards'][start_idx : end_idx]
            # goal = dataset['infos/goal'][start_idx : end_idx]

            num_observations = obs.shape[0]

            for chunk_idx in range(num_observations // stride - horizon):
                chunk_start_idx = chunk_idx * stride
                chunk_end_idx = chunk_start_idx + horizon

                observations.append(torch.tensor(obs[chunk_start_idx : chunk_end_idx], dtype=torch.float32))
                actions.append(torch.tensor(act[chunk_start_idx : chunk_end_idx], dtype=torch.float32))
                if cum_rewards:
                    rewards.append(torch.tensor(rew[chunk_start_idx : chunk_end_idx], dtype=torch.float32))
                else:
                    rewards.append(torch.tensor(np.diff(rew[chunk_start_idx : chunk_end_idx], axis=0, prepend=rew[chunk_start_idx, 0]), dtype=torch.float32))
                # goals.append(torch.tensor(goal[chunk_start_idx : chunk_end_idx], dtype=torch.float32))

        observations = torch.stack(observations)
        actions = torch.stack(actions)
        rewards = torch.stack(rewards)

        num_samples = observations.shape[0]

        print('Total data samples extracted: ', num_samples)
        num_test_samples = int(test_split * num_samples)

        if separate_test_trajectories:
            train_indices = np.arange(0, num_samples - num_test_samples)
            test_indices = np.arange(num_samples - num_test_samples, num_samples)
        else:
            test_indices = np.random.choice(np.arange(num_samples), num_test_samples, replace=False)
            train_indices = np.array(list(set(np.arange(num_samples)) - set(test_indices)))
        np.random.shuffle(train_indices)

        observations_train = observations[train_indices]
        actions_train = actions[train_indices]
        rewards_train = rewards[train_indices]

        observations_test = observations[test_indices]
        actions_test = actions[test_indices]
        rewards_test = rewards[test_indices]

        return dict(observations_train=observations_train,
                    actions_train=actions_train,
                    rewards_train=rewards_train,
                    observations_test=observations_test,
                    actions_test=actions_test,
                    rewards_test=rewards_test,
                    )

    elif 'maze2d' in env_name:

        if append_goals:
            dataset['observations'] = np.hstack([dataset['observations'], dataset['infos/goal']])

        obs = dataset['observations']
        act = dataset['actions']

        if get_rewards:
            rew = np.expand_dims(dataset['rewards'], axis=1)

        # reward = dataset['rewards'][start_idx : end_idx]
        # goal = dataset['infos/goal'][start_idx : end_idx]

        num_observations = int(obs.shape[0])

        for chunk_idx in range(num_observations // stride - horizon):
            chunk_start_idx = chunk_idx * stride
            chunk_end_idx = chunk_start_idx + horizon

            observations.append(torch.tensor(obs[chunk_start_idx : chunk_end_idx], dtype=torch.float32))
            actions.append(torch.tensor(act[chunk_start_idx : chunk_end_idx], dtype=torch.float32))
            if get_rewards:
                rewards.append(torch.tensor(rew[chunk_start_idx : chunk_end_idx], dtype=torch.float32))
            # goals.append(torch.tensor(goal[chunk_start_idx : chunk_end_idx], dtype=torch.float32))

        observations = torch.stack(observations)
        actions = torch.stack(actions)
        if get_rewards:
            rewards = torch.stack(rewards)
        # goals = torch.stack(goals)

        num_samples = observations.shape[0]

        print('Total data samples extracted: ', num_samples)
        num_test_samples = int(test_split * num_samples)

        if separate_test_trajectories:
            train_indices = np.arange(0, num_samples - num_test_samples)
            test_indices = np.arange(num_samples - num_test_samples, num_samples)
        else:
            test_indices = np.random.choice(np.arange(num_samples), num_test_samples, replace=False)
            train_indices = np.array(list(set(np.arange(num_samples)) - set(test_indices)))
        np.random.shuffle(train_indices)

        observations_train = observations[train_indices]
        actions_train = actions[train_indices]
        if get_rewards:
            rewards_train = rewards[train_indices]
        else:
            rewards_train = None
        # goals_train = goals[train_indices]

        observations_test = observations[test_indices]
        actions_test = actions[test_indices]
        if get_rewards:
            rewards_test = rewards[test_indices]
        else:
            rewards_test = None
        # goals_test = goals[test_indices]

        return dict(observations_train=observations_train,
                    actions_train=actions_train,
                    rewards_train=rewards_train,
                    # goals_train=goals_train,
                    observations_test=observations_test,
                    actions_test=actions_test,
                    rewards_test=rewards_test,
                    # goals_test=goals_test,
                    )

    else:
        obs = dataset['observations']
        act = dataset['actions']
        rew = np.expand_dims(dataset['rewards'],axis=1)
        dones = np.expand_dims(dataset['terminals'],axis=1)
        episode_step = 0
        chunk_idx = 0

        while chunk_idx < rew.shape[0]-horizon+1:
            chunk_start_idx = chunk_idx
            chunk_end_idx = chunk_start_idx + horizon

            observations.append(torch.tensor(obs[chunk_start_idx : chunk_end_idx], dtype=torch.float32))
            actions.append(torch.tensor(act[chunk_start_idx : chunk_end_idx], dtype=torch.float32))
            rewards.append(torch.tensor(rew[chunk_start_idx : chunk_end_idx], dtype=torch.float32))
            terminals.append(torch.tensor(dones[chunk_start_idx : chunk_end_idx], dtype=torch.float32))
            if np.sum(dones[chunk_start_idx : chunk_end_idx]>0):
                episode_step = 0
                chunk_idx += horizon
            elif(episode_step==1000-horizon):
                episode_step = 0
                chunk_idx += horizon
            else:
                episode_step += 1
                chunk_idx += 1

        observations = torch.stack(observations)
        actions = torch.stack(actions)
        rewards = torch.stack(rewards)
        terminals = torch.stack(terminals)

        num_samples = observations.shape[0]

        print('Total data samples extracted: ', num_samples)
        num_test_samples = int(test_split * num_samples)

        if separate_test_trajectories:
            train_indices = np.arange(0, num_samples - num_test_samples)
            test_indices = np.arange(num_samples - num_test_samples, num_samples)
        else:
            test_indices = np.random.choice(np.arange(num_samples), num_test_samples, replace=False)
            train_indices = np.array(list(set(np.arange(num_samples)) - set(test_indices)))
        np.random.shuffle(train_indices)

        observations_train = observations[train_indices]
        actions_train = actions[train_indices]
        rewards_train = rewards[train_indices]
        terminals_train = terminals[train_indices]

        observations_test = observations[test_indices]
        actions_test = actions[test_indices]
        rewards_test = rewards[test_indices]
        terminals_test = terminals[test_indices]

        return dict(observations_train=observations_train,
                    actions_train=actions_train,
                    rewards_train=rewards_train,
                    terminals_train=terminals_train,
                    observations_test=observations_test,
                    actions_test=actions_test,
                    rewards_test=rewards_test,
                    terminals_test=terminals_test
                    )

###############################################################################################


def hard_update(target, source):
	for target_param, param in zip(target.parameters(), source.parameters()):
		target_param.data.copy_(param.data)

def soft_update(target, source, tau=0.001):
	for target_param, param in zip(target.parameters(), source.parameters()):
		target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
          

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

def linear_beta_schedule(timesteps, beta_start=1e-4, beta_end=2e-2, dtype=torch.float32):
    betas = np.linspace(
        beta_start, beta_end, timesteps
    )
    return torch.tensor(betas, dtype=dtype)

def extract(x: torch.Tensor, t, x_shape):
    b, *_ = t.shape
    # x: [timestep, length], t: [b]
	# use gather to extract the t-th element along the first dimension
    out = torch.gather(x, 0, t.reshape(b, 1).expand(b, x.shape[1]))
    return out.unsqueeze(-1)



class Progress:

	def __init__(self, total, name='Progress', ncol=3, max_length=20, indent=0, line_width=100, speed_update_freq=100):
		self.total = total
		self.name = name
		self.ncol = ncol
		self.max_length = max_length
		self.indent = indent
		self.line_width = line_width
		self._speed_update_freq = speed_update_freq

		self._step = 0
		self._prev_line = '\033[F'
		self._clear_line = ' ' * self.line_width

		self._pbar_size = self.ncol * self.max_length
		self._complete_pbar = '#' * self._pbar_size
		self._incomplete_pbar = ' ' * self._pbar_size

		self.lines = ['']
		self.fraction = '{} / {}'.format(0, self.total)

		self.resume()

	def update(self, description, n=1):
		self._step += n
		if self._step % self._speed_update_freq == 0:
			self._time0 = time.time()
			self._step0 = self._step
		self.set_description(description)

	def resume(self):
		self._skip_lines = 1
		print('\n', end='')
		self._time0 = time.time()
		self._step0 = self._step

	def pause(self):
		self._clear()
		self._skip_lines = 1

	def set_description(self, params=[]):

		if type(params) == dict:
			params = sorted([
				(key, val)
				for key, val in params.items()
			])

		############
		# Position #
		############
		self._clear()

		###########
		# Percent #
		###########
		percent, fraction = self._format_percent(self._step, self.total)
		self.fraction = fraction

		#########
		# Speed #
		#########
		speed = self._format_speed(self._step)

		##########
		# Params #
		##########
		num_params = len(params)
		nrow = math.ceil(num_params / self.ncol)
		params_split = self._chunk(params, self.ncol)
		params_string, lines = self._format(params_split)
		self.lines = lines

		description = '{} | {}{}'.format(percent, speed, params_string)
		print(description)
		self._skip_lines = nrow + 1

	def append_description(self, descr):
		self.lines.append(descr)

	def _clear(self):
		position = self._prev_line * self._skip_lines
		empty = '\n'.join([self._clear_line for _ in range(self._skip_lines)])
		print(position, end='')
		print(empty)
		print(position, end='')

	def _format_percent(self, n, total):
		if total:
			percent = n / float(total)

			complete_entries = int(percent * self._pbar_size)
			incomplete_entries = self._pbar_size - complete_entries

			pbar = self._complete_pbar[:complete_entries] + self._incomplete_pbar[:incomplete_entries]
			fraction = '{} / {}'.format(n, total)
			string = '{} [{}] {:3d}%'.format(fraction, pbar, int(percent * 100))
		else:
			fraction = '{}'.format(n)
			string = '{} iterations'.format(n)
		return string, fraction

	def _format_speed(self, n):
		num_steps = n - self._step0
		t = time.time() - self._time0
		speed = num_steps / t
		string = '{:.1f} Hz'.format(speed)
		if num_steps > 0:
			self._speed = string
		return string

	def _chunk(self, l, n):
		return [l[i:i + n] for i in range(0, len(l), n)]

	def _format(self, chunks):
		lines = [self._format_chunk(chunk) for chunk in chunks]
		lines.insert(0, '')
		padding = '\n' + ' ' * self.indent
		string = padding.join(lines)
		return string, lines

	def _format_chunk(self, chunk):
		line = ' | '.join([self._format_param(param) for param in chunk])
		return line

	def _format_param(self, param):
		k, v = param
		return '{} : {}'.format(k, v)[:self.max_length]

	def stamp(self):
		if self.lines != ['']:
			params = ' | '.join(self.lines)
			string = '[ {} ] {}{} | {}'.format(self.name, self.fraction, params, self._speed)
			self._clear()
			print(string, end='\n')
			self._skip_lines = 1
		else:
			self._clear()
			self._skip_lines = 0

	def close(self):
		self.pause()


class Silent:

	def __init__(self, *args, **kwargs):
		pass

	def __getattr__(self, attr):
		return lambda *args: None