from comet_ml import Experiment
import sys, os, pickle , random , argparse, torch, gym, copy, time,math
sys.path.append(os.getcwd())
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, '/mnt/e/CORL/LTCD_proposed_2')
# Get the models module directory
models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))
# Add it to path
if models_dir not in sys.path:
    sys.path.insert(0, models_dir)
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import DataLoader, Subset
import torch.distributions.normal as Normal
# Import mixed precision components
from torch.cuda.amp import autocast, GradScaler
from training.AE_model import VAEModel , GRUEncoder, Decoder, Prior, BetaScheduler, DAEModel
from models.temporal_FiLM_AE import TemporalUnet_film
from models.diffusion_AE import Diffusion
from utils.utils import save_iql_checkpoint, save_checkpoint_DAE, load_iql_checkpoint, load_checkpoint_DAE, save_transformer_checkpoint, load_transformer_checkpoint
from utils.helpers import cycle
from datasets.dataset import SequenceDataset_v1, SplittableSequenceDataset, EncodedDAEDataset, EncodedTransformerDataset
from datasets.normalization import DatasetNormalizer
from sklearn.model_selection import train_test_split
from torch.optim import AdamW  
from models.transformer_model import HierarchicalTransformerPlanner, TransformerTrainer, SkillCriticQ, ValueCriticV, build_critic_batch_from_transformer_batch, critic_td_update_doubleq_Vbackup, critic_td_update_doubleq_Vbackup_CQL

from pathlib import Path
script_path = Path(__file__).resolve()
project_root = script_path.parent.parent
os.chdir(project_root)

from typing import Dict
os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
#-----------------------------------------------------------------------------#

#-----------------------------------------------------------------------------#
hyperparameters = {
    # --- Kitchen Envs ---  (scale from table; TV = 0.9 * max @ H=12)
    'kitchen-complete-v0':           {'lr': 1e-4, 'horizon': 6, 'n_timesteps': 5, 'scalar': 50, 'z_dim': 16, 'rtg': 300},  # max=320.36
    'kitchen-partial-v0':            {'lr': 1e-4, 'horizon': 6, 'n_timesteps': 5, 'scalar': 50, 'z_dim': 16, 'rtg': 400},  # max=444.23
    'kitchen-mixed-v0':              {'lr': 1e-4, 'horizon': 6, 'n_timesteps': 5, 'scalar': 50, 'z_dim': 16, 'rtg': 350},  # max=373.74

    # --- AntMaze Envs --- (only rows present in your table updated)
    'antmaze-umaze-v2':              {'lr': 1e-4, 'horizon': 16, 'n_timesteps': 5, 'scalar': 10,    'z_dim': 16, 'rtg': 50},      # (no H=12 row provided → unchanged)
    'antmaze-umaze-diverse-v2':      {'lr': 1e-4, 'horizon': 16, 'n_timesteps': 5, 'scalar': 10,    'z_dim': 16, 'rtg': 50},   # max=52.09
    'antmaze-medium-diverse-v2':     {'lr': 1e-4, 'horizon': 32, 'n_timesteps': 5, 'scalar': 10,    'z_dim': 16, 'rtg': 50},   # max=46.31
    'antmaze-large-diverse-v2':      {'lr': 1e-4, 'horizon': 32, 'n_timesteps': 5, 'scalar': 10,    'z_dim': 16, 'rtg': 55},   # max=51.13
    'antmaze-large-play-v2':         {'lr': 1e-4, 'horizon': 32, 'n_timesteps': 5, 'scalar': 10,    'z_dim': 16, 'rtg': 55},      # (no H=12 row provided → unchanged)

    # --- Maze2d Envs --- (use SHAPED rows, as your dict uses shaped scaling)
    'maze2d-umaze-v1':               {'lr': 1e-4, 'horizon': 32, 'n_timesteps': 5, 'scalar': 10,  'z_dim': 16, 'rtg': 80},   # shaped max=86.92
    'maze2d-medium-v1':              {'lr': 1e-4, 'horizon': 32, 'n_timesteps': 5, 'scalar': 100, 'z_dim': 16, 'rtg': 150},  # shaped max=160.54
    'maze2d-large-v1':               {'lr': 1e-4, 'horizon': 32, 'n_timesteps': 5, 'scalar': 100, 'z_dim': 16, 'rtg': 230},  # shaped max=251.24

    # --- Locomotion Envs --- (halfcheetah/hopper/walker2d)
    'halfcheetah-medium-expert-v2':  {'lr': 2e-4, 'horizon': 8,  'n_timesteps': 5, 'scalar': 2000, 'z_dim': 16, 'rtg': 3500}, # max=3589.05
    'halfcheetah-expert-v2':         {'lr': 2e-4, 'horizon': 8,  'n_timesteps': 5, 'scalar': 3000, 'z_dim': 16, 'rtg': 7000}, # max=7557.28
    'halfcheetah-medium-v2':         {'lr': 2e-4, 'horizon': 8,  'n_timesteps': 5, 'scalar': 1000, 'z_dim': 16, 'rtg': 3000}, # max=3412.66
    'halfcheetah-medium-replay-v2':  {'lr': 2e-4, 'horizon': 8,  'n_timesteps': 5, 'scalar': 4000, 'z_dim': 16, 'rtg': 7000}, # max=7557.28

    'hopper-medium-expert-v2':       {'lr': 2e-4, 'horizon': 16, 'n_timesteps': 5, 'scalar': 1000, 'z_dim': 16, 'rtg': 2000}, # max=2126.00
    'hopper-expert-v2':              {'lr': 2e-4, 'horizon': 16, 'n_timesteps': 5, 'scalar': 1000, 'z_dim': 16, 'rtg': 2500}, # max=2519.37
    'hopper-medium-v2':              {'lr': 2e-4, 'horizon': 16, 'n_timesteps': 5, 'scalar': 1000, 'z_dim': 16, 'rtg': 2000}, # max=2135.76
    'hopper-medium-replay-v2':       {'lr': 2e-4, 'horizon': 16, 'n_timesteps': 5, 'scalar': 1000, 'z_dim': 16, 'rtg': 2500}, # max=2519.37

    # Walker2d: only rows present in your table updated
    'walker2d-medium-expert-v2':     {'lr': 2e-4, 'horizon': 32, 'n_timesteps': 5, 'scalar': 1000, 'z_dim': 16, 'rtg': 3000},  # (no H=12 row provided → unchanged)
    'walker2d-expert-v2':            {'lr': 2e-4, 'horizon': 32, 'n_timesteps': 5, 'scalar': 1500, 'z_dim': 16, 'rtg': 3000}, # max=3363.62
    'walker2d-medium-v2':            {'lr': 2e-4, 'horizon': 32, 'n_timesteps': 5, 'scalar': 1500, 'z_dim': 16, 'rtg': 2500}, # max=2792.66
    'walker2d-medium-replay-v2':     {'lr': 2e-4, 'horizon': 32, 'n_timesteps': 5, 'scalar': 1000, 'z_dim': 16, 'rtg': 2500}, # max=2715.10

    # --- Adroit Envs ---
    'pen-human-v1':                  {'lr': 1e-4, 'horizon': 16, 'n_timesteps': 2, 'scalar': 3000, 'z_dim': 16, 'rtg': 9000},  # max=10029.80
    'pen-cloned-v1':                 {'lr': 1e-4, 'horizon': 16, 'n_timesteps': 2, 'scalar': 2000, 'z_dim': 16, 'rtg': 9000},  # max=10029.80
    'pen-expert-v1':                 {'lr': 1e-4, 'horizon': 16, 'n_timesteps': 2, 'scalar': 2000, 'z_dim': 16, 'rtg': 4000},  # max=4968.80

    'hammer-human-v0':               {'lr': 1e-4, 'horizon': 32, 'n_timesteps': 5, 'scalar': 1000, 'z_dim': 16, 'rtg': 7000},  # max=8275.97
    'hammer-cloned-v0':              {'lr': 1e-4, 'horizon': 32, 'n_timesteps': 5, 'scalar': 1000, 'z_dim': 16, 'rtg': 7000},  # max=8275.97
    'hammer-expert-v0':              {'lr': 1e-4, 'horizon': 32, 'n_timesteps': 5, 'scalar': 7000, 'z_dim': 16, 'rtg': 13000}, # max=15313.61

    'relocate-human-v0':             {'lr': 1e-4, 'horizon': 32, 'n_timesteps': 5, 'scalar': 1000, 'z_dim': 16, 'rtg': 4000},  # max=4319.05
    'relocate-cloned-v0':            {'lr': 1e-4, 'horizon': 32, 'n_timesteps': 5, 'scalar': 500, 'z_dim': 16, 'rtg': 3000},  # max=4319.05
    'relocate-expert-v0':            {'lr': 1e-4, 'horizon': 32, 'n_timesteps': 5, 'scalar': 1000, 'z_dim': 16, 'rtg': 2500},  # max=4664.53

    'door-human-v0':                 {'lr': 1e-4, 'horizon': 32, 'n_timesteps': 5, 'scalar': 250,  'z_dim': 16, 'rtg': 600},   # max=1073.51
    'door-cloned-v0':                {'lr': 1e-4, 'horizon': 32, 'n_timesteps': 5, 'scalar': 250,  'z_dim': 16, 'rtg': 600},   # max=1073.51
    'door-expert-v0':                {'lr': 1e-4, 'horizon': 32, 'n_timesteps': 5, 'scalar': 1000, 'z_dim': 16, 'rtg': 2000},  # max=2788.04

    # --- Other Envs ---
    'FetchPush-v1':                  {'lr': 1e-4, 'horizon': 16, 'n_timesteps': 5, 'scalar': 1.1,    'z_dim': 16, 'rtg': 1.0},     # unchanged
}
#-----------------------------------------------------------------------------#

#-----------------------------------------------------------------------------#
#kitchen-complete-v0  kitchen-partial-v0 antmaze-umaze-diverse-v2 maze2d-umaze-v1  Pen-cloned-v1 
#halfcheetah-medium-replay-v2  hopper-medium-replay-v2  walker2d-medium-replay-v2 
#halfcheetah-medium-v2  hopper-medium-v2  walker2d-medium-v2 
#halfcheetah-expert-v2  hopper-expert-v2  walker2d-expert-v2 
#kitchen-partial-v0  kitchen-mixed-v0  kitchen-complete-v0 
#antmaze-umaze-v2  antmaze-medium-diverse-v2  antmaze-large-diverse-v2 
#maze2d-umaze-v1  maze2d-medium-v1  maze2d-large-v1 
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='antmaze-medium-diverse-v2')  # 'antmaze-umaze-v2' 'pen-cloned-v1'
    parser.add_argument('--seed', type=int, default=5000)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--eval_freq', type=int, default=25_000 )
    parser.add_argument('--save_freq', type=int, default=50_000 ) 
    parser.add_argument('--totle_iteration', type=int, default= 5_00_000 )  #1_000_000
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--z_dim', type=int, default=16)
    parser.add_argument('--h_dim', type=int, default=256)
    parser.add_argument('--schedule', type=str, default='cosine', help='schedule for dae model')
    parser.add_argument('--beta_vae', type=float, default=0.05, help='beta for dae model')
    parser.add_argument('--tau', type=float, default=0.001, help='tau for dae model')
    parser.add_argument('--gpus',       type=int, default=1,       help='how many GPUs you want to use')

    parser.add_argument('--comet_api_key', type=str, default='6EMt49eBsRdI18hrCnHOpfcIC', help='Comet ML API Key')
    parser.add_argument('--comet_project_name', type=str, default='FAU_DAE_Transformer', help='Comet ML Project Name')
    parser.add_argument('--comet_file_name', type=str, default=None, help='FAU_Comet ML file Name')
    parser.add_argument('--normed_actions', action='store_true', default=False, help='Use normalized actions')
    parser.add_argument('--use_mixed_precision', action='store_true', default=False, help='Use mixed precision training (FP16) - safe for IQL')
    parser.add_argument('--policy_net_complex',  type=int, default=0, help='Use complex policy network: 0: High-Level Policy, 1: RealNVPPolicy')
    parser.add_argument('--diffusion_sequence', type=int, default=0, help='1: Use diffusion sequence, 0: Use diffusion single step')
    parser.add_argument('--checkpoint_suffix', type=str, default='', help='Suffix for checkpoint directory (e.g., "1" for checkpoints1/)')
    parser.add_argument('--return_scale', type=float, default=0, help='Return scale')
    parser.add_argument('--iql_adv_weight_type',  type=int, default=0, help=' IN IQL TRAINER 0: Normalized and Floored, 1: Not Normalized and Floored, 2: Quantile-based beta (robust)')
    parser.add_argument('--horizon', type=int, default=12, help='Horizon for IQL')
    parser.add_argument('--goal_flag', type=int, default=0, help='Goal flag')
    parser.add_argument('--reward_shaping', type=int, default=1, help='Use reward shaping')
    parser.add_argument('--context_length', type=int, default=16, help='Context length')
    parser.add_argument('--M2D_manualfix_data', type=int, default=1, help='Manual Relabel Rewards for maze2d')
    parser.add_argument('--AM_manualfix_data', type=int, default=1, help='ANT MAZE MANUAL FIX DATASET')
    parser.add_argument('--radius', type=float, default=0.5, help='Radius for maze2d and antmaze')
    parser.add_argument('--dataset_visualization_flag', type=int, default=1, help='dataset_visualization_flag for debugging')

    parser.add_argument('--Sigma_clamp', type=float, default=3, help='Sigma_clamp values for AE_model clamping Zpost')
    parser.add_argument('--Z_F_loss', type=int, default=1, help='Z-force loss adding flag in AEmodel')
    parser.add_argument('--InfoL_new', type=int, default=1, help='InfoL_new updated Infoloss in AE model')
    parser.add_argument('--KLbetaend', type=float, default=0.04, help='AEmodel KLbeta loss ending weight')
    parser.add_argument('--infoloss_tau', type=float, default=0.15, help='changing tau values for info loss')
    parser.add_argument('--stride', type=int, default=1, help='with which stride collect transformt dataset')
    parser.add_argument('--skill_normalized', type=int, default=1, help='Use skill normalization or not')
    parser.add_argument('--transf_mlmu', type=int, default=0, help='Use MLMU mask in transformer or not')
    parser.add_argument('--prior_zsigma_clamp', type=int, default=0, help='prior_zsigma_clamp ')
    parser.add_argument('--min_clamp_zsigma', type=float, default=0.005, help='Sigma_clamp values for AE_model clamping Zpost')
    parser.add_argument('--best_weight', type=int, default=1, help='use best weights')
    parser.add_argument('--weight_cnt', type=float, default=50000, help='checkpoint step number')
    parser.add_argument('--all_Z_scale_RTG', type=int, default=1, help='RTG scaling, z or scaled')    

    return parser.parse_args()
#-----------------------------------------------------------------------------#


#-----------------------------------------------------------------------------#
def to_device(data, device):
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    elif isinstance(data, dict):
        return {k: to_device(v, device) for k, v in data.items()}
    elif hasattr(data, 'to'):
        return data.to(device)
    return data

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def calculate_ETA(start_time, steps_done, total_steps, eval_freq, last_eval_duration):
    if steps_done == 0:
        return "--:--:--", "--", "--:--:--"
    elapsed_time = time.time() - start_time
    evals_done = (steps_done // eval_freq) + 1
    time_spent_in_validation = evals_done * last_eval_duration
    # Estimate the time spent only on training steps (optimizer.step(), etc.)
    pure_training_time = elapsed_time - time_spent_in_validation
    pure_training_time = max(0, pure_training_time)
    # --- 2. Project Future Time for Both Components ---
    avg_time_per_step = pure_training_time / steps_done if steps_done > 0 else 0
    steps_remaining = total_steps - steps_done
    eta_training_seconds = steps_remaining * avg_time_per_step
    # Project the time for remaining validation runs
    total_evals_to_run = total_steps // eval_freq
    evals_remaining = total_evals_to_run - evals_done
    eta_validation_seconds = evals_remaining * last_eval_duration
    # --- 3. Calculate Total ETA and Format ---
    total_eta_seconds = eta_training_seconds + eta_validation_seconds
    eta_seconds_int = int(total_eta_seconds)
    minutes, seconds = divmod(eta_seconds_int, 60)
    hours, minutes = divmod(minutes, 60)
    eta_formatted = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    # --- Formatted Elapsed Time ---
    elapsed_seconds_int = int(elapsed_time)
    e_m, e_s = divmod(elapsed_seconds_int, 60)
    e_h, e_m = divmod(e_m, 60)
    elapsed_formatted = f"{e_h:02d}:{e_m:02d}:{e_s:02d}"
    # --- Formatted Average Time ---
    # We calculate time per 100 steps for a more readable number
    avg_time_per_100_step_formatted = f"{(avg_time_per_step * 100):.2f}s"
    return elapsed_formatted, avg_time_per_100_step_formatted, eta_formatted
#-----------------------------------------------------------------------------#


#-----------------------------------------------------------------------------#

def build_param_groups(model: nn.Module, base_lr=3e-4, wd=0.1, head_lr_mult=5.0):
    # 1) Collect head params and their ids so we can exclude them from body groups
    head_params = list(model.skill_head.parameters()) + list(model.value_head.parameters())
    head_ids = {id(p) for p in head_params}

    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if id(p) in head_ids:
            continue  # skip heads here; we'll add them in dedicated groups below

        # no weight decay for biases and norms
        if p.dim() == 1 or name.endswith(".bias") or "layernorm" in name.lower() or "ln" in name.lower() or "norm" in name.lower():
            no_decay.append(p)
        else:
            decay.append(p)

    groups = [
        {"params": decay,     "lr": base_lr,               "weight_decay": wd},
        {"params": no_decay,  "lr": base_lr,               "weight_decay": 0.0},
        {"params": list(model.skill_head.parameters()), "lr": base_lr*head_lr_mult, "weight_decay": 0.0},
        {"params": list(model.value_head.parameters()), "lr": base_lr,              "weight_decay": 0.0},
    ]

    # 2) Sanity check: no duplicates across groups
    all_ids = []
    for g in groups:
        all_ids.extend([id(p) for p in g["params"]])
    assert len(all_ids) == len(set(all_ids)), "duplicate parameters across optimizer groups"

    # (optional) quick counts
    # print(f"[param groups] decay={sum(p.numel() for p in decay)}, "
    #       f"no_decay={sum(p.numel() for p in no_decay)}, "
    #       f"skill_head={sum(p.numel() for p in model.skill_head.parameters())}, "
    #       f"value_head={sum(p.numel() for p in model.value_head.parameters())}")

    return groups

def param_in_optim(optim, param):
    for g in optim.param_groups:
        for p in g['params']:
            if p is param:
                return True
    return False
#-----------------------------------------------------------------------------#


#-----------------------------------------------------------------------------#

class ModelCheckpointer:
    def __init__(self, min_improvement=0.001):
        self.very_best_validation_loss  = float('inf')
        self.best_validation_loss = float('inf') 
        self.best_step = 0
        self.min_improvement = min_improvement  # Minimum improvement threshold
        
    def should_save_checkpoint(self, validation_loss, step_count):     
        # Always update very_best_validation_loss if this is the absolute minimum
        if validation_loss < self.very_best_validation_loss:
            self.very_best_validation_loss = validation_loss
            print(f"             New all time absolute best validation loss: {self.very_best_validation_loss:.6f}")

        tolerance_threshold = self.very_best_validation_loss + self.min_improvement
        # Always save if  better  or negligible upward trend.
        if validation_loss <= tolerance_threshold:
            self.best_validation_loss = validation_loss
            self.best_step = step_count
            return True, self.best_validation_loss, self.best_step
            
        return False, self.best_validation_loss, self.best_step
#-----------------------------------------------------------------------------#

#-----------------------------------------------------------------------------#
def main():
    # set_seed(114514)
    torch.autograd.set_detect_anomaly(True) # <-- ADD THIS LINE
    args = parse_args()
    env_name = args.env_name
    w = 1.1
    rtg = hyperparameters[env_name]['rtg']
    z_dim = hyperparameters[env_name]['z_dim']
    set_seed(args.seed)
    env = gym.make(env_name)
    batch_size = args.batch_size
    device =  torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    horizon = args.horizon
    n_timesteps = hyperparameters[env_name]['n_timesteps']
    schedule = args.schedule
    beta_vae = args.beta_vae
    tau = args.tau
    context_length = next((ctx for threshold, ctx in [(8, 15), (12, 12), (16, 10), (22, 8), (28, 6)] if args.horizon <= threshold), 4)
    context_length = args.context_length
    if ('antmaze' in env_name or 'maze2d' in env_name):
            args.goal_flag = 1
    #----------------------------------------------------------------------------------------#
    print("================= Preparing Datset Generators =================")
    # Preparing dataset Genrator
    if args.goal_flag:
        encoded_dataset_filepath = os.path.join(f'datasets{args.checkpoint_suffix}/', 'D_transformer_dataset/', env_name+'/d_transf_goal_dataset_NorAct.pkl' ) if args.normed_actions else os.path.join(f'datasets{args.checkpoint_suffix}/', 'D_transformer_dataset/', env_name+'/d_transf_goal_dataset_UnNorAct.pkl' )
    else:
        encoded_dataset_filepath = os.path.join(f'datasets{args.checkpoint_suffix}/', 'D_transformer_dataset/', env_name+'/d_transf_dataset_NorAct.pkl' ) if args.normed_actions else os.path.join(f'datasets{args.checkpoint_suffix}/', 'D_transformer_dataset/', env_name+'/d_transf_dataset_UnNorAct.pkl' )
    dataset = EncodedTransformerDataset(encoded_dataset_filepath, goal_flag=args.goal_flag)
    # Split indices for training and validation sets (e.g., 95% train, 5% val)
    train_indices, val_indices = train_test_split(list(range(len(dataset))),test_size=0.10, random_state=42) # Use a fixed random state for reproducible splits)
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=args.num_workers if hasattr(args, 'num_workers') else 0)
    # loading datset normalizer for evaluation
    return_scale = args.return_scale if args.return_scale != 0 else hyperparameters[env_name]['scalar'] #if any(env in args.env_name for env in ('hopper', 'halfcheetah', 'walker2d', 'kitchen')) else 10
    return_scale = 1 if ('antmaze' in env_name and args.reward_shaping == 0) else return_scale
    dataset_original = SplittableSequenceDataset(env_name, horizon=horizon, returns_scale= return_scale, termination_penalty=None, load_path=None,
                                         subbatchsize= 8, normed_actions=args.normed_actions, goal_flag=args.goal_flag, 
                                         reward_shaping=args.reward_shaping , radius=args.radius, reward_relabel_maze2d=args.M2D_manualfix_data,
                                         AM_manualFIX_data=args.AM_manualfix_data, dataset_visualization_flag=args.dataset_visualization_flag)
    normalizer:DatasetNormalizer = dataset_original.normalizer
    train_loader_original = DataLoader(dataset_original, batch_size=256, shuffle=True, num_workers=0)
    #----------------------------------------------------------------------------------------#

    #----------------------------------------------------------------------------------------#
    print("================= Loading Latent Normalization mean, std values =================")
    if args.skill_normalized:
        Z_stats_path=os.path.join(f'datasets{args.checkpoint_suffix}/', 'D_transformer_dataset/', env_name+'/skill_stats.pkl')    
        with open(Z_stats_path, 'rb') as f:
            skill_stats = pickle.load(f)
    else:
        skill_stats = {'mean_z': np.zeros(z_dim), 'std_z': np.ones(z_dim)}
        
    #----------------------------------------------------------------------------------------#

    #----------------------------------------------------------------------------------------#
    print("================= Model Initialization =================")
    state_dim = state_dim_goal = dataset.data_list_of_dicts[0]['states'].shape[1]
    if ('antmaze' in env_name or 'maze2d' in env_name):
        state_dim = dataset.data_list_of_dicts[0]['states'].shape[1] - 2
        state_dim_goal = dataset.data_list_of_dicts[0]['states'].shape[1]
    # === Critics (Double-Q) ===
    hidden, layers = 512, 2      #
    Q1 = SkillCriticQ(state_dim_goal, z_dim, hidden=512, layers=layers).to(device)
    Q2 = SkillCriticQ(state_dim_goal, z_dim, hidden=hidden, layers=layers).to(device)
    V  = ValueCriticV(state_dim_goal,          hidden=hidden, layers=layers).to(device)
    Q1_targ = copy.deepcopy(Q1).to(device).eval()
    Q2_targ = copy.deepcopy(Q2).to(device).eval()
    V_targ  = copy.deepcopy(V).to(device).eval()
    # === Optims ===
    optQ1 = torch.optim.AdamW(Q1.parameters(), lr=3e-4, weight_decay=1e-4)
    optQ2 = torch.optim.AdamW(Q2.parameters(), lr=3e-4, weight_decay=1e-4)
    optV  = torch.optim.AdamW(V.parameters(),  lr=3e-4, weight_decay=1e-4)
    # === EMA / Polyak update helper ===
    def ema_update(target: nn.Module, online: nn.Module, tau=0.005):
        with torch.no_grad():
            for p_t, p in zip(target.parameters(), online.parameters()):
                p_t.data.lerp_(p.data, tau)
    # === Transformer Model ===
    transformer_model = HierarchicalTransformerPlanner(state_dim=state_dim_goal, skill_dim=z_dim, context_length=context_length,
                             n_head=8, n_layer=8, d_model=args.h_dim).to(device)
    transformer_model.Q1 = Q1; transformer_model.Q2 = Q2; transformer_model.V = V
    trainer = TransformerTrainer(model=transformer_model, skills_stats=skill_stats,  Q1=Q1, Q2=Q2, V=V, H=horizon, gamma=0.99, device=device, transf_mlmu=args.transf_mlmu)
    transformer_optimizer = AdamW(transformer_model.parameters(), lr=1e-4)
    # transformer_optimizer = AdamW(build_param_groups(transformer_model), betas=(0.9, 0.95))
    # print("skill_head.weight in optimizer? ", param_in_optim(transformer_optimizer, transformer_model.skill_head.weight))
    # print("skill_head.bias   in optimizer? ", param_in_optim(transformer_optimizer, transformer_model.skill_head.bias))
    loss_fn = nn.MSELoss()
    # Initialize mixed precision training (SAFE for IQL)
    use_mixed_precision = args.use_mixed_precision and torch.cuda.is_available()
    scaler = GradScaler() if use_mixed_precision else None

    # --------- LOAD DAE Model    
    use_attention = False
    film_flag =True
    dim_mults=(1, 2, 4, 8)
    a_dim = env.action_space.shape[0]
    h_dim = args.h_dim
    n_timesteps = 5
    encoder = GRUEncoder(state_dim, a_dim, z_dim, h_dim, Sigma_clamp=args.Sigma_clamp)
    length_sequence = horizon if args.diffusion_sequence else 1
    model = TemporalUnet_film(length_sequence, a_dim,  z_dim, attention=use_attention, dim_mults=dim_mults, state_dim=state_dim).to(device)
    diff_planner = Diffusion(z_dim, model, None, horizon=horizon, n_timesteps=n_timesteps, film_flag=film_flag, predict_epsilon=True, beta_schedule=schedule, w=w, device=device).to(device)# predict_epsilon=False
    prior   = Prior     (state_dim, z_dim, h_dim, prior_zsigma_clamp=args.prior_zsigma_clamp).to(device)
    dae_model = DAEModel(encoder, diff_planner, prior, beta=args.beta_vae, tau=tau, n_timesteps=n_timesteps, device=device,
	                diffusion_sequence=args.diffusion_sequence, goal_flag=0, env=env_name,
					Sigma_clamp=args.Sigma_clamp, min_clamp_zsigma=args.min_clamp_zsigma, Z_F_loss=args.Z_F_loss, InfoL_new=args.InfoL_new, KLbetaend=args.KLbetaend, infoloss_tau=args.infoloss_tau).to(device)
    # optimizer = AdamW(dae_model.parameters(), lr=1e-4, weight_decay=1e-4) 
    # optimizer_DAE = torch.optim.Adam(dae_model.parameters(), lr=lr, weight_decay=wd)
    for p in dae_model.ema_model.parameters():
        p.requires_grad = False
    encoder_params = list(dae_model.encoder.parameters()) + list(dae_model.prior.parameters())
    decoder_params = list(dae_model.decoder.parameters())
    optimizer_DAE = AdamW([
		{"params": encoder_params, "lr": 3e-4, "weight_decay": 0.0},  # higher LR, no WD
		{"params": decoder_params, "lr": 1e-4, "weight_decay": 1e-4},  # baseline LR, WD ok
	])
    checkpoint_dir = os.path.join(f'checkpoints{args.checkpoint_suffix}/', 'DAE_weights_KL/', env_name+'/')
    if args.best_weight:
        filename = 'DAE_'+ env_name + '_best_NorAct.pth' if args.normed_actions else 'DAE_'+ env_name + '_best_UnNorAct.pth'
    else:
        filename = 'DAE_'+ env_name + '_best_NorAct.pth' if args.normed_actions else 'DAE_'+ env_name + '_'+ args.weight_cnt +'.pth'
    PATH_TO_CHECKPOINT = os.path.join(checkpoint_dir, filename )
    start_step = load_checkpoint_DAE(PATH_TO_CHECKPOINT, dae_model, optimizer_DAE, device)
    
    if use_mixed_precision:
        print("✅ Mixed precision training enabled (FP16) - SAFE for IQL")
        print("   - Value network: FP16 (safe)")
        print("   - Q networks: FP16 (safe)")
        print("   - Policy network: FP32 (preserved for accuracy)")
    else:
        print("❌ Mixed precision training disabled (FP32)")



    #-----------------------------------------------------------------------------------------#

    #----------------------------------------------------------------------------------------#
    print("================= Initialzing Comet Experiment Logger =================")
    checkpoint_dir = os.path.join(f'checkpoints{args.checkpoint_suffix}/', 'Transformer_weights/', args.env_name+'/')  
    filename = 'Transf_'+ env_name  if args.comet_file_name is None else 'Transf_'+ env_name + '_' + args.comet_file_name
    os.makedirs(checkpoint_dir, exist_ok=True)
    comet_proj_name = '_Train_Transformer'
    comet_proj_name = '_NorAct' if args.normed_actions else '_UnNorAct'
    comet_proj_name = comet_proj_name + '_SimpPolicy' if args.policy_net_complex == 0 else comet_proj_name + '_CompPolicy'
    project_name_comet = args.comet_project_name + comet_proj_name
    experiment = Experiment(api_key = args.comet_api_key, project_name = project_name_comet)
    experiment.set_name(env_name)  
    experiment.auto_metric_logging = False
    experiment.auto_param_logging = False
    experiment.auto_histogram_logging = False
    experiment.log_parameters({
                            'z_dim':z_dim,
                            'state_dim':state_dim,
                            'env_name':env_name,
                            'filename':filename,
                            'batch_size': batch_size,
                            'totle_iteration': args.totle_iteration,
                            'save_freq': args.save_freq,
                            'normed_actions': args.normed_actions,
                            'use_mixed_precision': use_mixed_precision,
                            'policy_net_complex': args.policy_net_complex,
                            'goal_flag': args.goal_flag})
    #----------------------------------------------------------------------------------------#


    #----------------------------------------------------------------------------------------#
    def calculate_validation_q_value(iql_agent, val_loader, device, policy_net_complex=0, goal_flag=1):
        """
        Calculates the average predicted Q-value for the policy on a validation set.
        """
        iql_agent.policy_net.eval()
        iql_agent.q_net1.eval() ; iql_agent.q_net2.eval()  # We can just use one of the Q-nets for a consistent metric
        total_q_value = 0
        num_batches = 0
        with torch.no_grad(): # Ensure no gradients are computed for speed
            for batch_data in val_loader:
                if goal_flag:
                    s0_batch, goal_batch, z_batch, rtg_batch, sT_batch = batch_data # We only need s0 and z from this batch
                    s0_batch, goal_batch, z_batch, _, _ = to_device([s0_batch, goal_batch, z_batch, rtg_batch, sT_batch], device)
                else:
                    s0_batch, z_batch, _, _ = batch_data # We only need s0 and z from this batch
                    s0_batch, z_batch, _, _ = to_device([s0_batch, z_batch, _, _], device)
                    goal_batch = None
                z_batch = z_batch.to(device) # z from the dataset is used for Q-value calculation in IQL   
                # Use the policy to get a hypothetical skill for the state
                # This tests if the policy is choosing skills that the Q-net values highly
                if policy_net_complex==0:
                    z_pred_dist = iql_agent.policy_net.sample(s0_batch, goal_batch)
                    z_pred_mean = z_pred_dist
                else:
                    # RealNVPPolicy: sample() returns the sampled tensor directly
                    z_pred_mean = iql_agent.policy_net.sample(s0_batch)
                # Calculate the Q-value for the (s0, z_pred) pairs
                # q_values = iql_agent.q_net1(s0_batch, z_pred_mean) 
                q_values = torch.min(iql_agent.q_net1(s0_batch, goal_batch, z_pred_mean), iql_agent.q_net2(s0_batch, goal_batch, z_pred_mean))             
                total_q_value += q_values.mean().item()
                num_batches += 1
        # Set models back to training mode
        iql_agent.policy_net.train()
        iql_agent.q_net1.train()
        iql_agent.q_net2.train()
        return total_q_value / num_batches if num_batches > 0 else 0
    #----------------------------------------------------------------------------------------#

    #----------------------------------------------------------------------------------------#
    @torch.no_grad()
    def policy_sampler_fn(s_flat: torch.Tensor, num: int, device: torch.device) -> torch.Tensor:
        """
        Proposal sampler z ~ π(z|s). Uses the planner's state-only heads.
        """
        mu_s, logstd_s = transformer_model.cond_heads_from_state_only(s_flat, device)
        std_s = torch.exp(logstd_s).clamp_min(1e-6)
        BKS, Dz = mu_s.size()
        z = mu_s.unsqueeze(1) + std_s.unsqueeze(1) * torch.randn(BKS, num, Dz, device=s_flat.device)
        return z
    #----------------------------------------------------------------------------------------#  

    
    #----------------------------------------------------------------------------------------#
    print("================= Transformer TRAINING START =================")
    step_count = 0
    start_time = time.time()
    best_validation_loss = float('inf') # We want to MAXIMIZE the Q-value
    best_validation_q_value_for_skill = -float('inf') # We want to MAXIMIZE the Q-value
    best_eval_step = best_q_eval_step = -10
    last_eval_duration = 0.0
    best_adv  = -float('inf')
    best_nll  = float('inf')
    REL_NLL_MARGIN = 0.05   # +5% NLL slack
    ADV_EPS        = 1e-6

    # --------- Initializing Check Points    # --------- 
    checkpointer = ModelCheckpointer(min_improvement=0.5)   
    best_step = 0
    # --------- Initializing Training Loop   # --------- 
    for batch_idx, batch_data in enumerate(cycle(train_loader)):
        step_count += 1
        rtg, states, z, attention_mask = batch_data
        batch = [b.to(device) for b in batch_data]
        old_weight = transformer_model.skill_head.weight.clone()
        if args.all_Z_scale_RTG:
            scaled_rtg = batch[0]
        else:
            unnormed_rtg = normalizer.unnormalize(rtg ,'RTG')
            scaled_rtg = unnormed_rtg/return_scale
            scaled_rtg = scaled_rtg.to(device)
            batch[0] = scaled_rtg
        # --------- TRAIN/UPDATE MCritics TD step  # --------- 
        critic_batch = build_critic_batch_from_transformer_batch(batch, H=horizon, gamma=0.99, mean_z=trainer.mean_z, std_z=trainer.std_z, stride=args.stride, scaled_rtg=scaled_rtg)
        # logs_c = critic_td_update_doubleq_Vbackup( Q1, Q2, V, Q1_targ, Q2_targ, V_targ, critic_batch, gamma=0.99, H=horizon,
        #             optQ1=optQ1, optQ2=optQ2, optV=optV, tau=0.005)
        logs_c = critic_td_update_doubleq_Vbackup_CQL(
                Q1, Q2, V, Q1_targ, Q2_targ, V_targ,
                critic_batch, gamma=0.99, H=horizon,
                optQ1=optQ1, optQ2=optQ2, optV=optV, tau=0.005,
                cql_alpha=2, cql_num_samples=10, policy_sampler=policy_sampler_fn
            )
        # --------- TRAIN/UPDATE MODEL  # --------- 
        total_loss, metrics = trainer.compute_loss(batch, global_step=step_count)
        transformer_optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(transformer_model.parameters(), 1.0)
        transformer_optimizer.step()
        # --------- PRINT/LOG VALUES   # --------- 
        if step_count == 1  or step_count % 1000 == 0:
            # skill_head_grad = transformer_model.skill_head.weight.grad
            # print(f"{step_count}/{args.totle_iteration}, skill_head gradient norm: {skill_head_grad.norm().item():.6f}")
            # new_weight = transformer_model.skill_head.weight
            # print(f"{step_count}/{args.totle_iteration}, Weight change: {(new_weight - old_weight).norm().item():.8f}")
            print(f" ------ TRF_Step: {step_count}/{args.totle_iteration}, TRF_Los: {total_loss:.6f}, NLL_Los: {metrics['nll_loss']:.6f}, Value_Los: {metrics['value_loss']:.6f}, Adv_Mean: {metrics['advantage_mean']:.6f} , [Critic] q1={logs_c['q1_loss']:.6f}, q2={logs_c['q2_loss']:.6f}, v={logs_c['v_loss']:.6f}")
            experiment.log_metrics({'Transformer loss': total_loss.item()}, step=step_count)
            experiment.log_metrics({'NLL loss': metrics['nll_loss']}, step=step_count)
            experiment.log_metrics({'Value loss': metrics['value_loss']}, step=step_count)
            experiment.log_metrics({'Advantage mean': metrics['advantage_mean']}, step=step_count)
            experiment.log_metrics({'critic/q1_loss': logs_c['q1_loss']}, step=step_count)
            experiment.log_metrics({'critic/q2_loss': logs_c['q2_loss']}, step=step_count)
            experiment.log_metrics({'critic/v_loss':  logs_c['v_loss']}, step=step_count)
        # if iql_step_count == 1 or iql_step_count % args.save_freq == 0:
        #     save_iql_checkpoint(iql_agent, iql_step_count, f"./{checkpoint_dir}/DAE_{env_name}_{iql_step_count}.pth" )
        # --------- EVALUATION START   # --------- 
        if step_count == 1 or step_count % args.eval_freq == 0:
            print("         ================= EVALUATION START================= ", flush=True)
            eval_start_time = time.time()
            transformer_model.eval()
            total_val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    rtg, states, z, attention_mask = batch
                    batch = [b.to(device) for b in batch]
                    eval_loss, metrics = trainer.compute_loss(batch, eval=True)
                    total_val_loss += eval_loss
            transformer_model.train()
            validation_loss = total_val_loss / len(val_loader)
            print(f"Step {step_count}: Validation Loss = {validation_loss}")
            experiment.log_metric("validation/Transformer loss", validation_loss, step=step_count)
            # --------- MODEL BEST WEIGHTS SAVE   # --------- 
            if step_count > 50000:  # warmup steps
                flag_val, best_validation_loss, best_step = checkpointer.should_save_checkpoint(validation_loss, step_count)
                if flag_val or step_count == 1:  #if (validation_loss < best_validation_loss) :#and (step_count != 1) and (step_count > 200000):
                    best = 'best_NorAct' if args.normed_actions else 'best_UnNorAct'
                    save_transformer_checkpoint(transformer_model, transformer_optimizer, step_count, f"./{checkpoint_dir}/{filename}_{best}.pth")
                    # save_critics_min(f"./{checkpoint_dir}/{filename}_critics_{best}.pth", Q1, Q2, V) 
                    print(f"             ★★★ New best validation loss: {best_validation_loss:.6f} at step {step_count},")
            last_eval_duration = time.time() - eval_start_time
            minutes, seconds = divmod(int(last_eval_duration), 60);  hours, minutes = divmod(minutes, 60)
            print(f"           Evaluation finished in {hours:02d}:{minutes:02d}:{seconds:02d}s. ---")
            print("           ----------Validation END----------", flush=True)
        # --------- EARLY STOPPING OR MAX STEPS REACHED   # --------- 
        if step_count >= args.totle_iteration  or ((step_count - best_step > 100000) and step_count > 200000):
            if (step_count - best_step > 100000 and step_count > 200000):
                print("Early stopping - validation loss has plateaued")
            print(f" Transformer Step: {step_count}/{args.totle_iteration}, Training Transformer Loss: {total_loss.item():.6f} , validation loss: {validation_loss:.6f}")
            Total_time = time.time() - start_time
            minutes, seconds = divmod(int(Total_time), 60);  hours, minutes = divmod(minutes, 60)
            print(f"Total Transformer ELAPSED time: {hours:02d}:{minutes:02d}:{seconds:02d}s. ---")
            break
    #----------------------------------------------------------------------------------------#

#-----------------------------------------------------------------------------#

if __name__ == "__main__":
    main()


