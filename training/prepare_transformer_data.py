from comet_ml import Experiment
import sys, os, random , argparse, torch
sys.path.append(os.getcwd())
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Get the models module directory
models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))
# Add it to path
if models_dir not in sys.path:
    sys.path.insert(0, models_dir)
import numpy as np
from training.AE_model import GRUEncoder, Prior, DAEModel
from models.temporal_FiLM_AE import TemporalUnet_film
from models.diffusion_AE import Diffusion
from utils.utils import load_checkpoint_DAE
from datasets.dataset import SplittableSequenceDataset
from torch.optim import AdamW

#-----------------------------------------------------------------------------#
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
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


#-----------------------------------------------------------------------------#
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='antmaze-medium-diverse-v2')  #kitchen-complete-v0  'maze2d-umaze-v1' 'antmaze-umaze-v2'  'halfcheetah-medium-expert-v2'  'hopper-medium-expert-v2'  'walker2d-medium-expert-v2'  'pen-cloned-v1'  'hammer-cloned-v0'  'relocate-cloned-v0'  'door-cloned-v0'  'FetchPush-v1'
    parser.add_argument('--seed', type=int, default=5000)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--eval_freq', type=int, default=250000 )
    parser.add_argument('--save_freq', type=int, default=100000 ) 
    parser.add_argument('--totle_iteration', type=int, default= 20_000 )  #1_000_000
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--schedule', type=str, default='cosine')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--tau', type=float, default=0.005)
    parser.add_argument('--model', type=str, default='transformer')
    parser.add_argument('--z_dim', type=int, default=16)
    parser.add_argument('--h_dim', type=int, default=256)
    parser.add_argument('--beta_vae', type=float, default=0.05)
    parser.add_argument("--local_rank", type=int,   help="DDP: this script is launched with torch.distributed.run")
    parser.add_argument('--gpus',       type=int, default=1,       help='how many GPUs you want to use')
    parser.add_argument('--Load_policy',  type=bool, default=False)
    parser.add_argument('--contras_data', type=bool, default=False)
    parser.add_argument('--consistency_model', type=bool, default=True)
    parser.add_argument('--FiLM_flag',  type=bool, default=True)

    parser.add_argument('--test_split', type=float, default=0.15)

    parser.add_argument('--comet_api_key', type=str, default='', help='Comet ML API Key')
    parser.add_argument('--comet_project_name', type=str, default='Collect_dataset', help='Comet ML Project Name')
    parser.add_argument('--normed_actions', action='store_true', default=False, help='Use normalized actions (True for IQL, False for DAE)')
    parser.add_argument('--diffusion_sequence', type=int, default=0, help='1: Use diffusion sequence, 0: Use diffusion single step')
    parser.add_argument('--checkpoint_suffix', type=str, default='', help='Suffix for checkpoint directory (e.g., "1" for checkpoints1/)')
    parser.add_argument('--return_scale', type=float, default=0, help='Return scale')
    parser.add_argument('--horizon', type=int, default=12, help='Horizon for IQL')   
    parser.add_argument('--goal_flag', type=int, default=0, help='Goal flag')
    parser.add_argument('--reward_shaping', type=int, default=1, help='Use reward shaping')
    parser.add_argument('--context_length', type=int, default=16, help='Context length')

    parser.add_argument('--Sigma_clamp', type=float, default=3, help='Sigma_clamp values for AE_model clamping Zpost')
    parser.add_argument('--Z_F_loss', type=int, default=1, help='Z-force loss adding flag in AEmodel')
    parser.add_argument('--InfoL_new', type=int, default=1, help='InfoL_new updated Infoloss in AE model')
    parser.add_argument('--KLbetaend', type=float, default=0.04, help='AEmodel KLbeta loss ending weight')
    parser.add_argument('--infoloss_tau', type=float, default=0.15, help='changing tau values for info loss')
    parser.add_argument('--stride', type=int, default=1, help='with which stride collect transformt dataset')
    parser.add_argument('--prior_zsigma_clamp', type=int, default=0, help='prior_zsigma_clamp ')
    parser.add_argument('--min_clamp_zsigma', type=float, default=0.005, help='Sigma_clamp values for AE_model clamping Zpost')
    parser.add_argument('--best_weight', type=int, default=1, help='use best weights')
    parser.add_argument('--weight_cnt', type=float, default=50000, help='checkpoint step number')
    

    return parser.parse_args()
#-----------------------------------------------------------------------------#


#-----------------------------------------------------------------------------#
def main():
    # set_seed(114514)
    args = parse_args()
    gamma = args.gamma
    schedule = args.schedule
    eval_freq = args.eval_freq
    save_freq = args.save_freq
    env_name = args.env_name
    tau = args.tau
    model = args.model
    horizon = args.horizon
    n_timesteps = hyperparameters[env_name]['n_timesteps']
    lr = hyperparameters[env_name]['lr']
    w = 1.1
    z_dim = hyperparameters[env_name]['z_dim']
    try:
        load_path = hyperparameters[env_name]['load_path']
    except:
        load_path = None
    set_seed(args.seed)
    test_split = args.test_split
    H = horizon
    test_split = args.test_split
    load_path = None
    device =  torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    # Dynamic context length based on horizon for training transformer dataset
    context_length = next((ctx for threshold, ctx in [(8, 15), (12, 12), (16, 10), (22, 8), (28, 6)] if args.horizon <= threshold), 4)
    context_length = args.context_length
    if ('antmaze' in env_name   or 'maze2d' in env_name):
        args.goal_flag = 1
    #----------------------------------------------------------------------------------------#
	print("================= Preparing Datset Generators =================")
	return_scale = args.return_scale if args.return_scale != 0 else hyperparameters[env_name]['scalar'] #if any(env in args.env_name for env in ('hopper', 'halfcheetah', 'walker2d', 'kitchen')) else 10
	return_scale = 1 if ('antmaze' in env_name and args.reward_shaping == 0) else return_scale
	# if 'kitchen' in args.env_name:
	# 	return_scale = 10	# Preparing dataset Genrator
	dataset = SplittableSequenceDataset(
			env_name,
			horizon=horizon,
			returns_scale=return_scale,
			termination_penalty=None,
			load_path=load_path,
			contrastive_data = False,
			subbatchsize= 8,
			normed_actions = args.normed_actions,
			goal_flag = args.goal_flag,
			reward_shaping = args.reward_shaping,
		)
    #----------------------------------------------------------------------------------------#

    #----------------------------------------------------------------------------------------#
    print("================= Initializing and Loading Model =================")
    # Define Model Paramenrts
    state_dim = dataset.fields.observations.shape[2]
    a_dim = dataset.fields.actions.shape[2]
    z_dim = z_dim
    h_dim = 256
    wd = 0.0
    # Define Model
    use_attention = False
    film_flag =True
    dim_mults=(1, 2, 4, 8)
    encoder = GRUEncoder(state_dim, a_dim, z_dim, h_dim, Sigma_clamp=args.Sigma_clamp)
    length_sequence = horizon if args.diffusion_sequence else 1
    model = TemporalUnet_film(length_sequence, a_dim,  z_dim, attention=use_attention, dim_mults=dim_mults, state_dim=state_dim).to(device)
    diff_planner = Diffusion(z_dim, model, None, horizon=horizon, n_timesteps=n_timesteps, film_flag=film_flag, predict_epsilon=True, beta_schedule=schedule, w=w, device=device).to(device)# predict_epsilon=False
    prior   = Prior     (state_dim, z_dim, h_dim, prior_zsigma_clamp=args.prior_zsigma_clamp).to(device)
    dae_model = DAEModel(encoder, diff_planner, prior, beta=args.beta_vae, tau=tau, n_timesteps=n_timesteps, device=device,
	                diffusion_sequence=args.diffusion_sequence, goal_flag=0, env=env_name,
					Sigma_clamp=args.Sigma_clamp, min_clamp_zsigma=args.min_clamp_zsigma, Z_F_loss=args.Z_F_loss, InfoL_new=args.InfoL_new, KLbetaend=args.KLbetaend, infoloss_tau=args.infoloss_tau).to(device)
    # optimizer_DAE = torch.optim.Adam(dae_model.parameters(), lr=lr, weight_decay=wd)
    for p in dae_model.ema_model.parameters():
        p.requires_grad = False
    encoder_params = list(dae_model.encoder.parameters()) + list(dae_model.prior.parameters())
    decoder_params = list(dae_model.decoder.parameters())
    optimizer_DAE = AdamW([
		{"params": encoder_params, "lr": 3e-4, "weight_decay": 0.0},  # higher LR, no WD
		{"params": decoder_params, "lr": 1e-4, "weight_decay": 1e-4},  # baseline LR, WD ok
	])
    # Load pretrained model if exists
    checkpoint_dir = os.path.join(f'checkpoints{args.checkpoint_suffix}/', 'DAE_weights_KL/', env_name+'/')
    cnt=1
    if args.best_weight:
        filename = 'DAE_'+ env_name + '_best_NorAct.pth' if args.normed_actions else 'DAE_'+ env_name + '_best_UnNorAct.pth'
    else:
        filename = 'DAE_'+ env_name + '_best_NorAct.pth' if args.normed_actions else 'DAE_'+ env_name + '_'+ args.weight_cnt +'.pth'
    PATH_TO_CHECKPOINT = os.path.join(checkpoint_dir, filename )
    start_step = load_checkpoint_DAE(PATH_TO_CHECKPOINT, dae_model, optimizer_DAE, device)
    #----------------------------------------------------------------------------------------#


    #----------------------------------------------------------------------------------------#
    print("================= Initialzing Comet Experiment Logger =================")
    path_dataset_dir = os.path.join(f'datasets{args.checkpoint_suffix}/', 'D_transformer_dataset/', env_name+'/')   
    os.makedirs(path_dataset_dir, exist_ok=True)
    comet_proj_name = '_Collect_transformer_Dataset'
    comet_proj_name = comet_proj_name + '_NorAct' if args.normed_actions else comet_proj_name + '_UnNorAct'
    project_name_comet = args.comet_project_name + comet_proj_name
    experiment = Experiment(api_key = args.comet_api_key, project_name = project_name_comet)
    experiment.set_name(env_name)  
    experiment.log_parameters({'lr':lr,
                            'h_dim':h_dim,
                            'z_dim':z_dim,
                            'H':H,
                            'a_dim':a_dim,
                            'state_dim':state_dim,
                            'env_name':env_name,
                            'filename':filename,
                            'test_split': test_split,
                            'normed_actions': args.normed_actions,
                            'context_length': context_length})
    #----------------------------------------------------------------------------------------#
    


    #----------------------------------------------------------------------------------------#
    print("================= Creating and Saving transformer Dataset =================")
    transformer_data =dataset.build_transformer_sequences(skill_encoder=dae_model.encoder, 
                                            context_length=context_length, gamma=0.99, device=device, z_path=path_dataset_dir, stride=args.stride)
    if args.goal_flag:
        d_enc_data = 'd_transf_goal_dataset_NorAct.pkl' if args.normed_actions else 'd_transf_goal_dataset_UnNorAct.pkl'
    else:
        d_enc_data = 'd_transf_dataset_NorAct.pkl' if args.normed_actions else 'd_transf_dataset_UnNorAct.pkl'
    with open(os.path.join(path_dataset_dir, d_enc_data), 'wb') as f:
        pickle.dump(transformer_data, f)
    print(f"Encoded dataset saved with {len(transformer_data)} samples. to {os.path.join(path_dataset_dir, d_enc_data)} ")
    #----------------------------------------------------------------------------------------#

if __name__ == "__main__":
    main()


