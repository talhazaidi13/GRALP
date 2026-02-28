from comet_ml import Experiment
import sys, os, pickle , random , argparse, torch, gym, time, datetime, math
sys.path.append(os.getcwd())
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Get the models module directory
models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))
# Add it to path
if models_dir not in sys.path:
    sys.path.insert(0, models_dir)
import numpy as np
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import DataLoader
from torch.optim import AdamW                 # NEW
# Import mixed precision components
from torch.cuda.amp import GradScaler	
from training.AE_model import  GRUEncoder, Prior, DAEModel

# from models.temporal_FiLM import TemporalUnet_film
from models.temporal_FiLM_AE import TemporalUnet_film
# from models.diffusion import Diffusion
from models.diffusion_AE import Diffusion
from utils.helpers import cycle
from datasets.dataset import SplittableSequenceDataset

from pathlib import Path
script_path = Path(__file__).resolve()
project_root = script_path.parent.parent
os.chdir(project_root)
from typing import Dict
os.environ['PYOPENGL_PLATFORM'] = 'osmesa'


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
    pure_training_time = elapsed_time - time_spent_in_validation
    pure_training_time = max(0, pure_training_time)
    avg_time_per_step = pure_training_time / steps_done if steps_done > 0 else 0
    steps_remaining = total_steps - steps_done
    eta_training_seconds = steps_remaining * avg_time_per_step
    total_evals_to_run = total_steps // eval_freq
    evals_remaining = total_evals_to_run - evals_done
    eta_validation_seconds = evals_remaining * last_eval_duration
    total_eta_seconds = eta_training_seconds + eta_validation_seconds
    eta_seconds_int = int(total_eta_seconds)
    minutes, seconds = divmod(eta_seconds_int, 60)
    hours, minutes = divmod(minutes, 60)
    eta_formatted = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    elapsed_seconds_int = int(elapsed_time)
    e_m, e_s = divmod(elapsed_seconds_int, 60)
    e_h, e_m = divmod(e_m, 60)
    elapsed_formatted = f"{e_h:02d}:{e_m:02d}:{e_s:02d}"
    avg_time_per_100_step_formatted = f"{(avg_time_per_step * 100):.2f}s"
    return elapsed_formatted, avg_time_per_100_step_formatted, eta_formatted



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
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='antmaze-umaze-diverse-v2') 
    parser.add_argument('--seed', type=int, default=5000)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--eval_freq', type=int, default=250_000 )
    parser.add_argument('--save_freq', type=int, default=50_000 ) 
    parser.add_argument('--totle_iteration', type=int, default= 500_000 )  #1_000_000
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--schedule', type=str, default='cosine')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--tau', type=float, default=0.001)
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
    parser.add_argument('--comet_api_key', type=str, default='', help='Comet ML API Key') # Added
    parser.add_argument('--comet_project_name', type=str, default='DAE_KL', help='Comet ML Project Name')
    parser.add_argument('--normed_actions', action='store_true', default=False, help='Use normalized actions')
    parser.add_argument('--use_mixed_precision', action='store_true', default=False, help='Use mixed precision training (FP16)')
    parser.add_argument('--diffusion_sequence', type=int, default=0, help='1: Use diffusion sequence, 0: Use diffusion single step')
    parser.add_argument('--checkpoint_suffix', type=str, default='', help='Suffix for checkpoint directory (e.g., "1" for checkpoints1/)')
    parser.add_argument('--return_scale', type=float, default=0, help='Return scale')
    parser.add_argument('--horizon', type=int, default=12, help='Horizon for IQL')   
    parser.add_argument('--goal_flag', type=int, default=0, help='Use goal flag')
    parser.add_argument('--reward_shaping', type=int, default=0, help='Use reward shaping')

    parser.add_argument('--Sigma_clamp', type=float, default=3, help='Sigma_clamp values for AE_model clamping Zpost')
    parser.add_argument('--Z_F_loss', type=int, default=1, help='Z-force loss adding flag in AEmodel')
    parser.add_argument('--InfoL_new', type=int, default=1, help='InfoL_new updated Infoloss in AE model')
    parser.add_argument('--KLbetaend', type=float, default=0.04, help='AEmodel KLbeta loss ending weight')
    parser.add_argument('--infoloss_tau', type=float, default=0.15, help='changing tau values for info loss')
    parser.add_argument('--prior_zsigma_clamp', type=int, default=0, help='prior_zsigma_clamp ')
    parser.add_argument('--min_clamp_zsigma', type=float, default=0.005, help='Sigma_clamp values for AE_model clamping Zpost')
    parser.add_argument('--lambda_info', type=float, default=0.20, help='in DAE training for info loss weight, makes the decoder actually uses z')
    parser.add_argument('--lambda_cov', type=float, default=0.001, help='lambda coverience weight for CDAE training')
    return parser.parse_args()


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



def main():
	# set_seed(114514)
	args = parse_args()
	gamma = args.gamma
	print("#"*100)
	print("\n📥 Received args:\n" + "\n".join(f"{k:>25} : {v}" for k, v in vars(args).items()))
	print("#"*100)
	schedule = args.schedule
	env_name = args.env_name
	save_freq = 10000 if 'pen-expert' in env_name else args.save_freq 
	tau = args.tau
	model = args.model
	horizon = args.horizon
	n_timesteps = hyperparameters[env_name]['n_timesteps']
	lr = hyperparameters[env_name]['lr']
	w = 1.1
	rtg = hyperparameters[env_name]['rtg']
	z_dim = hyperparameters[env_name]['z_dim']
	try:
		load_path = hyperparameters[env_name]['load_path']
	except:
		load_path = None
	set_seed(args.seed)
	env = gym.make(env_name)
	batch_size = args.batch_size
	test_split = args.test_split
	H = horizon
	test_split = args.test_split
	load_from_checkpoint = False
	train_diffusion_prior = False
	beta = args.beta_vae # 1.0 # 0.1, 0.01, 0.001
	scale = None
	load_path = None
	device =  torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

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
	# Get train and test splits
	train_dataset, test_dataset = dataset.get_train_test_split(test_size=0.2)
	# Create data loaders
	train_loader = DataLoader(
		train_dataset, 
		batch_size=batch_size, 
		shuffle=True, 
		num_workers=args.num_workers if hasattr(args, 'num_workers') else 0
	)
	test_loader = DataLoader(
		test_dataset, 
		batch_size=batch_size, 
		shuffle=False,  # Usually no need to shuffle test data
		num_workers=args.num_workers if hasattr(args, 'num_workers') else 0
	)
	# Verify the split
	print(f"Total dataset size: {len(dataset)}")
	print(f"Training examples: {len(train_loader.dataset)} ({len(train_loader.dataset)/len(dataset):.1%})")
	print(f"Testing examples: {len(test_loader.dataset)} ({len(test_loader.dataset)/len(dataset):.1%})")
	#----------------------------------------------------------------------------------------#




	#----------------------------------------------------------------------------------------#
	print("================= Initializing Model =================")
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
	                diffusion_sequence=args.diffusion_sequence, goal_flag=args.goal_flag, env=env_name,
					Sigma_clamp=args.Sigma_clamp, min_clamp_zsigma=args.min_clamp_zsigma, Z_F_loss=args.Z_F_loss, InfoL_new=args.InfoL_new, 
					KLbetaend=args.KLbetaend, infoloss_tau=args.infoloss_tau, lambda_info=args.lambda_info, lambda_cov=args.lambda_cov).to(device)
	# optimizer = torch.optim.Adam(dae_model.parameters(), lr=lr, weight_decay=wd)
	# optimizer  = AdamW(dae_model.parameters(), lr=1e-4, weight_decay=1e-4)  # Reduced from 2e-4 to 1e-4
	for p in dae_model.ema_model.parameters():
		p.requires_grad = False
	encoder_params = list(dae_model.encoder.parameters()) + list(dae_model.prior.parameters())
	decoder_params = list(dae_model.decoder.parameters())
	optimizer = AdamW([
		{"params": encoder_params, "lr": 3e-4, "weight_decay": 0.0},  # higher LR, no WD
		{"params": decoder_params, "lr": 1e-4, "weight_decay": 1e-4},  # baseline LR, WD ok
	])
	scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.totle_iteration, eta_min=1e-5)  # Reduced from 2e-5 to 1e-5
	
	# Initialize mixed precision training
	use_mixed_precision = args.use_mixed_precision and torch.cuda.is_available()
	scaler = GradScaler() if use_mixed_precision else None
	nan_detected = False  # Track if NaN was detected
	
	if use_mixed_precision:
		print("✅ Mixed precision training enabled (FP16)")
	else:
		print("❌ Mixed precision training disabled (FP32)")
    #----------------------------------------------------------------------------------------#




	#----------------------------------------------------------------------------------------#
	print("================= Initialzing Comet Experiment Logger =================")
	# checkpoint_dir = os.path.join('checkpoints/', 'DAE_weights_KL/', args.env_name+'/') 
	checkpoint_dir = os.path.join(f'checkpoints{args.checkpoint_suffix}/', 'DAE_weights_KL/', args.env_name+'/')    
	os.makedirs(checkpoint_dir, exist_ok=True)
	filename = 'DAE_'+ env_name
	comet_proj_name = '_Train_AE'
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
							'l2_reg':wd,
							'beta':beta,
							'env_name':env_name,
							'filename':filename,
							'train_diffusion_prior': train_diffusion_prior,
							'test_split': test_split,
							'normed_actions': args.normed_actions,
							'use_mixed_precision': use_mixed_precision})
    #----------------------------------------------------------------------------------------#

	


	#----------------------------------------------------------------------------------------#
	if load_from_checkpoint:
		PATH = os.path.join(checkpoint_dir,filename+'_best_sT.pth')
		checkpoint = torch.load(PATH)
		dae_model.load_state_dict(checkpoint['model_state_dict'])
    #----------------------------------------------------------------------------------------#



	#----------------------------------------------------------------------------------------#
	print("=================TRAINING START=================", flush=True)
	cnt = 0; step_start_ema = 5000; 
	start_time = time.time(); 
	for _, batch in enumerate(cycle(train_loader)):
		cnt += 1
		loss: Dict = dae_model.get_losses(batch, cnt)
		optimizer.zero_grad(set_to_none=True)
		loss['total_loss'].backward()
		torch.nn.utils.clip_grad_norm_(encoder_params, 1.0)
		torch.nn.utils.clip_grad_norm_(decoder_params, 1.0)
		optimizer.step()
		scheduler.step()
		dae_model.step_ema(cnt, step_start_ema)
		if cnt % 1000 ==0  or cnt == 1:
			experiment.log_metrics(loss, step=cnt)
		if cnt % 1000 ==0  or cnt == 1:
			elapsed_time, avg_time_per_100_step, eta_formatted = calculate_ETA(start_time, cnt, args.totle_iteration, args.save_freq, 0 )
			print(f"EstT:{eta_formatted}, ElpT:{elapsed_time}, T/100E:{avg_time_per_100_step}, ------ Tr_step:{cnt}, " + ", ".join( f"{key.split('/', 1)[-1]}: {value:.4f}" for key, value in loss.items()), flush=True)
		# --------- EARLY STOPPING OR MAX STEPS REACHED   # --------- 
		if cnt >= args.totle_iteration :
			print(f"Tr_step:{cnt}, " + ", ".join(f"{key}: {value:.4f}" for key, value in loss.items()))
			break

	#--------------------------------------------#

if __name__ == "__main__":
    main()


