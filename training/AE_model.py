import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence
import copy
from utils.utils import alpha_schedule



def _fb_lambda_schedule(current_step, warmup_steps=200000, start_value=0.0, end_value=0.05):
    """Anneals the free bits floor (lambda) from a start to an end value."""
    if current_step >= warmup_steps:
        return end_value
    progress = current_step / warmup_steps
    return start_value + (end_value - start_value) * progress

def _soft_freebits_loss(kl_per_dim, lam, tau=0.25):
    """Applies a soft floor to the KL divergence, ensuring a non-zero gradient."""
    # smooth version of: max(kl, lam)
    return lam + F.softplus((kl_per_dim - lam) / tau) * tau

# ========= Diffusion AE Model =========

class DAEModel(nn.Module):
    def __init__(self, encoder, diff_planner, prior, beta=1.0, tau=0.001, n_timesteps=None, device='cuda:0', 
                    diffusion_sequence=False, goal_flag=False, env=None, min_clamp_zsigma=0.005,
                    Sigma_clamp=2.0, Z_F_loss=False, InfoL_new=False, KLbetaend=0.04, infoloss_tau=0.15, lambda_info=0.20, lambda_cov=0.001):
        super(DAEModel, self).__init__()
        self.encoder = encoder
        self.decoder = diff_planner
        self.prior = prior
        self.beta = beta

        self.ema = EMA(1-tau)
        self.ema_model = copy.deepcopy(self.decoder)
        self.update_ema_every = 1

        self.n_timesteps = n_timesteps
        self.device = device
        self.sqrt_alpha = self.decoder.sqrt_alphas_cumprod[n_timesteps-1].unsqueeze(-1)
        self.sqrt_one_minus_alphas_cumprod = self.decoder.sqrt_one_minus_alphas_cumprod[n_timesteps-1].unsqueeze(-1)
        self.diffusion_sequence = diffusion_sequence
        self.goal_flag = goal_flag
        self.env = env
        self.act_proj = nn.Sequential(nn.Linear(self.encoder.a_dim, 128), nn.ReLU(), nn.Linear(128, self.encoder.z_dim))
        self.max_clamp_zsigma = Sigma_clamp
        self.Z_forceloss = Z_F_loss
        self.InfoL_new=InfoL_new
        self.KLbetaend=KLbetaend
        self.infoloss_tau = infoloss_tau
        self.min_clamp_zsigma = min_clamp_zsigma
        self.lambda_info = lambda_info
        self.lambda_cov=lambda_cov

    def forward(self, batch, ep_nu=0):
        observations, next_observations, actions, rtg, conditions, goal = batch
        observations, next_observations, actions, rtg, cond, goal = self.to_device([observations, next_observations, actions, rtg.unsqueeze(-1), conditions[0], goal], self.device)
        # posterior:
        z_post_means, z_post_sigmas = self.encoder(observations, actions)      
        if torch.isnan(z_post_means).any() or torch.isnan(z_post_sigmas).any():
            z_post_means = torch.where(torch.isnan(z_post_means), torch.randn_like(z_post_means) * 0.01, z_post_means)
            z_post_sigmas = torch.where(torch.isnan(z_post_sigmas), torch.ones_like(z_post_sigmas) * 0.01, z_post_sigmas)
        z_post_sigmas = torch.clamp(z_post_sigmas, min=self.min_clamp_zsigma, max=self.max_clamp_zsigma)
        z_post_dist = Normal(z_post_means, z_post_sigmas)  # qϕ​(z∣τ)
        z_post_sampled = self.reparameterize(z_post_means, z_post_sigmas)
        # prior:        
        z_prior_means, z_prior_sigmas = self.prior(observations[:, 0:1, :]) 
        if torch.isnan(z_prior_means).any() or torch.isnan(z_prior_sigmas).any():
            z_prior_means = torch.where(torch.isnan(z_prior_means), torch.randn_like(z_prior_means) * 0.01, z_prior_means)
            z_prior_sigmas = torch.where(torch.isnan(z_prior_sigmas), torch.ones_like(z_prior_sigmas) * 0.01, z_prior_sigmas)
        z_prior_sigmas = torch.clamp(z_prior_sigmas, min=self.min_clamp_zsigma, max=self.max_clamp_zsigma)   
        z_prior_dist = Normal(z_prior_means, z_prior_sigmas)                    # pω​(z∣s0​)
        z_prior_sampled = self.reparameterize(z_prior_means, z_prior_sigmas)
        return z_post_means, z_post_sigmas, z_post_sampled, z_post_dist, z_prior_dist , z_prior_sampled
   

    def reparameterize(self, mean, std):
        eps = torch.randn_like(mean)
        return mean + std * eps

    def get_losses(self, batch, ep_nu=0, eval_flag=False):  
        observations, next_observations, actions, rtg, conditions, goal = batch
        observations, next_observations, actions, rtg, cond, goal = self.to_device([observations, next_observations, actions, rtg.unsqueeze(-1), conditions[0], goal], self.device)
        z_post_means, z_post_sigmas, z_post_sampled, z_post_dist, z_prior_dist , z_prior_sampled = self.forward(batch, ep_nu)
        ## ---------- 1) TRAINING ENCODER: Calculate Encoder/Prior loss  (KL_Loss)
        lam = _fb_lambda_schedule(ep_nu, warmup_steps=100000, end_value=0.1)  #0.02
        kl_per_dim = kl_divergence(z_post_dist, z_prior_dist)
        kl_true = torch.mean(torch.sum(kl_per_dim, dim=-1))
        soft_kl_per_dim = _soft_freebits_loss(kl_per_dim, lam=lam)
        kl_loss = torch.mean(torch.sum(soft_kl_per_dim, dim=-1))
        z_force_loss = torch.tensor(0.0, device=self.device)
        ##---------- 2) TRAINING DIFFUSION: Calculate diffusion loss (L_simple)
        if self.diffusion_sequence:
            B, T, Ds = observations.shape
            _, _, Da = actions.shape
            z_seq = z_post_sampled.expand(B, T, -1)
            if eval_flag==True:
                with torch.no_grad(): 
                    diffusion_loss , diffusion_loss_raw, x_recon= self.ema_model.loss(actions, observations, z_seq, eval_flag=eval_flag)
            else:
                diffusion_loss , diffusion_loss_raw, x_recon= self.decoder.loss(actions, observations, z_seq, eval_flag=eval_flag)
        else:
            batch_size, trajectory_len, Ds = observations.shape
            states_flat = observations.reshape(batch_size * trajectory_len, -1)
            actions_flat = actions.reshape(batch_size * trajectory_len, 1, -1)
            z_post_expanded = z_post_sampled.expand(-1, trajectory_len, -1)
            z_post_flat = z_post_expanded.reshape(batch_size * trajectory_len, -1)
            #--------------------------------------------------------------------------
            ####  Info Loss
            if self.InfoL_new:
                idxs = [actions.size(1)//4, actions.size(1)//2, 3*actions.size(1)//4]
            else:
                idxs = [actions.size(1)//2]                     # [B, adim]
            info_terms = []
            z_mu, _ = self.encoder(observations, actions)  # [B,1,Dz]; keep grads!
            k = F.normalize(z_mu.squeeze(1), dim=-1)
            tau = self.infoloss_tau#0.12
            for t_idx in idxs:
                a_t = actions[:, t_idx, :]
                q = F.normalize(self.act_proj(a_t), dim=-1)
                logits = (q @ k.t()) / tau
                labels = torch.arange(q.size(0), device=q.device)
                info_terms.append(F.cross_entropy(logits, labels))
            info_loss = sum(info_terms) / len(info_terms)  
            #--------------------------------------------------------------------------
            combined_cond_for_unet = torch.cat((states_flat, z_post_flat), dim=-1)
            # diffusion_loss , x_recon= self.decoder.loss(actions, cond, z_sampled[:,0,:])
            if eval_flag==True:
                with torch.no_grad(): 
                    z_prior_expanded = z_prior_sampled.expand(-1, trajectory_len, -1)
                    z_prior_flat = z_prior_expanded.reshape(batch_size * trajectory_len, -1)
                    diff_loss_post , diffusion_loss_raw_post, x_recon= self.decoder.loss(actions_flat, states_flat, z_post_flat, eval_flag=eval_flag)
                    diff_loss_prior , diffusion_loss_raw_prior, x_recon= self.decoder.loss(actions_flat, states_flat, z_prior_flat, eval_flag=eval_flag)
                    KL_beta = alpha_schedule(ep_nu, start_epoch=25000, end_epoch=200000, start_value=0.001, end_value=0.02) 
                    total_loss = diff_loss_post + KL_beta * kl_loss 
                    return {"val_total_loss": total_loss, "val_diffusion_loss_post": diff_loss_post, "val_diffusion_loss_prior": diff_loss_prior, "val_kl_loss": kl_loss , "val_kl_true": kl_true}
            else:
                diffusion_loss , diffusion_loss_raw, x_recon= self.decoder.loss(actions_flat, states_flat, z_post_flat, eval_flag=eval_flag, global_step=ep_nu)
        #--------------------------------------------------------------------------
        if not eval_flag and self.Z_forceloss:
            ####  z-force Loss : Make the decoder care about z (directly)
            z_samp = z_post_sampled.detach()  # take a sampled z, no encoder gradient
            if z_samp.dim() == 3:
                z_samp = z_samp.squeeze(1) 
            z_flat = z_samp.repeat_interleave(trajectory_len, dim=0)
            z_flat.requires_grad_(True) 
            t_mid = torch.full((actions_flat.size(0),), self.n_timesteps // 2, device=self.device)
            x_mid = self.decoder.q_sample(actions_flat, t_mid)  # noisy actions
            a_pred = self.decoder.model(x_mid, torch.cat([states_flat, z_flat], dim=-1),time=t_mid,  use_dropout=True, force_dropout=False, global_step=ep_nu,)  # predict actions
            g_flat  = torch.autograd.grad(a_pred.mean(), z_flat, create_graph=True, allow_unused=True, retain_graph=True)[0]  # derivative wrt z. compute how much the predicted action changes if z changes.
            if g_flat is None:
                z_force_loss = torch.tensor(0.0, device=self.device)
            else:
                g = g_flat.view(batch_size, trajectory_len, -1).sum(dim=1)
                g_norm = torch.clamp(g.norm(dim=1), min=1e-3, max=self.max_clamp_zsigma)
                z_force_loss = -torch.log(g_norm.mean() + 1e-8)   # loss is large if the gradient is small.
        #--------------------------------------------------------------------------        
        # ---------- 3) TOTAL LOSS: Combine diffusion loss and KL loss with a beta schedule
        KL_beta = alpha_schedule(ep_nu, start_epoch=25000, end_epoch=100000, start_value=0.0005, end_value=self.KLbetaend) #0.025
        total_loss = diffusion_loss + KL_beta * kl_loss 
        # sigma_reg = 1e-4 * (torch.log(z_post_sigmas)**2).mean()
        z_force_w = 0.01 * min(1.0, ep_nu / 20000.0)   # cap 1%, slow ramp
        lambda_info =  self.lambda_info  #0.20  #0.10
        total_loss = diffusion_loss + KL_beta * kl_loss + lambda_info * info_loss + z_force_w * z_force_loss
        #--------------------------------------------------------------------------
        z_means_centered = z_post_means - z_post_means.mean(dim=0, keepdim=True)
        var_vec = (z_means_centered ** 2).mean(dim=0)
        target_var = 1.0
        L_cov = ((var_vec - target_var) ** 2).mean()
        lambda_cov = self.lambda_cov
        total_loss = total_loss + lambda_cov * L_cov
        return {"total_loss": total_loss, "diffusion_loss": diffusion_loss,  "diffusion_loss_raw": diffusion_loss_raw,  "kl_loss": kl_loss,  "info_loss": info_loss}

    
    
    
    def reset_parameters(self):
        self.ema_model.load_state_dict(self.decoder.state_dict())

    def step_ema(self, step, step_start_ema):
        if step < step_start_ema:
            self.reset_parameters()
            return
        if step % self.update_ema_every == 0:
            self.ema.update_model_average(self.ema_model, self.decoder)

    def to_device(self, data, device):
        if isinstance(data, (list, tuple)):
            return [self.to_device(x, device) for x in data]
        elif isinstance(data, dict):
            return {k: self.to_device(v, device) for k, v in data.items()}
        elif hasattr(data, 'to'):
            return data.to(device)
        return data

# ========= Encoder Module =========
class GRUEncoder(nn.Module):
    """
    GRU-based encoder that concatenates state and action,
    applies a linear embedding, and then processes via a bidirectional GRU.
    """
    def __init__(self, state_dim, a_dim, z_dim, h_dim, n_gru_layers=4, normalize_latent=False, Sigma_clamp=2.0):
        super(GRUEncoder, self).__init__()
        self.state_dim = state_dim
        self.a_dim = a_dim
        self.normalize_latent = normalize_latent
        self.z_dim = z_dim

        self.emb_layer = nn.Sequential(
            # nn.LayerNorm(state_dim),
            nn.Linear(state_dim, h_dim),
            nn.GELU(),
            nn.Linear(h_dim, h_dim),
            nn.GELU()
        )
        self.rnn = nn.GRU(h_dim + a_dim, h_dim, batch_first=True, bidirectional=True, num_layers=n_gru_layers)
        self.mean_layer = nn.Sequential(
            nn.Linear(2 * h_dim, h_dim),
            nn.GELU(),
            nn.Linear(h_dim, z_dim)
        )
        self.sig_layer = nn.Sequential(
            nn.Linear(2 * h_dim, h_dim),
            nn.GELU(),
            nn.Linear(h_dim, z_dim),
            nn.Softplus()
        )
        self.max_clamp_zsigma = Sigma_clamp
        self.skip = nn.Linear(state_dim, h_dim, bias=False)

    def forward(self, states, actions):
        s_emb = self.emb_layer(states) + self.skip(states)
        s_emb_a = torch.cat([s_emb, actions], dim=-1)
        feats, _ = self.rnn(s_emb_a)
        # Using the final time step representation
        hn = feats[:, -1:, :]
        z_mean = self.mean_layer(hn)
        z_sigma = self.sig_layer(hn)
        # NaN detection and handling
        if torch.isnan(z_mean).any() or torch.isnan(z_sigma).any():
            print("WARNING: NaN detected in encoder output!")
            # Replace NaN values with small random values
            z_mean = torch.where(torch.isnan(z_mean), 
                               torch.randn_like(z_mean) * 0.01, 
                               z_mean)
            z_sigma = torch.where(torch.isnan(z_sigma), 
                                torch.ones_like(z_sigma) * 0.01, 
                                z_sigma)
        
        # Ensure positive standard deviations
        z_sigma = torch.clamp(z_sigma, min=0.005, max=self.max_clamp_zsigma)
        
        if self.normalize_latent:
            z_mean = z_mean / torch.norm(z_mean, dim=-1, keepdim=True)
        return z_mean, z_sigma






# ========= Decoder and LowLevel Policy Modules =========
class Decoder(nn.Module):
    def __init__(self, state_dim, a_dim, z_dim, h_dim=256, a_dist='normal'):
        """
        Decoder module that uses an autoregressive low-level policy.
        
        Args:
            state_dim (int): Dimension of state.
            a_dim (int): Dimension of action.
            z_dim (int): Dimension of latent variable.
            h_dim (int): Hidden dimension.
            a_dist (str): Distribution type for actions.
        """
        super(Decoder, self).__init__()
        self.state_dim = state_dim
        self.a_dim = a_dim
        self.z_dim = z_dim
        self.a_dist = a_dist
        
        self.ll_policy = AutoregressiveLowLevelPolicy(
            state_dim, a_dim, z_dim, h_dim, a_dist=a_dist, fixed_sig=None
        )
        self.state_dynamics = StateDynamics(state_dim,z_dim,h_dim)

    def forward(self, states, actions, z, state_decoder=False):
        """
        Decodes latent z and states into action distribution parameters.
        """
        s_0 = states[:,0:1,:]
        a_mean, a_sig = self.ll_policy(states, actions, z)
        if state_decoder:
            sT_mean, sT_sig = self.state_dynamics(s_0, z.detach())
            return sT_mean, sT_sig, a_mean, a_sig
        return a_mean, a_sig


class StateDynamics(nn.Module):
    '''
    P(s_T|s_0,z) is our "abstract dynamics model", because it predicts the resulting state transition over T timesteps given a skill 
    (so similar to regular dynamics model, but in skill space and also temporally extended)
    See Encoder and Decoder for more description
    '''
    def __init__(self,state_dim,z_dim,h_dim,per_element_sigma=True):
        super(StateDynamics,self).__init__()
        self.layers = nn.Sequential(nn.Linear(state_dim+z_dim,h_dim),nn.ReLU(),nn.Linear(h_dim,h_dim),nn.ReLU())
        self.mean_layer = nn.Sequential(nn.Linear(h_dim,h_dim),nn.ReLU(),nn.Linear(h_dim,state_dim))
        self.sig_layer  = nn.Sequential(nn.Linear(h_dim,h_dim),nn.ReLU(),nn.Linear(h_dim,state_dim),nn.Softplus())
        self.state_dim = state_dim
        self.per_element_sigma = per_element_sigma

    def forward(self,s0,z):
        s0_z = torch.cat([s0,z],dim=-1)
        feats = self.layers(s0_z)
        sT_mean = self.mean_layer(feats)
        sT_sig  = self.sig_layer(feats)
        if not self.per_element_sigma:
            sT_sig = torch.cat(self.state_dim*[sT_sig],dim=-1)
        return sT_mean,sT_sig



class AutoregressiveLowLevelPolicy(nn.Module):
    def __init__(self, state_dim, a_dim, z_dim, h_dim, a_dist='normal', fixed_sig=None):
        """
        Implements an autoregressive policy that decodes a latent into actions.
        """
        super(AutoregressiveLowLevelPolicy, self).__init__()
        self.a_dim = a_dim
        self.a_dist = a_dist
        # Create a low-level policy for each action dimension.
        self.policy_components = nn.ModuleList([
            LowLevelPolicy(state_dim + i, 1, z_dim, h_dim, a_dist=a_dist, fixed_sig=fixed_sig)
            for i in range(a_dim)
        ])

    def forward(self, state, actions, z):
        """
        Autoregressively generates action means and sigmas.
        
        Args:
            state (Tensor): [B, T, state_dim]
            actions (Tensor): [B, T, a_dim] (previously generated actions)
            z (Tensor): [B, 1, z_dim]
        
        Returns:
            Tuple: (a_means [B, T, a_dim], a_sigmas [B, T, a_dim])
        """
        a_means_list = []
        a_sigmas_list = []
        for i in range(self.a_dim):
            # Concatenate the previously generated actions along the feature dimension.
            state_a = torch.cat([state, actions[:, :, :i]], dim=-1)
            a_mean_i, a_sig_i = self.policy_components[i](state_a, z)  # [B, T, 1]
            a_means_list.append(a_mean_i)
            if self.a_dist != 'softmax':
                a_sigmas_list.append(a_sig_i)
        a_means = torch.cat(a_means_list, dim=-1)
        a_sigmas = torch.cat(a_sigmas_list, dim=-1)
        return a_means, a_sigmas
    
    def sample(self,state,z):
        actions = []
        for i in range(self.a_dim):
            # Concat state, a up to i, and z_tiled
            state_a = torch.cat([state]+actions,dim=-1)
            # pass through ith policy component
            a_mean_i, a_sig_i = self.policy_components[i](state_a,z)  # these are batch_size x T x 1
            a_i = self.reparameterize(a_mean_i,a_sig_i)
            #a_i = a_mean_i
            if self.a_dist == 'tanh_normal':
                a_i = nn.Tanh()(a_i)
            actions.append(a_i)

        return torch.cat(actions,dim=-1)
    
    def reparameterize(self, mean, std):
        eps = torch.randn_like(mean)
        return mean + std * eps
    
    def numpy_policy(self,state,z, device='cuda:0'):
        '''
        maps state as a numpy array and z as a pytorch tensor to a numpy action
        '''
        state = torch.reshape(torch.tensor(state, device=torch.to(device), dtype=torch.float32), (1,1,-1))
        action = self.sample(state,z)
        # action = action.detach().cpu().numpy()
        return action.reshape([self.a_dim,])


class LowLevelPolicy(nn.Module):
    def __init__(self, state_dim, a_dim, z_dim, h_dim, a_dist='normal', fixed_sig=None):
        """
        A single low-level policy component to produce action parameters.
        """
        super(LowLevelPolicy, self).__init__()
        self.a_dist = a_dist
        self.a_dim = a_dim
        self.fixed_sig = fixed_sig

        self.layers = nn.Sequential(
            nn.Linear(state_dim + z_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU()
        )
        self.mean_layer = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, a_dim)
        )
        self.sig_layer = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, a_dim)
        )

    def forward(self, state, z):
        """
        Forward pass for low-level policy.
        
        Args:
            state (Tensor): [B, T, state_dim]
            z (Tensor): [B, 1, z_dim]
        
        Returns:
            Tuple: (action_mean, action_sigma) both of shape [B, T, a_dim]
        """
        # Tile latent z to match the time dimension of state.
        z_tiled = z.tile([1,state.shape[-2],1]) 
        state_z = torch.cat([state, z_tiled], dim=-1)
        feats = self.layers(state_z)
        a_mean = self.mean_layer(feats)
        a_sigma = nn.Softplus()(self.sig_layer(feats))
        return a_mean, a_sigma

# ========= Prior Module =========

class Prior(nn.Module):
    def __init__(self, state_dim, z_dim, h_dim, goal_dim = 0, prior_zsigma_clamp=0):
        """
        Prior network to generate a latent distribution from the initial state (and goal).
        
        Args:
            state_dim (int): Dimension of state.
            z_dim (int): Dimension of latent variable.
            goal_dim (int): Dimension of goal (if applicable).
            h_dim (int): Hidden dimension.
        """
        super(Prior, self).__init__()
        self.state_dim = state_dim
        self.z_dim = z_dim
        self.goal_dim = goal_dim
        self.layers = nn.Sequential(
            nn.Linear(state_dim + goal_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU()
        )
        self.mean_layer = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, z_dim)
        )
        self.sig_layer = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, z_dim),
            nn.Softplus()
        )
        self.prior_zsigma_clamp = prior_zsigma_clamp
    def forward(self, s0, goal=None):
        """
        Forward pass for the prior network.
        
        Args:
            s0 (Tensor): Initial state, shape [B, 1, state_dim].
            goal (Tensor, optional): Goal information.
        
        Returns:
            Tuple: (z_mean, z_sigma) for the prior.
        """
        # If goal is not provided, we assume a default zero tensor.
 
        feats = self.layers(s0)
        z_mean = self.mean_layer(feats)
        z_sigma = self.sig_layer(feats)
        if self.prior_zsigma_clamp:
            z_sigma = 1.5 + 1.5 * torch.tanh(z_sigma / 2) 
        if torch.isnan(z_mean).any() or torch.isnan(z_sigma).any():
            print("WARNING: NaN detected in Prior output!")
        return z_mean, z_sigma
 

    def get_loss(self,states,actions,goal=None):
        '''
        To be used only for low level action Prior training
        '''
        a_mean, a_sig = self.forward(states,goal)

        a_dist = Normal(a_mean,a_sig)
        return - torch.mean(a_dist.log_prob(actions))
    



#==========================================================================================================================================#
#=======================================================  EMA CLASS ========================================================================#
#==========================================================================================================================================#


class EMA():
    '''
        empirical moving average
    '''
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new