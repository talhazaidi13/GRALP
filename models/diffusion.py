
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.helpers import (cosine_beta_schedule,
                     linear_beta_schedule,
                     vp_beta_schedule,
                     extract,
                     Losses)
from utils.utils import Progress, Silent

def apply_condition(seq, cond):
    for key, value in cond.items():
        seq[:, key] = value.clone()
    return seq



# ----- Diffusion Policy -----
class Diffusion(nn.Module):
    def __init__(self, state_dim, model, max_action, horizon=20,
                 beta_schedule='linear', n_timesteps=100, film_flag=False,
                 loss_type='l2', clip_denoised=False, predict_epsilon=False, w=1, device=None):
        super(Diffusion, self).__init__()

        self.state_dim = state_dim
        self.max_action = max_action
        self.horizon = horizon
        self.model = model
        self.w = w
        
        if beta_schedule == 'linear':
            self.max_steps = 100
            betas = linear_beta_schedule(100)
        elif beta_schedule == 'cosine':
            self.max_steps = 20
            betas = cosine_beta_schedule(20)
        elif beta_schedule == 'vp':
            self.max_steps = 10
            betas = vp_beta_schedule(10)

        self.gamma = [1 for _ in np.linspace(0, 1, horizon)]
        self.gamma = torch.tensor(self.gamma, dtype=torch.float32) # shape (horizon,)
        # betas shape (n_timesteps, 1)
        betas = betas.unsqueeze(1)
        betas = betas * self.gamma # shape (n_timesteps, horizon)
        self.FiLM_flag = film_flag  # Feature-wise Linear Modulation 

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones((1, self.horizon)), alphas_cumprod[:-1]])

        self.n_timesteps = int(n_timesteps)
        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        self.device = device

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        # math: Var[x_{t-1}] = Var[x_t] * (1 - alpha_{t-1}) / (1 - alpha_t)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)

        ## log calculation clipped because the posterior variance
        ## is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped',
                             torch.log(torch.clamp(posterior_variance, min=1e-20)))
        
        self.register_buffer('posterior_mean_coef1',
                             betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
                             (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))

        self.loss_fn = Losses[loss_type]()

        # self.contrastive_embd_layer1 = nn.Sequential(
        #         nn.Linear( self.state_dim   , 256),
        #         eval(f"nn.{'Softmax'}()"),
        #     ) 

    # ------------------------------------------ sampling ------------------------------------------#

    def predict_start_from_noise(self, x_t, t, noise):
        '''
            if self.predict_epsilon, model output is (scaled) noise;
            otherwise, model predicts x0 directly
        '''
        # return x0; from x_t-1 = \sqrt(\bar{\alpha}_t) xt + \sqrt(1 - \bar{\alpha}_t) \epsilon_t

        if self.predict_epsilon:
            return (
                    extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                    extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )
        else:
            return noise

    def q_posterior(self, x_start, x_t, t):
        # caculate mean:u_q = (beta_t * sqrt(\bar{\alpha}_{t-1}) / (1 - \bar{\alpha}_t)) * x_start + \\
        #               (1 - \bar{\alpha}_{t-1}) * sqrt(\alpha_t) / (1 - \bar{\alpha}_t) * x_t
        posterior_mean = (
                extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, c, latentcritic=None, use_sfg=False, sfg_strength=0.1):
        # CLASSIFIER FREE GUIDANCE: USE MODELS 2 Times. 
        # The core idea of CFG is to amplify the influence of the condition c on the generation process, leading to samples that more strongly exhibit the desired conditional characteristics.
        epsilon_cond = self.model(x, c, t, use_dropout=False)                          # Use conditional with MASKING ie randomly masking some of the condition to zero. This helps in regulaization. like The model predicts the noise given the current noisy data x_t AND the specific condition c you want 
        epsilon_uncond = self.model(x, c, t, use_dropout=False, force_dropout=True)    # Make conditional c  to zero ie without condition predict noise like This is how the model would denoise if it weren't trying to adhere to any particular instruction.
        epsilon = epsilon_uncond + self.w*(epsilon_cond - epsilon_uncond)
        if use_sfg and latentcritic is not None and sfg_strength > 0:   # SFG: Score Feature Guidance
            with torch.enable_grad():
                x_t_for_grad = x.detach().requires_grad_(True)
                epsilon_cond_for_grad = self.model(x_t_for_grad, c, t, use_dropout=False)
                epsilon_uncond_for_grad = self.model(x_t_for_grad, c, t, use_dropout=False, force_dropout=True)
                current_epsilon_for_grad_path = epsilon_uncond_for_grad + self.w * (epsilon_cond_for_grad - epsilon_uncond_for_grad)
                pred_x0_for_critic_grad_path = self.predict_start_from_noise(x_t_for_grad, t, current_epsilon_for_grad_path)
                # pred_x0_for_critic_grad_path = self.predict_start_from_noise(x_t_for_grad, t, epsilon)
                s0_rep = c[:, :-1].view(x.shape[0], -1)
                q_vals_for_guidance = latentcritic(s0_rep, pred_x0_for_critic_grad_path.view(x.shape[0], -1))
                guidance_score = torch.autograd.grad(outputs=q_vals_for_guidance.sum(), inputs=x_t_for_grad, create_graph=False)[0]
            sigma_t_noise_scale = extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
            epsilon_to_use = epsilon - sigma_t_noise_scale * sfg_strength * guidance_score
            x_recon_effective = self.predict_start_from_noise(x, t, noise=epsilon_to_use)
            x_recon = x_recon_effective
        else:
            x_recon = self.predict_start_from_noise(x, t=t, noise=epsilon)
        if self.clip_denoised:
            x_recon.clamp_(-1., 1.)
        else:
            assert RuntimeError()
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance
    

    # @torch.no_grad()
    def p_sample(self, x, t, c, latentcritic=None):
        b, l, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, c=c, latentcritic=latentcritic)  # calculate the mean and logVariance that define the Gaussian distribution for slightly cleaner/denoised sample of input x  ie (moving from noisy xt to cleaner x0) 
        noise = torch.randn_like(x)   #A fresh sample of Gaussian noise (noise_for_step) is drawn. This introduces stochasticity into the generation process (unless the variance is zero).
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1))) 
        # x_{t-1} = mean + std * noise
        # where std = sqrt(Var[x_t] * (1 - alpha_{t-1}) / (1 - alpha_t))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise  #samples x_t−1 from the Gaussian distribution 

    # @torch.no_grad()
    def p_sample_loop(self, x, cond, reward, shape, verbose=False, return_diffusion=False, latentcritic=None):
        device = self.betas.device
        Cond_Rew = torch.cat((cond[0], reward), dim=1)
        batch_size = shape[0]
        # x = torch.randn(shape, device=device)
        # x = apply_condition(x, cond)
        if return_diffusion: diffusion = [x]
        progress = Progress(self.n_timesteps) if verbose else Silent()

        for i in reversed(range(0, self.n_timesteps)):
            timesteps = torch.full((batch_size, ), i, device=device, dtype=torch.long)
            # x = self.p_sample(x, timesteps, reward)
            # x = apply_condition(x, cond)
            # progress.update({'t': i})
            x = self.p_sample(x, timesteps, Cond_Rew, latentcritic)

            if latentcritic is not None:
                x.requires_grad_(True)
                with torch.enable_grad(): 
                    s0_rep = cond[0].view(batch_size, -1) 
                    q_vals = latentcritic(s0_rep, x.view(batch_size, -1)) 
                    grads = torch.autograd.grad(q_vals.sum(), x)[0]
                x = (x + 0.01 * grads).detach()  # refine latent each step 

            progress.update({'t': i})

            if return_diffusion: diffusion.append(x)

        progress.close()

        if return_diffusion:
            return x, torch.stack(diffusion, dim=1)
        else:
            return x

    # @torch.no_grad()
    def sample(self, x, cond, reward, latentcritic=None ,*args, **kwargs):
        batch_size = x.shape[0]
        shape = (batch_size, self.horizon, self.state_dim)
        x = self.p_sample_loop(x, cond, reward, shape,  latentcritic=latentcritic, *args, **kwargs)
        return x
    
    # ----------------------------------------- ddim sample ----------------------------------------#
    
    def ddim_sample(self, x, cond, reward, ddim_timesteps=10, ddim_discr_method="uniform", ddim_eta=0.1, clip_denoised=False, deterministic_noise_flag=0, verbose=False, return_diffusion=False, latentcritic=None):
        if ddim_discr_method == 'uniform':
            c = self.max_steps // ddim_timesteps
            ddim_timestep_seq = np.asarray(list(range(0, self.max_steps, c)))
        else:
            raise NotImplementedError()
        
        if clip_denoised:
            assert self.predict_epsilon, "clip_denoised=True requires predict_epsilon=True"
        
        ddim_timestep_seq = ddim_timestep_seq + 1
        # clip to max_steps
        # ddim_timestep_seq = np.clip(ddim_timestep_seq, 0, self.max_steps-1)
        
        # previous sequence
        # ddim_timestep_prev_seq = ddim_timestep_seq[:-1]
        ddim_timestep_prev_seq = np.append(np.array([0]), ddim_timestep_seq)

        batch_size = x.shape[0]
        device = self.betas.device
        x = apply_condition(x, cond)
        # x is pure noise
        for i in reversed(range(0, ddim_timesteps)):
            timesteps = torch.full((batch_size, ), ddim_timestep_seq[i], device=device, dtype=torch.long)
            prev_timesteps = torch.full((batch_size,), ddim_timestep_prev_seq[i], device=device, dtype=torch.long)
            
            # 1. get current and previous alpha_cumprod
            alpha_cumprod_t = extract(self.alphas_cumprod, timesteps, x.shape)  # ᾱ_{t_{s+1}}
            alpha_cumprod_t_prev = extract(self.alphas_cumprod, prev_timesteps, x.shape)  # ᾱ_{t_s}

            # 2. NOT IN STANDARD DDIM: get betas to calculate fixed standard deviation as Σθ (ai, s, z) = βiI where βi is the variance. take sqrt for std dev. 
            betas = extract(self.betas, timesteps -1 if torch.all(timesteps > 0) else timesteps , x.shape)
            std_dev_i = torch.sqrt(torch.clamp(betas, min=0.0))

            # 3. predict noise using model by Classidier free guidance CFG
            # ε_θ(x_{t_{s+1}}, t_{s+1}, c) = ε_θ(x_{t_{s+1}}, t_{s+1}, ∅) + w * (ε_θ(x_{t_{s+1}}, t_{s+1}, c) - ε_θ(x_{t_{s+1}}, t_{s+1}, ∅))
            epsilon_cond = self.model(x, reward, timesteps, use_dropout=False)
            epsilon_uncond = self.model(x, reward, timesteps, use_dropout=False, force_dropout=True)
            pred_noise = epsilon_uncond + self.w*(epsilon_cond - epsilon_uncond)
            ddim_eta = ddim_eta if deterministic_noise_flag==0 else 0.0

            # 4. get the predicted x_0  same as predict_start_from_noise step in DDPM
            # This is x̂_0 = (x_{t_{s+1}} - sqrt(1-ᾱ_{t_{s+1}})ε_θ) / sqrt(ᾱ_{t_{s+1}})
            pred_x0 = (x - torch.sqrt((1. - alpha_cumprod_t)) * pred_noise) / torch.sqrt(alpha_cumprod_t)
            if clip_denoised:
                pred_x0 = torch.clamp(pred_x0, min=-1., max=1.)


            # 5. compute variance: "sigma_t(η)" -> see formula (16)
            # This term controls the amount of stochasticity in the DDIM step and depends on the parameter η (ddim_eta in code):
            # σ_{t_{s+1}}^2 = η^2 * [(1 − ᾱ_{t_s})/(1 − ᾱ_{t_{s+1}})] * [1 − ᾱ_{t_{s+1}}/ᾱ_{t_s}]
            # σ_{t_{s+1}} = η * sqrt([(1 − ᾱ_{t_s})/(1 − ᾱ_{t_{s+1}})]) * sqrt([1 − ᾱ_{t_{s+1}}/ᾱ_{t_s}])
            # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
            # sigmas_t = ddim_eta * torch.sqrt((1 - alpha_cumprod_t_prev) / (1 - alpha_cumprod_t) * (1 - alpha_cumprod_t / alpha_cumprod_t_prev))
            term_in_sqrt =                   (1 - alpha_cumprod_t_prev) / (1 - alpha_cumprod_t) * (1 - alpha_cumprod_t / alpha_cumprod_t_prev)
            sigmas_t = ddim_eta * torch.sqrt(torch.clamp(term_in_sqrt, min=0.0)) # Clamp for stability

            # 6. compute "direction pointing to x_t" of formula (12)
            # This is sqrt(1 - ᾱ_{t_s} - σ_{t_{s+1}}^2) * ε_θ
            # pred_dir_xt = torch.sqrt(            1 - alpha_cumprod_t_prev - sigmas_t**2) * pred_noise
            coeff_dir =   torch.sqrt(torch.clamp(1 - alpha_cumprod_t_prev - sigmas_t**2, min=0.0)) 
            
            # 7. compute x_{t-1} of formula (12)
            # x_{t_s} = sqrt(ᾱ_{t_s}) * x̂_0 + "direction_term" + σ_{t_{s+1}} * z
            x_prev = (torch.sqrt(alpha_cumprod_t_prev) * pred_x0) +  (coeff_dir * pred_noise) +  (sigmas_t * torch.randn_like(x))

            # 8. NOT A STANDARD DDIM: Add Noise with fixed standard deviation
            if deterministic_noise_flag:
                if torch.any(prev_timesteps > 0):
                    random_noise_for_step = torch.randn_like(x)
                    x_prev += std_dev_i * random_noise_for_step

            # 9. NOT A STANDARD DDIM: Gradient Aided Guidance
            if latentcritic is not None:
                x_prev.requires_grad_(True)
                with torch.enable_grad(): 
                    s0_rep = cond[0].view(batch_size, -1) 
                    q_vals = latentcritic(s0_rep, x_prev.view(batch_size, -1)) 
                    grads = torch.autograd.grad(q_vals.sum(), x_prev)[0]
                x_prev = (x_prev + 0.01 * grads).detach()  # refine latent each step 

            # 9. Apply condition
            x = x_prev
            x = apply_condition(x, cond)

        return x

    # ------------------------------------------ training ------------------------------------------#

    def q_sample(self, x_start, t, noise=None, cond=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        # math: create noisy version xt of orignal input x0 by adding noise epsilon_t
        # x_{t} = \sqrt{1 - \alpha_t} * x_0 + \sqrt{\alpha_t} * \epsilon_t
        sample = (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )
        return sample

    def p_losses(self, x_start, cond, reward, t, weights=1.0, contrastive_data=None, alpha_contr=0.1):
        Cond_Rew = torch.cat((cond, reward), dim=1)
        noise = torch.randn_like(x_start)
        noise[:, 0] = 0.0
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise, cond=cond)
        x_recon = self.model(x_noisy, Cond_Rew, time=t) #x_recon.Size([64, 1, 16]) = x_noisy.Size([64, 1, 16]) Cond_Rew.Size([64, 61])
        if self.predict_epsilon:
            loss = self.loss_fn(x_recon, noise, weights)
        else:  # x 0-prediction or data-prediction diffuison approach
            # x_recon = apply_condition(x_recon, cond)
            loss = self.loss_fn(x_recon, x_start, weights)

        # if self.FiLM_flag:
        #     a_hat = self.decoder(cond[:, :d_state], x_start)
        cons_loss = 0
        if contrastive_data is not None:
            reduc_weight_posi = torch.ones( x_start.shape[0], 32 ).to(self.device) / 32
            reduc_weight_nega = torch.ones( x_start.shape[0], 32 ).to(self.device) / 32
            traj_reduce_weight = torch.ones( x_start.shape[0], 32 ).to(self.device) / 32
            Pos_states, Neg_states, Pos_rewards, Neg_rewards =contrastive_data
            # print('x_start:', x_start.shape, 'reduc_weight_posi:',reduc_weight_posi.shape, 'Pos_states:',Pos_states.shape, 'Neg_states:',Neg_states.shape)
            state_reconstr = self.predict_start_from_noise(x_start, t=t, noise=x_recon)
            x_recon_embd  = self.model.contrastive_embd_layer( state_reconstr )    # torch.Size([64, 1, 256])
            positive_embd = self.model.contrastive_embd_layer( Pos_states ) # torch.Size([64, 32, 1, 256])
            negative_embd = self.model.contrastive_embd_layer( Neg_states ) # torch.Size([64, 32, 1, 256])
            # cons_loss = self.contrastive_loss(query=x_recon_embd, positive_key=positive_embd, negative_keys=negative_embd, reduc_weight_posi=reduc_weight_posi, 
                                            #   reduc_weight_nega=reduc_weight_nega , traj_reduce_weight=traj_reduce_weight  )
            cons_loss = self.contrastive_loss(x_recon_embd, positive_embd, negative_embd, reduc_weight_posi, reduc_weight_nega, traj_reduce_weight)
        loss_all = (cons_loss* alpha_contr) + loss  # 0.1
        loss_diff = loss ; loss_contrastive = cons_loss 
        return loss_diff , loss_contrastive, loss_all, x_recon

    def loss(self, x, cond=None, reward=None, weights=1.0, contrastive_data=None, alpha_diff_cntr=0.1):
        batch_size = len(x)
        t = torch.randint(0, self.max_steps, (batch_size, ), device=x.device).long()
        return self.p_losses(x, cond, reward, t, weights, contrastive_data, alpha_diff_cntr)

    def forward(self, x, cond, reward, latentcritic=None, *args, **kwargs):
        return self.sample(x, cond, reward, latentcritic, *args, **kwargs)
    

    def contrastive_loss(self, query, positive_key, negative_keys, reduc_weight_posi, reduc_weight_nega, traj_reduce_weight, mask = None ):
        temperature = 0.1  # args
        # num_posi = positive_key.shape[1]
        # cos_positive = F.cosine_similarity(  query.unsqueeze(1) , positive_key  , -1  , eps= 1e-12 )
        # cos_negative = F.cosine_similarity(  query.unsqueeze(1) , negative_keys , -1  , eps= 1e-12  )
        # if mask is not None:
        #     cos_positive[mask] = 1
        #     cos_negative[mask] = -1   
        # exp_pos = torch.exp(cos_positive/temperature) 
        # exp_nega = torch.exp(cos_negative/temperature ) 
        # # traj_reduce_weight = torch.ones(   [1,1,Configs.horizon]    , device=exp_nega.device )
        # # exp_pos = torch.sum(  exp_pos*traj_reduce_weight , -1 )
        # # exp_nega = torch.sum(  exp_nega*traj_reduce_weight , -1 )
        # # numerator = torch.sum(exp_pos*reduc_weight_posi , -1 )/num_posi 
        # # denominator  = torch.sum(exp_nega*reduc_weight_nega , -1 )
        # exp_pos_sum = torch.sum(exp_pos.squeeze(-1)  * traj_reduce_weight, -1)  # Result: [batch_size]
        # exp_nega_sum = torch.sum(exp_nega.squeeze(-1)  * traj_reduce_weight, -1)  # Result: [batch_size]
        # numerator = torch.sum(exp_pos_sum.unsqueeze(1) * reduc_weight_posi, -1) / num_posi  # Result: [batch_size]
        # denominator = torch.sum(exp_nega_sum.unsqueeze(1) * reduc_weight_nega, -1) 
        
        temperature = 0.1 
        batch_size = query.shape[0]
        embed_dim = query.shape[-1]
        query_flat = query.view(batch_size, embed_dim)
        positive_flat = positive_key.view(batch_size, positive_key.shape[1], embed_dim)
        negative_flat = negative_keys.view(batch_size, negative_keys.shape[1], embed_dim)
        cos_positive = F.cosine_similarity(
            query_flat.unsqueeze(1),  # [batch_size, 1, embed_dim] torch.Size([64, 1, 256])
            positive_flat,  # [batch_size, num_positives, embed_dim] torch.Size([64, 32, 256]) 
            dim=2 , eps= 1e-12  # Compare along embedding dimension
        )  # Result: [batch_size, num_positives]  torch.Size([64, 32])
        cos_negative = F.cosine_similarity(
            query_flat.unsqueeze(1),  # [batch_size, 1, embed_dim]  torch.Size([64, 1, 256])
            negative_flat,  # [batch_size, num_negatives, embed_dim] torch.Size([64, 32, 256]) 
            dim=2 , eps= 1e-12  # Compare along embedding dimension
        )  # Result: [batch_size, num_negatives] torch.Size([64, 32])
        if mask is not None:
            cos_positive[mask] = 1
            cos_negative[mask] = -1
        exp_pos = torch.exp(cos_positive / temperature)
        exp_neg = torch.exp(cos_negative / temperature)
        exp_pos = exp_pos * traj_reduce_weight  # torch.Size([64, 32]) = torch.Size([64, 32]) * torch.Size([64, 32])
        exp_neg = exp_neg * traj_reduce_weight  # torch.Size([64, 32]) = torch.Size([64, 32]) * torch.Size([64, 32])
        exp_pos_sum = exp_pos.sum(dim=1)  # [batch_size]
        exp_neg_sum = exp_neg.sum(dim=1)  # [batch_size]
        # if reduc_weight_posi is not None and reduc_weight_nega is not None:
        #     exp_pos_sum = (exp_pos_sum.unsqueeze(1) * reduc_weight_posi).sum(dim=1)
        #     exp_neg_sum = (exp_neg_sum.unsqueeze(1) * reduc_weight_nega).sum(dim=1)
        loss = -torch.log(exp_pos_sum / (exp_pos_sum + exp_neg_sum + 1e-12))
        loss1 = loss.mean()

        return loss1

    def alpha_schedule(self, epoch, start_epoch=0, end_epoch=10000, start_value=0.01, end_value=0.1):
        if epoch < start_epoch:
            return start_value
        elif epoch > end_epoch:
            return end_value
        else:
            progress = (epoch - start_epoch) / (end_epoch - start_epoch)
            return start_value + progress * (end_value - start_value)


class Uncond_Diffusion(nn.Module):
    def __init__(self, state_dim, model, max_action, horizon=20,
                 beta_schedule='linear', n_timesteps=100,
                 loss_type='l2', clip_denoised=False, predict_epsilon=True):
        super(Uncond_Diffusion, self).__init__()

        self.state_dim = state_dim
        self.max_action = max_action
        self.horizon = horizon
        self.model = model

        if beta_schedule == 'linear':
            self.max_steps = 100
            betas = linear_beta_schedule(100)
        elif beta_schedule == 'cosine':
            self.max_steps = 20
            betas = cosine_beta_schedule(20)
        elif beta_schedule == 'vp':
            self.max_steps = 10
            betas = vp_beta_schedule(10)

        self.gamma = [1 for _ in np.linspace(0, 1, horizon)]
        self.gamma = torch.tensor(self.gamma, dtype=torch.float32) # shape (horizon,)
        # betas shape (n_timesteps, 1)
        betas = betas.unsqueeze(1)
        betas = betas * self.gamma # shape (n_timesteps, horizon)

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones((1, self.horizon)), alphas_cumprod[:-1]])

        self.n_timesteps = int(n_timesteps)
        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        # math: Var[x_{t-1}] = Var[x_t] * (1 - alpha_{t-1}) / (1 - alpha_t)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)

        ## log calculation clipped because the posterior variance
        ## is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped',
                             torch.log(torch.clamp(posterior_variance, min=1e-20)))
        
        self.register_buffer('posterior_mean_coef1',
                             betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
                             (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))

        self.loss_fn = Losses[loss_type]()

    # ------------------------------------------ sampling ------------------------------------------#

    def predict_start_from_noise(self, x_t, t, noise):
        '''
            if self.predict_epsilon, model output is (scaled) noise;
            otherwise, model predicts x0 directly
        '''
        # return x0; from x_t = \sqrt(\bar{\alpha}_t) x0 + \sqrt(1 - \bar{\alpha}_t) \epsilon_t

        if self.predict_epsilon:
            return (
                    extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                    extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )
        else:
            return noise

    def q_posterior(self, x_start, x_t, t):
        # caculate mean:u_q = (beta_t * sqrt(\bar{\alpha}_{t-1}) / (1 - \bar{\alpha}_t)) * x_start + \\
        #               (1 - \bar{\alpha}_{t-1}) * sqrt(\alpha_t) / (1 - \bar{\alpha}_t) * x_t
        posterior_mean = (
                extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t):
        epsilon = self.model(x, t, use_dropout=False)
        x_recon = self.predict_start_from_noise(x, t=t, noise=epsilon)
        if self.clip_denoised:
            x_recon.clamp_(-1., 1.)
        else:
            assert RuntimeError()

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    # @torch.no_grad()
    def p_sample(self, x, t):
        b, l, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t)
        noise = torch.randn_like(x)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        # x_{t-1} = mean + std * noise
        # where std = sqrt(Var[x_t] * (1 - alpha_{t-1}) / (1 - alpha_t))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    # @torch.no_grad()
    def p_sample_loop(self, x,  shape, t=None, verbose=False, return_diffusion=False):
        device = self.betas.device

        batch_size = shape[0]
        # x = torch.randn(shape, device=device)
        if t is None:
            t = self.n_timesteps
        if return_diffusion: diffusion = [x]
        progress = Progress(self.n_timesteps) if verbose else Silent()
        print(t)
        for i in reversed(range(0, t)):
            timesteps = torch.full((batch_size, ), i, device=device, dtype=torch.long)

            x = self.p_sample(x, timesteps)
            
            progress.update({'t': i})

            if return_diffusion: diffusion.append(x)

        progress.close()

        if return_diffusion:
            return x, torch.stack(diffusion, dim=1)
        else:
            return x

    # @torch.no_grad()
    def sample(self, x, *args, **kwargs):
        batch_size = x.shape[0]
        shape = (batch_size, self.horizon, self.state_dim)
        x = self.p_sample_loop(x, shape, *args, **kwargs)
        return x
    def t_sample(self, x, t, *args, **kwargs):
        batch_size = x.shape[0]
        shape = (batch_size, self.horizon, self.state_dim)
        x = self.p_sample_loop(x,  shape, t = t, *args, **kwargs)
        return x
    
    # ----------------------------------------- ddim sample ----------------------------------------#
    
    def ddim_sample(self, x, cond, reward, ddim_timesteps=20, ddim_discr_method="uniform", ddim_eta=0.1, clip_denoised=False):
        if ddim_discr_method == 'uniform':
            c = self.max_steps // ddim_timesteps
            ddim_timestep_seq = np.asarray(list(range(0, self.max_steps, c)))
        else:
            raise NotImplementedError()
        
        if clip_denoised:
            assert self.predict_epsilon, "clip_denoised=True requires predict_epsilon=True"
        
        ddim_timestep_seq = ddim_timestep_seq + 1
        # clip to max_steps
        # ddim_timestep_seq = np.clip(ddim_timestep_seq, 0, self.max_steps-1)
        
        # previous sequence
        # ddim_timestep_prev_seq = ddim_timestep_seq[:-1]
        ddim_timestep_prev_seq = np.append(np.array([0]), ddim_timestep_seq)

        batch_size = x.shape[0]
        device = self.betas.device
        x = apply_condition(x, cond)
        # x is pure noise
        for i in reversed(range(0, ddim_timesteps)):
            timesteps = torch.full((batch_size, ), ddim_timestep_seq[i], device=device, dtype=torch.long)
            prev_timesteps = torch.full((batch_size,), ddim_timestep_prev_seq[i], device=device, dtype=torch.long)
            
            # 1. get current and previous alpha_cumprod
            alpha_cumprod_t = extract(self.alphas_cumprod, timesteps, x.shape)
            alpha_cumprod_t_prev = extract(self.alphas_cumprod, prev_timesteps, x.shape)

            # 2. predict noise using model
            epsilon_cond = self.model(x, reward, timesteps, use_dropout=False)
            epsilon_uncond = self.model(x, reward, timesteps, use_dropout=False, force_dropout=True)
            pred_noise = epsilon_uncond + self.w*(epsilon_cond - epsilon_uncond)

            # 3. get the predicted x_0
            pred_x0 = (x - torch.sqrt((1. - alpha_cumprod_t)) * pred_noise) / torch.sqrt(alpha_cumprod_t)
            if clip_denoised:
                pred_x0 = torch.clamp(pred_x0, min=-1., max=1.)
            
            # 4. compute variance: "sigma_t(η)" -> see formula (16)
            # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
            sigmas_t = ddim_eta * torch.sqrt(
                (1 - alpha_cumprod_t_prev) / (1 - alpha_cumprod_t) * (1 - alpha_cumprod_t / alpha_cumprod_t_prev))
            
            # 5. compute "direction pointing to x_t" of formula (12)
            pred_dir_xt = torch.sqrt(1 - alpha_cumprod_t_prev - sigmas_t**2) * pred_noise
            
            # 6. compute x_{t-1} of formula (12)
            x_prev = torch.sqrt(alpha_cumprod_t_prev) * pred_x0 + pred_dir_xt + sigmas_t * torch.randn_like(x)

            x = x_prev
            x = apply_condition(x, cond)

        return x

    # ------------------------------------------ training ------------------------------------------#

    def q_sample(self, x_start, t, noise=None, cond=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        # math:
        # x_{t-1} = \sqrt{1 - \alpha_t} * x_0 + \sqrt{\alpha_t} * \epsilon_t
        sample = (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )
        return sample

    def p_losses(self, x_start, t, weights=1.0):
        
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_recon = self.model(x_noisy, time=t)

        if self.predict_epsilon:
            loss = self.loss_fn(x_recon, noise, weights)
        else:
            loss = self.loss_fn(x_recon, x_start, weights)

        return loss

    def loss(self, x, weights=1.0):
        batch_size = len(x)
        t = torch.randint(0, self.max_steps, (batch_size, ), device=x.device).long()
        return self.p_losses(x, t, weights)

    def forward(self, x, t=None, *args, **kwargs):
        print(t)
        if t is not None:
            return self.t_sample(x, t, *args, **kwargs)
        return self.sample(x, *args, **kwargs)