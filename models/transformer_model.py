import torch
import torch.nn as nn
import math
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
import torch.nn.functional as F

# (You would have your other imports here for dataset loading, DAE model, etc.)

# ==============================================================================
# 1. Positional Encoding (batch_first compatible)
# ==============================================================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0)) # Shape: (1, max_len, d_model)

    # def forward(self, x: torch.Tensor) -> torch.Tensor:
    def forward(self, x: torch.Tensor, *, start_idx: int = 0) -> torch.Tensor:
        # x shape: (Batch, SeqLen, Dim)
        seq_len = x.size(1)
        x = x + self.pe[:, start_idx : start_idx + seq_len]
        # x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

# ==============================================================================
# 2. Optional: RoPE for longer contexts
# ==============================================================================
class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x):
        # x: (B, S, D)
        device = x.device
        seq_len = x.size(1)
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)   # (S, D/2)
        sin, cos = freqs.sin(), freqs.cos()
        sin = torch.repeat_interleave(sin, 2, dim=-1)       # (S, D)
        cos = torch.repeat_interleave(cos, 2, dim=-1)
        return sin, cos


def apply_rope(x, sin, cos):
    # x: (B, S, D)
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    xr = torch.stack((-x2, x1), dim=-1).reshape_as(x)      # rotate pairs
    return x * cos + xr * sin


# ==============================================================================
# 1. MLP Critic Value Functions
# ==============================================================================
class MLPBlock(nn.Module):
    def __init__(self, d_in, d_hidden):
        super().__init__()
        self.fc = nn.Linear(d_in, d_hidden)
        self.ln = nn.LayerNorm(d_hidden)    # LayerNorm inside block
        self.act = nn.GELU()                # GELU activation
    def forward(self, x):
        return self.act(self.ln(self.fc(x)))

class SkillCriticQ(nn.Module):
    """Q(s_with_goal, z) -> scalar"""
    def __init__(self, state_dim, skill_dim, hidden=512, layers=2):
        super().__init__()
        d_in = state_dim + skill_dim
        blocks = [MLPBlock(d_in, hidden)]
        for _ in range(layers-1):
            blocks.append(MLPBlock(hidden, hidden))
        self.backbone = nn.Sequential(*blocks)
        self.out = nn.Linear(hidden, 1)
        nn.init.normal_(self.out.weight, std=1e-3)   # small last layer init
        nn.init.zeros_(self.out.bias)
    def forward(self, s, z):
        h = torch.cat([s, z], dim=-1)
        h = self.backbone(h)
        return self.out(h).squeeze(-1)

class ValueCriticV(nn.Module):
    """V(s_with_goal) -> scalar"""
    def __init__(self, state_dim, hidden=512, layers=2):
        super().__init__()
        blocks = [MLPBlock(state_dim, hidden)]
        for _ in range(layers-1):
            blocks.append(MLPBlock(hidden, hidden))
        self.backbone = nn.Sequential(*blocks)
        self.out = nn.Linear(hidden, 1)
        nn.init.normal_(self.out.weight, std=1e-3)
        nn.init.zeros_(self.out.bias)
    def forward(self, s):
        h = self.backbone(s)
        return self.out(h).squeeze(-1)

@torch.no_grad()
def ema_update(target: nn.Module, online: nn.Module, tau=0.005):
    for p_t, p in zip(target.parameters(), online.parameters()):
        p_t.data.lerp_(p.data, tau)


def _masked_mean(x, m):
    return (x * m).sum() / (m.sum() + 1e-8)


def critic_td_update_doubleq_Vbackup_CQL(Q1, Q2, V, Q1_targ, Q2_targ, V_targ, critic_batch, gamma, H,
        optQ1, optQ2, optV,tau=0.005,
        cql_alpha=4.0,             # strength of conservatism (1–5 good on AntMaze ,  2 normal)
        cql_num_samples=24,        # proposal samples per (s_t)
        policy_sampler=None,        # function: sampler(s) -> [N,Dz] from current π(z|s)
        ):
    s   = critic_batch['s_t']      # (B,K,S)
    z_d = critic_batch['z_t']      # (B,K,Dz)  dataset z (normalized)
    sH  = critic_batch['s_tH']     # (B,K,S)
    rH  = critic_batch['r_H_t']    # (B,K)
    m   = critic_batch['mask']     # (B,K)
    B, K, S = s.shape
    Dz = z_d.size(-1)
    device = s.device

    # ---------- TD targets (IQL-style V-backup you already had) ----------
    with torch.no_grad():
        v_next = V_targ(sH.reshape(-1, S)).view(B, K)
        yQ = rH + (gamma**H) * v_next
        yQ = yQ.clamp(-50.0, 50.0)
    # Flatten
    s_flat  = s.reshape(-1, S)
    z_dflat = z_d.reshape(-1, Dz)
    m_flat  = m.reshape(-1)

    # Current Q
    q1 = Q1(s_flat, z_dflat).view(B, K)
    q2 = Q2(s_flat, z_dflat).view(B, K)
    v  =  V(s_flat).view(B, K)
    q1 = q1.clamp(-100.0, 100.0)
    q2 = q2.clamp(-100.0, 100.0)
    v  = v.clamp (-100.0, 100.0)

    # ---------- Losses ----------
    q1_td = F.smooth_l1_loss(q1, yQ, reduction='none')    # (B,K)
    q2_td = F.smooth_l1_loss(q2, yQ, reduction='none')    # (B,K)
    v_td  = F.mse_loss(v, yQ, reduction='none')           # (B,K)
    
    # ---------- CQL regularizer (NEW) ----------
    # Sample z~πθ(.|s) and compute log-sum-exp Q on proposals vs data Q
    # Proposals: draw from current policy; fall back to Gaussian around z_d if sampler missing
    with torch.no_grad():
        z_list = []
        if policy_sampler is not None:
            # returns [B*K, N, Dz]
            z_pol  = policy_sampler(s_flat, num=cql_num_samples, device=device)  # normalized z
            z_list.append(z_pol )
        # (b) behavior-near proposals (around dataset z)
        z_uni = torch.empty(s_flat.size(0), 12, Dz, device=device).uniform_(-3, 3)
        z_beh = z_dflat.unsqueeze(1) + 0.50 * torch.randn(s_flat.size(0), 12, Dz, device=device)
        z_list.append(z_beh)
        z_list.append(z_uni)
        z_prop = torch.cat(z_list, dim=1)                                        # [B*K, N, Dz]
        N_prop = z_prop.size(1)

    s_rep = s_flat.unsqueeze(1).expand(-1, N_prop, -1).reshape(-1, S)
    z_rep = z_prop.reshape(-1, Dz)
    q1_prop = Q1(s_rep, z_rep).view(B*K, N_prop)
    q2_prop = Q2(s_rep, z_rep).view(B*K, N_prop)

    # logsumexp over proposals, compare to data Q
    lse_q1 = torch.logsumexp(q1_prop, dim=1) - math.log(N_prop)
    lse_q2 = torch.logsumexp(q2_prop, dim=1) - math.log(N_prop)
    lse_q1 = lse_q1.view(B, K)
    lse_q2 = lse_q2.view(B, K)

    # ---------- Conservative gaps (GATED) ----------
    # Only penalize when proposals look better than data (overestimation risk).
    cql1 = F.relu(lse_q1 - q1)   # (B,K)
    cql2 = F.relu(lse_q2 - q2)   # (B,K)


    # ---------- Final losses ----------
    q1_loss = _masked_mean(q1_td, m) + cql_alpha * _masked_mean(cql1, m) + 1e-4 * (q1**2).mean()  # NEW: + L2
    q2_loss = _masked_mean(q2_td, m) + cql_alpha * _masked_mean(cql2, m) + 1e-4 * (q2**2).mean()  # NEW: + L2
    v_loss  = _masked_mean(v_td, m)

    optQ1.zero_grad(set_to_none=True); q1_loss.backward(); optQ1.step()
    optQ2.zero_grad(set_to_none=True); q2_loss.backward(); optQ2.step()
    optV .zero_grad(set_to_none=True); v_loss .backward(); optV .step()

    # Target nets
    for pt, p in zip(Q1_targ.parameters(), Q1.parameters()): pt.data.lerp_(p.data, tau)
    for pt, p in zip(Q2_targ.parameters(), Q2.parameters()): pt.data.lerp_(p.data, tau)
    for pt, p in zip(V_targ .parameters(), V .parameters()): pt.data.lerp_(p.data, tau)


    return {'q1_loss': q1_loss.item(), 'q2_loss': q2_loss.item(), 'v_loss': v_loss.item(), 'cql1_mean': _masked_mean(cql1, m).item(), 'cql2_mean': _masked_mean(cql2, m).item()}




@torch.no_grad()
def build_critic_batch_from_transformer_batch(batch, H:int, gamma:float, mean_z: torch.Tensor, std_z: torch.Tensor, stride:int=1, scaled_rtg: torch.Tensor=None):
    """
    states already include goal features. We form tuples for TD targets:
    (s_t, z_t, s_{t+H}, r_H_t, mask) for each valid t.
    """
    rtgs_znorm, states, skills, attn_mask, = batch
    B, K, S = states.shape
    Dz = skills.size(-1)
    skill_shift = max(1, H // stride) 
    valid_len = K - skill_shift
    if valid_len <= 0:
        raise ValueError(f"H={H} larger than K={K}")
    rtgs = scaled_rtg
    
    s_t   = states[:, :valid_len, :]          # (B, K-H, S)
    s_tp1 = states[:, skill_shift:, :]        # (B, K-H, S)
    z_t   = skills[:, :valid_len, :]          # (B, K-H, Dz)
    z_t = (z_t - mean_z) / std_z

    # 1-step skill TD reward: the chunk return for step t is RTG_t - γ_skill * RTG_{t+1}
    gamma_skill = gamma ** H
    r_t = rtgs[:, :valid_len, 0] - (gamma_skill) * rtgs[:, skill_shift:, 0]   # (B, K-1)


    if attn_mask is None:
        mask = torch.ones(B, valid_len, device=states.device)
    else:
        mask = (attn_mask[:, :valid_len] * attn_mask[:, skill_shift:]).float()  # both endpoints valid

    return {'s_t': s_t, 'z_t': z_t, 's_tH': s_tp1, 'r_H_t': r_t, 'mask': mask}

@torch.no_grad()
def plan_skill_cem(mu_t, log_std_t, s_t, Q1, Q2, pop=256, elites=0.2, iters=10, temp=0.30, std_scale=0.6, lam_disagree=0.15,
                   prescreen_pop=64, prescreen_topk=16, include_mu=True, step_cap_coeff=0.5, z_min=-5, z_max=5):
    device = mu_t.device
    B, Dz = mu_t.shape
    elites_k = max(1, int(pop * elites))
    # ---------- helpers ----------
    def _score(z_: torch.Tensor) -> torch.Tensor:
        # z_: [B, N, Dz]
        N = z_.size(1)
        s_rep = s_t.unsqueeze(1).expand(-1, N, -1).reshape(B * N, s_t.size(-1))
        z_rep = z_.reshape(B * N, Dz)
        q1 = Q1(s_rep, z_rep).view(B, N)
        q2 = Q2(s_rep, z_rep).view(B, N)
        return torch.minimum(q1, q2) - lam_disagree * (q1 - q2).abs()
    def _topk_elites(Z, scores, k):
        idx = torch.topk(scores, k=k, dim=1).indices                           # [B, k]
        elites_z = torch.gather(Z, 1, idx.unsqueeze(-1).expand(-1, -1, Dz))    # [B, k, Dz]
        return elites_z
    # ---------- pre-screen initialization (one-time) ----------
    base_std = torch.exp(log_std_t) * temp * std_scale                         # tighter dispersion
    presamples = mu_t[:, None, :] + base_std[:, None, :] * torch.randn(B, prescreen_pop, Dz, device=device)
    # uni = torch.empty(B, max(1, prescreen_pop // 2), Dz, device=device).uniform_(z_min, z_max)
    # presamples = torch.cat([presamples, uni], dim=1)
    if include_mu:
        presamples = torch.cat([mu_t[:, None, :], presamples], dim=1)          # ensure μ included
    # presamples = _clamp_tensor(presamples, z_min, z_max)
    prescores  = _score(presamples)
    elite0     = _topk_elites(presamples, prescores, k=prescreen_topk)
    mean       = elite0.mean(dim=1)                                            # [B, Dz]
    std        = elite0.std(dim=1).clamp_min(1e-6)                             # [B, Dz]
    # ---------- temperature schedule (anneal) ----------
    # e.g., 0.30 → 0.25 → 0.20 → 0.15 → 0.12 → 0.10 → ... (len = iters)
    temps = []
    t0, t_end = temp, 0.10
    for i in range(iters):
        a = i / max(1, iters - 1)
        temps.append(t0 + a * (t_end - t0))
    # ---------- main CEM loop ----------
    for i in range(iters):
        std_eff = (std * std_scale).clamp_min(1e-6)
        cand = mean[:, None, :] + std_eff[:, None, :] * torch.randn(B, pop, Dz, device=device)
        # cand = cand._clamp_tensor(z_min, z_max)
        if include_mu:
            # splice deterministic μ into the population (helps when landscape is flat)
            cand[:, 0, :] = mean
        scores = _score(cand)                                                  # [B, pop]
        elites_z = _topk_elites(cand, scores, k=elites_k)
        elite_mean = elites_z.mean(dim=1)                                      # [B, Dz]
        elite_std  = elites_z.std(dim=1).clamp_min(1e-6)
        # ---- Trust-region on mean update ----
        delta = elite_mean - mean                                              # [B, Dz]
        cap   = step_cap_coeff * std.mean(dim=1, keepdim=True)                 # scalar per batch
        delta_norm = delta.norm(p=2, dim=1, keepdim=True) + 1e-12
        scale = torch.clamp(cap / delta_norm, max=1.0)                         # ≤ 1 if over cap
        mean  = mean + scale * delta
        # ---- EMA on std (stability) ----
        std = (0.7 * std + 0.3 * elite_std).clamp_min(1e-6)
    return mean  # final μ as chosen skill (B, Dz)

# ==============================================================================
# 2. The Hierarchical Transformer Planner Model
# ==============================================================================
class HierarchicalTransformerPlanner(nn.Module):
    def __init__(self,
                 state_dim: int,
                 skill_dim: int,
                 context_length: int,
                 n_head: int,
                 n_layer: int,
                 d_model: int,
                 dropout: float = 0.1,
                 log_std_bounds: tuple = (-2.5, 1), #(-1.5, 2)):
                 rtg_goal_drop_p: float = 0.3,    #0.25,   # keep for RTG dropout only  ie # drop RTG ~25% of tokens during training
                 use_rope: bool = True):         # set True to use RoPE for positional encoding or set False to keep your sinusoidal PE
        super().__init__()
        self.context_length = context_length
        self.d_model = d_model
        self.skill_dim = skill_dim
        self.log_std_min, self.log_std_max = log_std_bounds
        self.rtg_goal_drop_p = rtg_goal_drop_p
        self.use_rope = use_rope
        # Modality embeddings
        self.rtg_embed = nn.Linear(1, d_model)
        self.state_embed = nn.Linear(state_dim, d_model)
        self.skill_embed = nn.Linear(skill_dim, d_model)
        # Add layer normalization for embeddings
        self.rtg_ln = nn.LayerNorm(d_model)
        self.state_ln = nn.LayerNorm(d_model)
        self.skill_ln = nn.LayerNorm(d_model)
        # Initialize embeddings more conservatively
        for layer in [self.rtg_embed, self.state_embed, self.skill_embed]:
            nn.init.normal_(layer.weight, std=0.02)
            nn.init.zeros_(layer.bias)
        # Positional encoding: RoPE (preferred) or fallback sinusoidal
        if self.use_rope:
            self.rope = RotaryPositionalEmbedding(d_model)
            self.pos_encoder = None
        else:
            self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=3 * context_length)
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_head, dim_feedforward=4 * d_model,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layer)
        # Prediction Heads
        self.skill_head = nn.Linear(d_model, 2 * skill_dim) # For mean and log_std
        self.value_head = nn.Linear(d_model, 1) # For predicting value/RTG
        self.bos = nn.Parameter(torch.zeros(1, 1, skill_dim))
        self.use_mask_token = True
        self.mask_embed = nn.Parameter(torch.zeros(1, 1, skill_dim))

        # --- after creating self.skill_head ---
        with torch.no_grad():
            nn.init.normal_(self.skill_head.weight, std=0.02)
            nn.init.zeros_(self.skill_head.bias)
            # first half of bias = mu, second half = log_std
            self.skill_head.bias[:self.skill_dim].zero_()
            self.skill_head.bias[self.skill_dim:].fill_(-1.0)   # log_std init ≈ -1.0

    def cond_heads_from_state_only(self, s_flat: torch.Tensor, device):
        """
        Lightweight proposal head π(z|s) used ONLY for CQL sampling.
        Does NOT replace the main Transformer forward.
        """
        if not hasattr(self, 'state_to_mu'):
            self.state_to_mu = nn.Sequential(
                nn.Linear(s_flat.size(-1), self.d_model),
                nn.LayerNorm(self.d_model),
                nn.GELU(),
                nn.Linear(self.d_model, self.skill_dim),
            ).to(device)
            self.state_to_logstd = nn.Sequential(
                nn.Linear(s_flat.size(-1), self.d_model),
                nn.LayerNorm(self.d_model),
                nn.GELU(),
                nn.Linear(self.d_model, self.skill_dim),
            ).to(device)
        mu = self.state_to_mu(s_flat)
        logstd = torch.clamp(self.state_to_logstd(s_flat),
                            self.log_std_min, self.log_std_max)
        return mu, logstd

    def _causal_mask(self, seq_len: int, device) -> torch.Tensor:
        return nn.Transformer.generate_square_subsequent_mask(seq_len).to(device)

    def _encode_tokens(self, rtgs, states, skills, eval_flag, Context_mask=None):
        B, K, _ = states.shape
        skills_in = torch.cat([self.bos.expand(B, 1, -1), skills[:, :-1, :]], dim=1)
        if (Context_mask is not None) and (not eval_flag):
            ctx_vis = (1.0 - Context_mask).float()  # (B,K), 1 = visible, 0 = predict
            vis_t = ctx_vis                     ## hide current position t for masked t
            vis_t_plus_1 = torch.cat([ torch.ones(B, 1, device=skills.device, dtype=ctx_vis.dtype), ctx_vis[:, :-1] ], dim=1)  # (B,K)
            vis_combined = (vis_t * vis_t_plus_1)
            if self.use_mask_token:
                skills_in = torch.where(vis_combined.unsqueeze(-1).bool(), skills_in, self.mask_embed.expand_as(skills_in))
            else:
                skills_in = skills_in * vis_combined.unsqueeze(-1)
            # skills_in = skills_in * vis_combined.unsqueeze(-1)
        # ---- RTG dropout only (train time). DO NOT touch states. ----
        if eval_flag == 0 and self.rtg_goal_drop_p > 0.0:
            drop = (torch.rand(B, K, 1, device=states.device) < self.rtg_goal_drop_p).float()
            rtgs = rtgs * (1.0 - drop)
        # Apply embeddings and layer norm
        rtg_emb = self.rtg_ln(self.rtg_embed(rtgs))
        state_emb = self.state_ln(self.state_embed(states))
        skill_emb = self.skill_ln(self.skill_embed(skills_in))
        stacked = torch.stack((rtg_emb, state_emb, skill_emb), dim=2)
        x_full = stacked.reshape(B, 3 * K, self.d_model)
        return x_full

    def forward(self, rtgs, states, skills, src_key_padding_mask=None, eval_flag=False, bidir_training=False, Context_mask=None):
        B, K, _ = states.shape
        x_full = self._encode_tokens(rtgs, states, skills, eval_flag, Context_mask=Context_mask)
        # positions
        if self.use_rope:
            orig_dtype = x_full.dtype
            x32 = x_full.float()
            sin, cos = self.rope(x32)            # (S,D)
            x32 = apply_rope(x32, sin, cos)
            x_full = x32.to(orig_dtype)
        else:
            x_full = self.pos_encoder(x_full, start_idx=0) # No offset needed for full sequence
        S = x_full.size(1) # Total sequence length, e.g., 60
        # mask: bidirectional at train, causal at eval/inference
        if bidir_training and not eval_flag:
            causal_mask = None
        else:
            causal_mask = torch.triu(torch.ones(S, S, device=x_full.device), diagonal=1).bool()
        x_out = self.transformer_encoder(x_full, mask=causal_mask, src_key_padding_mask=src_key_padding_mask)
        state_outputs = x_out[:, 1::3, :]
        skill_outputs = x_out[:, 2::3, :]  
        skill_params = self.skill_head(skill_outputs)
        value_preds  = self.value_head(state_outputs)
        mu, raw_log_std = torch.chunk(skill_params, 2, dim=-1)
        log_std = torch.clamp(raw_log_std, self.log_std_min, self.log_std_max)
        # log_std = self.log_std_min + 0.5 * (torch.tanh(raw_log_std) + 1.0) * (self.log_std_max - self.log_std_min)
        return mu, log_std, value_preds

    # Add this method to your HierarchicalTransformerPlanner class
    @staticmethod
    def _safe_causal_mask(seq_len: int, device, dtype=torch.float32) -> torch.Tensor:
        mask = torch.full((seq_len, seq_len), -1e9, device=device, dtype=dtype)
        mask = torch.triu(mask, diagonal=1) # Set upper triangle (future) to -1e9
        return mask


    @torch.no_grad()
    def sample_next_skill(self, rtgs, states, skills, src_key_padding_mask, num_history, deterministic=True, temperature=0.6, num_candidates=5, use_guidance=False):
        self.eval()
        B, K, _ = states.shape
        device = states.device
        # -------------------------------------------------------------------------------------------------- #
        # --- helpers ---
        def _as_idx_vector(nh):
            # returns a (B,) LongTensor of last valid indices (nh-1), clamped to [0, K-1]
            if isinstance(nh, int):
                idx = torch.full((B,), nh - 1, device=device)
            elif isinstance(nh, torch.Tensor):
                idx = nh.to(device).long() - 1
            else:
                idx = torch.as_tensor(nh, device=device) - 1
            return idx.clamp_(0, K - 1)
        def _gather_last_time(x, idx_vec):
            # x: (B, K, D) ; idx_vec: (B,)
            b = torch.arange(x.size(0), device=x.device)
            return x[b, idx_vec, :]
        # -------------------------------------------------------------------------------------------------- #
        # --- Prepare masks based on the padded inputs ---
        # --- Call the main forward pass ---
        mu, log_std, vpred  = self.forward(rtgs, states, skills, src_key_padding_mask, eval_flag=True)
        idx_vec  = _as_idx_vector(num_history)
        mu_t     = _gather_last_time(mu,      idx_vec) 
        log_std_t = _gather_last_time(log_std, idx_vec) 
        log_std_t = torch.clamp(log_std_t, self.log_std_min, self.log_std_max) 
        #-----------------------------------------------------------------
        state_t = _gather_last_time(states, idx_vec)   # (B,S)
        if hasattr(self, "Q1") and hasattr(self, "Q2") and self.Q1 is not None and self.Q2 is not None:
            z_star = plan_skill_cem(mu_t, log_std_t, state_t, self.Q1, self.Q2)
            return z_star
        #-----------------------------------------------------------------

# ==============================================================================
# 3. Trainer Class to Handle the Loss Calculation
# ==============================================================================
class TransformerTrainer:
    def __init__(self, model, skills_stats=None, awr_beta=3.0, device='cuda',
                 Q1=None, Q2=None, V=None, goal_dim=2, H=8, gamma=0.99, transf_mask=False):
        self.model = model
        self.awr_beta = awr_beta
        self.value_loss_fn = nn.MSELoss(reduction='none')
        self.mean_z = torch.from_numpy(skills_stats['mean_z']).float().to(device)
        self.std_z = torch.from_numpy(skills_stats['std_z']).float().to(device)
        self.Q1 = Q1
        self.Q2 = Q2
        self.V = V
        self.goal_dim = goal_dim
        self.H = H
        self.gamma = gamma
        self.device = device
        self.mask_context = transf_mask
        self.mask_mode="adaptive"
        self.mask_ratio_low = 0.2
        self.mask_ratio_high = 0.5

    def nll_gaussian_loss(self, z_true, mu, log_std):
        var = torch.exp(2.0 * log_std) + (0.05 ** 2)  # ε-floor; you can tune 0.03–0.07
        nll_tok = 0.5 * ( ((z_true - mu) ** 2) / var + torch.log(var) + math.log(2.0 * math.pi) )
        return nll_tok.mean(dim=-1), nll_tok.sum(dim=-1)  # use same object for reporting

    def _masked_mean(self, x, mask, eps=1e-8):
        # x: (B, K) ; mask: (B, K) float 0/1
        return (x * mask).sum() / (mask.sum() + eps)

    def compute_loss(self, batch, eval=False, global_step=0):
        rtgs, states, skills, attn_mask = batch
        # ---------- Valid-token mask from the dataset (B,K)----------
        mask = attn_mask.float() 
        skills = (skills - self.mean_z) / self.std_z
        # ---------- padding mask over valid tokens ----------
        padding_mask = None
        if attn_mask is not None:
            expanded_mask = attn_mask.repeat_interleave(3, dim=1)
            padding_mask = (expanded_mask == 0)
        B, K, Dz = skills.shape
        mask_context = None
        if self.mask_context:
            #--------------------------------------------------------------------------------------------------
            #--------------------------------------------------------------------------------------------------
            # ---------- Context mask over valid tokens ----------
            if eval:
                mask_ratio = torch.full((B, 1), 0.3, device=skills.device)
            else:
                mask_ratio = torch.zeros((B, 1), device=skills.device)
                # mask_ratio = r.clamp(0.3, 0.7)
            rnd = torch.rand(B, K, device=skills.device)
            Context_mask = (rnd < mask_ratio).float() * mask   # (B,K)
            # ---------- [OPTIONAL] Force-mask a valid index if all tokens are masked ----------
            needs_one = (Context_mask.sum(dim=1) == 0)
            if needs_one.any():
                # force-mask a valid index; example: last valid token per row
                last_valid = attn_mask.size(1) - 1 - torch.flip(attn_mask, dims=[1]).float().argmax(dim=1, keepdim=True)
                rows = torch.nonzero(needs_one, as_tuple=True)[0]
                Context_mask[rows, last_valid[rows, 0]] = 1.0   
            mask_context = Context_mask
        # ---------- Forward pass through the model ----------
        mu, log_std, value_preds = self.model(rtgs, states, skills, src_key_padding_mask=padding_mask, eval_flag=eval,
                                            bidir_training=not eval, Context_mask=mask_context)
        # ---------- Skill Prediction Loss (NLL) ----------
        var = torch.exp(2.0 * log_std) + (0.05 ** 2) 
        nll_tok = 0.5 * (((skills - mu) ** 2) / var + torch.log(var) + math.log(2*math.pi))
        nll_tok = nll_tok.sum(dim=-1) 
        if self.mask_context:
            nll_Context = (nll_tok * Context_mask).sum() / (Context_mask.sum() + 1e-8)
        # ---------- Advantage Weighting from critics ----------
        mask_use = mask_context if self.mask_context else mask
        with torch.no_grad():
            B, K, S = states.shape ; Dz = skills.size(-1)
            s_flat = states.reshape(-1, S)
            z_flat = skills.reshape(-1, Dz)
            # use critics attached to the model
            q1 = self.model.Q1(s_flat, z_flat).view(B, K)
            q2 = self.model.Q2(s_flat, z_flat).view(B, K)
            v  = self.model.V (s_flat).view(B, K)
            adv = (torch.minimum(q1, q2) - v)
            AW_TEMPERATURE = 2.0
            w = torch.exp((adv / (AW_TEMPERATURE + 1e-8)).clamp(max=20.0)) * mask_use
            w = torch.clamp(w, min=1e-3)  # small floor
            w = w / ((w * mask_use).sum() / (mask_use.sum() + 1e-8) + 1e-8)
        policy_loss  = (nll_tok * w).sum() / (mask_use.sum() + 1e-8)
        # --- BRAC-style behavior regularization in z-space (optional but helpful) ---
        with torch.no_grad():
            mu_b = skills  # dataset latent as behavior prior
        kl_tok = 0.5 * ( ((mu - mu_b) ** 2) / var + 2*log_std - 1 - torch.log(var) )
        kl_tok = kl_tok.sum(dim=-1)
        BRAC_BETA      = 0.02
        EXPECTILE_TAU  = 0.8
        brac_reg = BRAC_BETA * (kl_tok * mask_use).sum() / (mask_use.sum() + 1e-8)
        # ---------- Value head (expectile/IQL-like to stabilize) ----------
        def expectile_loss_masked(pred, target, tau=0.8, m=None):
            e = target - pred
            w = torch.where(e >= 0, tau, 1.0 - tau)
            if m is None:
                return (w * (e ** 2)).mean()
            return (w * (e ** 2) * m).sum() / (m.sum() + 1e-8)
        value_loss = expectile_loss_masked(value_preds.squeeze(-1), rtgs.squeeze(-1), tau=EXPECTILE_TAU, m=mask)
        # ---------- Total loss ----------
        reg = 1e-3 * log_std.mean()
        coef = self.value_coef_schedule(global_step, warmup_steps=10_000, ramp_steps=60_000, value_coef_max=0.25, value_coef_min=0.10)
        nll_Context = nll_Context if self.mask_context else policy_loss
        skill_loss = policy_loss
        total_loss = skill_loss + brac_reg  + coef * value_loss + reg
        with torch.no_grad():
            B, K, S = states.shape
            s_last = states[:, -1, :]              # (B,S)
            mu_t   = mu[:, -1, :].detach()         # (B,Dz)
            logstd_t = log_std[:, -1, :].detach()
            std_t    = torch.exp(logstd_t)
        def ramp(step, warm=5_000, peak=5e-3, hold=50_000):
            if step < warm: return peak * (step / warm)
            if step < hold: return peak
            return peak * 0.5
        if attn_mask is not None:
            probs = attn_mask.float()
            valid_counts = probs.sum(dim=1, keepdim=True)
            needs_one = (valid_counts.squeeze(1) == 0)
            if needs_one.any():
                last_valid = attn_mask.size(1) - 1 - torch.flip(attn_mask, dims=[1]).float().argmax(dim=1, keepdim=True)
                probs[needs_one, :] = 0.0
                probs[needs_one, last_valid[needs_one, 0]] = 1.0
                valid_counts = probs.sum(dim=1, keepdim=True)
            probs = probs / (valid_counts + 1e-8)
            t_idx = torch.multinomial(probs, 1).squeeze(1)
        else:
            t_idx = torch.randint(0, K, (B,), device=states.device)
        s_pick   = states[torch.arange(B, device=states.device), t_idx, :]
        mu_t     =     mu[torch.arange(B, device=states.device), t_idx, :].detach()
        logstd_t = log_std[torch.arange(B, device=states.device), t_idx, :].detach()
        std_t    = torch.exp(logstd_t)
        mu_s, logstd_s = self.model.cond_heads_from_state_only(s_pick, self.device)
        std_s          = torch.exp(torch.clamp(logstd_s, self.model.log_std_min, self.model.log_std_max))
        kl_t_s       = 0.5 * (2.0*(logstd_s - logstd_t) + ((std_t**2 + (mu_t - mu_s)**2) / (std_s**2 + 1e-8)) - 1.0).sum(dim=-1).mean()
        kl_s_t       = 0.5 * (2.0*(logstd_t - logstd_s) +((std_s**2 + (mu_s - mu_t)**2) / (std_t**2 + 1e-8)) - 1.0).sum(dim=-1).mean()
        distill_loss = 0.5*(kl_t_s + kl_s_t)
        wd_distill = ramp(global_step)
        total_loss = total_loss + wd_distill *distill_loss
        target_logstd = -0.2
        anti_collapse = 1e-3 * F.relu(target_logstd - logstd_s).mean()
        total_loss = total_loss + anti_collapse
        ################################################################################
        # skill_loss = nll_sum
        if eval:
            return total_loss.item(), {"skill_loss": nll_Context.item()}
        return total_loss, {
            "total_loss": total_loss.item(),
            "nll_loss": (skill_loss).item(),
            "value_loss": value_loss.item(),
            "advantage_mean": (adv * mask).sum().item() / (mask.sum().item() + 1e-8)
        }
    

    def beta_schedule(self, step, warmup=10_000, ramp=20_000, start=0.0, end=0.5):
        if step < warmup:                      # flat 0
            return start
        t = min(1.0, (step - warmup) / ramp)   # linear to 0.5
        return start + t * (end - start)
        
    def value_coef_schedule(self, current_step, warmup_steps=20_000, ramp_steps=50_000, value_coef_max=1, value_coef_min=0.25):
        s = current_step
        if s < warmup_steps:
            return value_coef_min
        elif s < warmup_steps + ramp_steps:
            t = (s - warmup_steps) / float(ramp_steps)
            return value_coef_min + t * (value_coef_max - value_coef_min)
        else:
            return value_coef_max
