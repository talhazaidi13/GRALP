import torch
import torch.nn as nn
import einops
from einops.layers.torch import Rearrange

from utils.helpers import (
    SinusoidalPosEmb,
    Downsample1d,
    Upsample1d,
    Conv1dBlock,
    Residual,
    PreNorm,
    LinearAttention,
)



# ---------- helper: Feature-wise Linear Modulation FiLM----------
class FusionFiLM(nn.Module):
    def __init__(self, chan, tdim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(chan + tdim, chan * 2),
            nn.GELU(),
            nn.Linear(chan * 2, chan)
        )
    def forward(self, h, eps_hat, t_emb):
        # h, eps_hat: [B, C, L] t_emb: [B, tdim]
        mod = self.mlp(torch.cat([eps_hat.mean(-1), t_emb], dim=-1))
        return h + mod[..., None]  
    
class StandardFiLMLayer(nn.Module):
    def __init__(self, feature_channels, conditioning_dim, horizon):
        super().__init__()
        # This MLP takes the conditioning_dim and outputs 2 * feature_channels
        # (one set for gamma, one for beta, for each feature channel)
        self.mlp = nn.Sequential(
            nn.Linear(conditioning_dim, feature_channels * 2), # Adjust hidden layers as needed
            nn.GELU(),
            nn.Linear(feature_channels * 2, feature_channels*4),
            nn.GELU(),
            nn.Linear(feature_channels * 4, feature_channels*2)
        )
        self.feature_channels = feature_channels
        self.horizon = horizon
        if self.horizon > 1:
            self.rearrange = Rearrange('b t c -> b c t') #  if SQUENCE

    def forward(self, h, c_embedding):
        # h: feature maps [Batch, Channels, Horizon]
        # c_embedding: combined conditioning [Batch, ConditioningDim]

        # Generate gamma and beta from c_embedding
        gamma_beta = self.mlp(c_embedding) #  Output shape: [Batch, Channels * 2]  #  if NO SQUENCE
        if self.horizon == 1:
            # Split into gamma and beta
            # Reshape them to be [Batch, Channels, 1] for broadcasting with h
            gamma = gamma_beta[:, :self.feature_channels].unsqueeze(-1)  #  if NO SQUENCE
            beta = gamma_beta[:, self.feature_channels:].unsqueeze(-1)  #  if NO SQUENCE
        else:
            gamma_beta = self.rearrange(gamma_beta)   #  if SQUENCE
            gamma = gamma_beta[:, :self.feature_channels, :]   #  if SQUENCE
            beta =  gamma_beta[:, self.feature_channels:, :]   #  if SQUENCE

        return gamma * h + beta # Apply FiLM
    


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
    if isinstance(m, nn.Embedding):
        nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
    if isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)

class ResidualTemporalBlock(nn.Module):

    def __init__(self, inp_channels, out_channels, embed_dim, horizon, kernel_size=5):
        super().__init__()
        self.blocks = nn.ModuleList([
            Conv1dBlock(inp_channels, out_channels, kernel_size),
            Conv1dBlock(out_channels, out_channels, kernel_size),
        ])

        
        if horizon == 1:
            self.time_mlp = nn.Sequential(nn.Mish(), nn.Linear(embed_dim, out_channels),Rearrange('batch t -> batch t 1'))
        else:
            self.time_mlp = nn.Sequential(nn.Mish(), nn.Linear(embed_dim, out_channels),Rearrange('b t c -> b c t'))
            
        self.residual_conv = nn.Conv1d(inp_channels, out_channels, 1) \
            if inp_channels != out_channels else nn.Identity()

    def forward(self, x, t=None):
        residual = self.residual_conv(x)
        out = self.blocks[0](x)
        if t is not None:
            out = out + self.time_mlp(t)
        out = self.blocks[1](out)
        return out + residual


class TemporalUnet_film(nn.Module):

    def __init__(
        self,
        horizon,
        transition_dim,
        cond_dim,
        dim=64,
        dim_mults=(1, 2, 4, 8),
        attention=False, state_dim=None, 
        state_dropout_start=0.7, state_dropout_end=0.3,
        state_dropout_steps=350_000
    ):
        super().__init__()
        self.state_dim = state_dim
        # dropout schedule params
        self.drop_start = state_dropout_start
        self.drop_end   = state_dropout_end
        self.drop_steps = float(state_dropout_steps)

        dims = [transition_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        print(f'[ models/temporal ] Channel dimensions: {in_out}')
        
        self.horizon = horizon
        # ---------- global embeddings (time + external cond) -------------
        time_dim = dim * 2

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, dim * 2), nn.Mish(), nn.Linear(dim * 2, dim),
        )

        self.cond_r_emb = nn.Sequential(
            nn.Linear(state_dim+cond_dim, dim), nn.Mish(),
            nn.Linear(dim, dim*2), nn.Mish(), nn.Linear(dim*2, dim)
        )

        self.state_emb = nn.Sequential(
            nn.Linear(state_dim, dim), nn.Mish(),
            nn.Linear(dim, dim)
        )
        self.z_emb = nn.Sequential(
            nn.Linear(cond_dim, dim), nn.Mish(),
            nn.Linear(dim, dim)
        )
        self.cond_fuse = nn.Linear(2*dim, dim)

        self.contrastive_embd_layer = nn.Sequential(
            nn.Linear( transition_dim   , 256),
            eval(f"nn.{'Softmax'}()"),
        )
        

        # ---------- Down path -------------------------------------
        self.downs, self.fuse_down = nn.ModuleList([]), nn.ModuleList([])
        num_resolutions = len(in_out)
        print(in_out)
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                ResidualTemporalBlock(dim_in, dim_out, embed_dim=time_dim, horizon=horizon),
                ResidualTemporalBlock(dim_out, dim_out, embed_dim=time_dim, horizon=horizon),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))) if attention else nn.Identity(),
                # Downsample1d(dim_out) if not is_last else nn.Identity()
                nn.Identity()
            ]))
            # self.fuse_down.append(FusionFiLM(dim_out, time_dim))
            self.fuse_down.append(StandardFiLMLayer(dim_out, time_dim, horizon))

            # if not is_last:
            #     horizon = horizon // 2
        
        # ---------- Mid path -------------------------------------
        mid_dim = dims[-1]
        self.mid_block1 = ResidualTemporalBlock(mid_dim, mid_dim, embed_dim=time_dim, horizon=horizon)
        self.mid_attn = Residual(PreNorm(mid_dim, LinearAttention(mid_dim))) if attention else nn.Identity()
        self.mid_block2 = ResidualTemporalBlock(mid_dim, mid_dim, embed_dim=time_dim, horizon=horizon)
        # self.fuse_mid = FusionFiLM(mid_dim, time_dim)
        self.fuse_mid = StandardFiLMLayer(mid_dim, time_dim, horizon)

        # ---------- Up path -------------------------------------
        self.ups, self.fuse_up = nn.ModuleList([]), nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(nn.ModuleList([
                ResidualTemporalBlock(dim_out * 2, dim_in, embed_dim=time_dim, horizon=horizon),
                ResidualTemporalBlock(dim_in, dim_in, embed_dim=time_dim, horizon=horizon),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))) if attention else nn.Identity(),
                # Upsample1d(dim_in) if not is_last else nn.Identity()
                nn.Identity()
            ]))
            # self.fuse_up.append(FusionFiLM(dim_in, time_dim))
            self.fuse_up.append(StandardFiLMLayer(dim_in, time_dim, horizon))
            # if not is_last:
            #     horizon = horizon * 2

        self.final_conv = nn.Sequential(
            Conv1dBlock(dim, dim, kernel_size=5),
            nn.Conv1d(dim, transition_dim, 1),
        )
        
    def _drop_p(self, global_step: int):
        if self.drop_steps <= 0:
            return self.drop_end
        r = min(max(global_step / self.drop_steps, 0.0), 1.0)
        return self.drop_start + (self.drop_end - self.drop_start) * r
      
    def forward(self, x, cond=None, time=None, eps_cond=None, use_dropout=True, force_dropout=False, return_eps=False, global_step=0):
        '''
            x : [ batch x horizon x transition ]
        '''
        x = einops.rearrange(x, 'b h t -> b t h')
        # ---- global conditioning -------------------------------
        cond_emb = self.cond_r_emb(cond)    # cond= condition+reward
        #--------------------------------------------------------------
         # 1. Split the combined conditioning vector back into state and z
        states_cond = cond[:, :self.state_dim]
        z_cond = cond[:, self.state_dim:]
        # 2. Apply State-Only Dropout (the critical step)
        if use_dropout and (not force_dropout):
            p = self._drop_p(global_step)                 # ~0.7 → 0.1
            p_full  = 0.15 #0.15
            u = torch.rand((), device=x.device)
            if u < p_full:
                states_cond = torch.zeros_like(states_cond)
                z_cond      = torch.zeros_like(z_cond)
            else:
                mask_shape = (states_cond.size(0), 1) if self.horizon==1 else (states_cond.size(0), 1, 1)
                mask = (torch.rand(mask_shape, device=x.device) > p).float()
                states_cond = states_cond * mask              # z is kept intact


        # ---- SAMPLING UNCONDITIONAL: make it match (A) exactly ----
        if force_dropout:
            states_cond = torch.zeros_like(states_cond)
            z_cond      = torch.zeros_like(z_cond)
        s_emb = self.state_emb(states_cond)
        z_emb = self.z_emb(z_cond)
        # cond_emb = self.cond_fuse(torch.cat([s_emb, z_emb], dim=-1))
        z_scale = 2.0       # try 1.5–3.0; start 2.0  
        z_gate  = 0.25      # small residual gate 0.1–0.5
        cond_emb_both = self.cond_fuse(torch.cat([s_emb, z_scale * z_emb], dim=-1))
        cond_emb = cond_emb_both + z_gate * z_emb   # makes the conditioning vector always contain a direct z-only component, so FiLM layers receive a z signal even if the s→cond path dominates.
        #--------------------------------------------------------------
        
        # if force_dropout:
        #     cond_emb = cond_emb * 0.0
        t = self.time_mlp(time)
        if self.horizon == 1:
            t = torch.cat([t, cond_emb], dim=-1)              #  if NO SQUENCE
        else:
            t_expanded = t.unsqueeze(1).expand(-1, cond_emb.shape[1], -1)
            t = torch.cat([t_expanded, cond_emb], dim=-1)

        # ---- Downsampling path ----------------------------------
        h = []
        assert torch.isnan(x).sum() == 0, 'nan in temporal'
        for (resnet, resnet2, attn, downsample ), fuse in zip(self.downs, self.fuse_down):
            x = resnet(x, t)  ; x = fuse(x, t)
            x = resnet2(x, t) ; x = fuse(x, t)
            x = attn(x)
            h.append(x)
            x = downsample(x)

        # ---- Bottleneck ----------------------------------------
        x = self.mid_block1(x, t) ; x = self.fuse_mid(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        # ---- Upsampling path -----------------------------------
        for (resnet, resnet2, attn, upsample), fuse in zip(self.ups, self.fuse_up):
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, t) ; x = fuse(x, t)
            x = resnet2(x, t) ; x = fuse(x, t)
            x = attn(x)
            x = upsample(x)
        x = self.final_conv(x)
        x = einops.rearrange(x, 'b t h -> b h t')
        return (x) if return_eps else x


class UncondTemporalUnet(nn.Module):

    def __init__(
        self,
        horizon,
        transition_dim,
        cond_dim,
        dim=64,
        dim_mults=(1, 2, 4, 8),
        attention=False, state_dim=None,
    ):
        super().__init__()
        self.state_dim = state_dim
        dims = [transition_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        print(f'[ models/temporal ] Channel dimensions: {in_out}')
        

        # ---------- global embeddings (time + external cond) -------------
        time_dim = dim 

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, dim * 2), nn.Mish(), nn.Linear(dim * 2, dim),
        )

        self.cond_r_emb = nn.Sequential(
            nn.Linear(state_dim+1, dim), nn.Mish(),
            nn.Linear(dim, dim*2), nn.Mish(), nn.Linear(dim*2, dim)
        )

        self.contrastive_embd_layer = nn.Sequential(
            nn.Linear( transition_dim   , 256),
            eval(f"nn.{'Softmax'}()"),
        )

        # ---------- Down path -------------------------------------
        self.downs, self.fuse_down = nn.ModuleList([]), nn.ModuleList([])
        num_resolutions = len(in_out)
        print(in_out)
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                ResidualTemporalBlock(dim_in, dim_out, embed_dim=time_dim, horizon=horizon),
                ResidualTemporalBlock(dim_out, dim_out, embed_dim=time_dim, horizon=horizon),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))) if attention else nn.Identity(),
                Downsample1d(dim_out) if not is_last else nn.Identity()
                # nn.Identity()
            ]))
            self.fuse_down.append(FusionFiLM(dim_out, time_dim))

            if not is_last:
                horizon = horizon // 2
        
        # ---------- Mid path -------------------------------------
        mid_dim = dims[-1]
        self.mid_block1 = ResidualTemporalBlock(mid_dim, mid_dim, embed_dim=time_dim, horizon=horizon)
        self.mid_attn = Residual(PreNorm(mid_dim, LinearAttention(mid_dim))) if attention else nn.Identity()
        self.mid_block2 = ResidualTemporalBlock(mid_dim, mid_dim, embed_dim=time_dim, horizon=horizon)
        self.fuse_mid = FusionFiLM(mid_dim, time_dim)

        # ---------- Up path -------------------------------------
        self.ups, self.fuse_up = nn.ModuleList([]), nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(nn.ModuleList([
                ResidualTemporalBlock(dim_out * 2, dim_in, embed_dim=time_dim, horizon=horizon),
                ResidualTemporalBlock(dim_in, dim_in, embed_dim=time_dim, horizon=horizon),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))) if attention else nn.Identity(),
                Upsample1d(dim_in) if not is_last else nn.Identity()
                # nn.Identity()
            ]))
            self.fuse_up.append(FusionFiLM(dim_in, time_dim))
            if not is_last:
                horizon = horizon * 2

        self.final_conv = nn.Sequential(
            Conv1dBlock(dim, dim, kernel_size=5),
            nn.Conv1d(dim, transition_dim, 1),
        )


    def forward(self, x, cond=None, time=None, eps_cond=None, use_dropout=True, force_dropout=False, return_eps=False):
        '''
            x : [ batch x horizon x transition ]
        '''

        x = einops.rearrange(x, 'b h t -> b t h')
        t = self.time_mlp(time)
        # cond_emb = self.cond_r_emb(cond)
        # if use_dropout:
        #     mask = torch.bernoulli(torch.ones([x.shape[0], 1])* 0.5).to(x.device)
        #     cond_emb = cond_emb * mask
        # if force_dropout:
        #     cond_emb = cond_emb * 0.0
        
        # t = torch.cat([t, cond_emb], dim=-1)
        eps_hat = eps_cond

        # ---- Downsampling path ----------------------------------
        h = []
        # assert torch.isnan(x).sum() == 0, 'nan in temporal'
        for (resnet, resnet2, attn, downsample ), fuse in zip(self.downs, self.fuse_down):
            x = resnet(x, t)  ; x = fuse(x, eps_hat, t)
            x = resnet2(x, t) ; x = fuse(x, eps_hat, t)
            x = attn(x)
            h.append(x)
            x = downsample(x)

        # ---- Bottleneck ----------------------------------------
        x = self.mid_block1(x, t) ; x = self.fuse_mid(x, eps_hat, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        # ---- Upsampling path -----------------------------------
        for (resnet, resnet2, attn, upsample), fuse in zip(self.ups, self.fuse_up):
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, t) ; x = fuse(x, eps_hat, t)
            x = resnet2(x, t) ; x = fuse(x, eps_hat, t)
            x = attn(x)
            x = upsample(x)
        x = self.final_conv(x)
        x = einops.rearrange(x, 'b t h -> b h t')
        return (x, eps_hat) if return_eps else x
    


