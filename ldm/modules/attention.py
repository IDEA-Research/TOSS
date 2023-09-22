from inspect import isfunction
import math
import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat
from typing import Optional, Any

from ldm.modules.diffusionmodules.util import checkpoint
import pdb

try:
    import xformers
    import xformers.ops
    XFORMERS_IS_AVAILBLE = True
except:
    XFORMERS_IS_AVAILBLE = False

# CrossAttn precision handling
import os
_ATTN_PRECISION = os.environ.get("ATTN_PRECISION", "fp32")
# _ATTN_PRECISION = os.environ.get("ATTN_PRECISION", "fp16")

def exists(val):
    return val is not None


def uniq(arr):
    return{el: True for el in arr}.keys()


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def max_neg_value(t):
    return -torch.finfo(t.dtype).max


def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor


# feedforward
class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class SpatialSelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b,c,h,w = q.shape
        q = rearrange(q, 'b c h w -> b (h w) c')
        k = rearrange(k, 'b c h w -> b c (h w)')
        w_ = torch.einsum('bij,bjk->bik', q, k)

        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = rearrange(v, 'b c h w -> b c (h w)')
        w_ = rearrange(w_, 'b i j -> b j i')
        h_ = torch.einsum('bij,bjk->bik', v, w_)
        h_ = rearrange(h_, 'b c (h w) -> b c h w', h=h)
        h_ = self.proj_out(h_)

        return x+h_


class SelfAttention(nn.Module):
    def __init__(self, query_dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(query_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim), nn.Dropout(dropout) )

    def forward(self, x):
        q = self.to_q(x) # B*N*(H*C)
        k = self.to_k(x) # B*N*(H*C)
        v = self.to_v(x) # B*N*(H*C)

        B, N, HC = q.shape 
        H = self.heads
        C = HC // H 

        q = q.view(B,N,H,C).permute(0,2,1,3).reshape(B*H,N,C) # (B*H)*N*C
        k = k.view(B,N,H,C).permute(0,2,1,3).reshape(B*H,N,C) # (B*H)*N*C
        v = v.view(B,N,H,C).permute(0,2,1,3).reshape(B*H,N,C) # (B*H)*N*C

        sim = torch.einsum('b i c, b j c -> b i j', q, k) * self.scale  # (B*H)*N*N
        attn = sim.softmax(dim=-1) # (B*H)*N*N

        out = torch.einsum('b i j, b j c -> b i c', attn, v) # (B*H)*N*C
        out = out.view(B,H,N,C).permute(0,2,1,3).reshape(B,N,(H*C)) # B*N*(H*C)

        return self.to_out(out)


class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask=None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        # force cast to fp32 to avoid overflowing
        if _ATTN_PRECISION =="fp32":
            with torch.autocast(enabled=False, device_type = 'cuda'):
                q, k = q.float(), k.float()
                sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        else:
            sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        
        del q, k
    
        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        sim = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', sim, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)


class MemoryEfficientCrossAttention(nn.Module):
    # https://github.com/MatthieuTPHR/diffusers/blob/d80b531ff8060ec1ea982b65a1b8df70f73aa67c/src/diffusers/models/attention.py#L223
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        print(f"Setting up {self.__class__.__name__}. Query dim is {query_dim}, context_dim is {context_dim} and using "
              f"{heads} heads.")
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.heads = heads
        self.dim_head = dim_head

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim), nn.Dropout(dropout))
        self.attention_op: Optional[Any] = None

    def forward(self, x, context=None, mask=None):
        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        b, _, _ = q.shape
        q, k, v = map(
            lambda t: t.unsqueeze(3)
            .reshape(b, t.shape[1], self.heads, self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b * self.heads, t.shape[1], self.dim_head)
            .contiguous(),
            (q, k, v),
        )

        # actually compute the attention, what we cannot get enough of
        out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=None, op=self.attention_op)

        if exists(mask):
            raise NotImplementedError
        out = (
            out.unsqueeze(0)
            .reshape(b, self.heads, out.shape[1], self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b, out.shape[1], self.heads * self.dim_head)
        )
        return self.to_out(out)


class MemoryEfficientCrossAttention_pose(nn.Module):
    # https://github.com/MatthieuTPHR/diffusers/blob/d80b531ff8060ec1ea982b65a1b8df70f73aa67c/src/diffusers/models/attention.py#L223
    def __init__(self, query_dim, context_dim=None, pos_dim=None, heads=8, dim_head=64, img_size=32, dropout=0.0):
        super().__init__()
        print(f"Setting up {self.__class__.__name__}. Query dim is {query_dim}, context_dim is {context_dim} and using "
              f"{heads} heads.")
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim), nn.Dropout(dropout))
        self.attention_op: Optional[Any] = None

        self.to_q_pose = nn.Linear(pos_dim, img_size*img_size*self.heads*20*40//dim_head, bias=False)
        self.to_k_pose = nn.Linear(pos_dim, img_size*img_size*self.heads*20*40//dim_head, bias=False)

    def forward(self, x, context, pos_emb, mask=None):
        b, l, d = x.shape
        h = self.heads
        # content proj
        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        b, _, _ = q.shape
        q, k, v = map(
            lambda t: t.unsqueeze(3)
            .reshape(b, t.shape[1], self.heads, self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b * self.heads, t.shape[1], self.dim_head)
            .contiguous(),
            (q, k, v),
        )

        # pos emb proj
        q_pos = self.to_q_pose(pos_emb).reshape(b,l,-1)
        k_pos = self.to_k_pose(pos_emb).reshape(b,l,-1)
        q_pos, k_pos = map(
            lambda t: t.unsqueeze(3)
            .reshape(b, t.shape[1], self.heads, -1)
            .permute(0, 2, 1, 3)
            .reshape(b * self.heads, t.shape[1], -1)
            .contiguous(),
            (q_pos, k_pos),
        )

        q = torch.cat([q, q_pos], dim=-1)
        k = torch.cat([k, k_pos], dim=-1)

        # actually compute the attention, what we cannot get enough of
        # out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=None, op=self.attention_op)

        # if exists(mask):
        #     raise NotImplementedError
        # out = (
        #     out.unsqueeze(0)
        #     .reshape(b, self.heads, out.shape[1], self.dim_head)
        #     .permute(0, 2, 1, 3)
        #     .reshape(b, out.shape[1], self.heads * self.dim_head)
        # )

        # an unefficient implementation
        # force cast to fp32 to avoid overflowing
        if _ATTN_PRECISION =="fp32":
            with torch.autocast(enabled=False, device_type = 'cuda'):
                q, k = q.float(), k.float()
                sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        else:
            sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        
        del q, k
    
        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        sim = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', sim, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)


class GatedCrossAttentionDense(nn.Module):
    ATTENTION_MODES = {
        "softmax": CrossAttention,  # vanilla attention
        "softmax-xformers": MemoryEfficientCrossAttention
    }
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        attn_mode = "softmax-xformers" if XFORMERS_IS_AVAILBLE else "softmax"
        assert attn_mode in self.ATTENTION_MODES
        attn_cls = self.ATTENTION_MODES[attn_mode]
        self.attn = attn_cls(query_dim=query_dim, context_dim=context_dim,
                              heads=heads, dim_head=dim_head, dropout=dropout) 
        
        self.norm1 = nn.LayerNorm(query_dim)
        self.alpha_attn = nn.Parameter(torch.tensor(0.), requires_grad=True)

        # self.ff = FeedForward(query_dim, glu=True)
        # self.norm2 = nn.LayerNorm(query_dim)
        # self.alpha_dense = nn.Parameter(torch.tensor(0.), requires_grad=True)
        
        # self.register_parameter('alpha_attn', nn.Parameter(torch.tensor(0.)) )
        # self.register_parameter('alpha_dense', nn.Parameter(torch.tensor(0.)) )

        # this can be useful: we can externally change magnitude of tanh(alpha)
        # for example, when it is set to 0, then the entire model is same as original one 
        self.scale = 1  

    def forward(self, x, context=None):
        x = x + self.scale*torch.tanh(self.alpha_attn) * self.attn(self.norm1(x), context=context)  
        # x = x + self.scale*torch.tanh(self.alpha_dense) * self.ff(self.norm2(x)) 
        
        return x 

class GatedSelfAttentionDense(nn.Module):
    def __init__(self, query_dim, context_dim,  n_heads, d_head, enable_gate_attn=True):
        super().__init__()
        
        # we need a linear projection since we need cat visual feature and obj feature
        self.linear = nn.Linear(context_dim, query_dim)

        self.attn = SelfAttention(query_dim=query_dim, heads=n_heads, dim_head=d_head)
        self.ff = FeedForward(query_dim, glu=True)

        self.norm1 = nn.LayerNorm(query_dim)
        self.norm2 = nn.LayerNorm(query_dim)

        if enable_gate_attn:
            self.register_parameter('alpha_attn', nn.Parameter(torch.tensor(0.)) )
            self.register_parameter('alpha_dense', nn.Parameter(torch.tensor(0.)) )

        # this can be useful: we can externally change magnitude of tanh(alpha)
        # for example, when it is set to 0, then the entire model is same as original one 
        self.scale = 1  
        self.enable_gate_attn = enable_gate_attn

    def forward(self, x, objs):

        N_visual = x.shape[1]
        objs = self.linear(objs)
        
        if self.enable_gate_attn:
            x = x + self.scale*torch.tanh(self.alpha_attn) * self.attn(self.norm1(torch.cat([x,objs],dim=1)))[:,0:N_visual,:]
            x = x + self.scale*torch.tanh(self.alpha_dense) * self.ff(self.norm2(x))  
        else:
            x = x + self.scale * self.attn(self.norm1(torch.cat([x,objs],dim=1)))[:,0:N_visual,:]
            x = x + self.scale * self.ff(self.norm2(x))  
        
        return x 




# cross attn between condition img and noisy img
class BasicTransformerBlock_temp(nn.Module):
    ATTENTION_MODES = {
        "softmax": CrossAttention,  # vanilla attention
        "softmax-xformers": MemoryEfficientCrossAttention
    }
    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None, gated_ff=True, checkpoint=True,
                 disable_self_attn=False, gate_mode='cross_attn', enable_gate_attn=False):
        super().__init__()
        attn_mode = "softmax-xformers" if XFORMERS_IS_AVAILBLE else "softmax"
        # attn_mode = "softmax"

        assert attn_mode in self.ATTENTION_MODES
        attn_cls = self.ATTENTION_MODES[attn_mode]
        self.gate_mode = gate_mode
        self.disable_self_attn = disable_self_attn
        self.attn1 = attn_cls(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout,
                              context_dim=context_dim if self.disable_self_attn else None)  # is a self-attention if not self.disable_self_attn
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = attn_cls(query_dim=dim, context_dim=context_dim,
                              heads=n_heads, dim_head=d_head, dropout=dropout)  # is self-attn if context is none

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)

        # adding a new layer
        if gate_mode == "cross_attn":
            self.attn_mid = attn_cls(query_dim=dim, context_dim=context_dim,
                                heads=n_heads, dim_head=d_head, dropout=dropout)
            self.norm_mid = nn.LayerNorm(dim)
        else:
            self.attn_mid = GatedSelfAttentionDense(query_dim=dim, context_dim=context_dim, \
                                    n_heads=n_heads, d_head=d_head, enable_gate_attn=enable_gate_attn)
        
        self.checkpoint = checkpoint

    def forward(self, x, context=None, pos_emb=None):
        return checkpoint(self._forward, (x, context, pos_emb), self.parameters(), self.checkpoint)

    def _forward(self, x, context=None, pos_emb=None):
        # divide cond and target
        batch_size = x.shape[0]
        x_cond = x[batch_size//2:,...]
        x = x[:batch_size//2,...]

        # cross attn between cond and target
        # x_cross = torch.cat([x,x_cond], dim=1)
        x = self.attn1(self.norm1(x), self.norm1(x_cond)) + x

        # mid attn
        if self.gate_mode == "cross_attn":
            x = self.attn_mid(self.norm_mid(x), context=pos_emb) + x
        else:
            x = self.attn_mid(x, objs=pos_emb)

        # text attn
        x = self.attn2(self.norm2(x), context=context) + x

        # reconcat
        x = torch.cat([x, x_cond], dim=0)

        # ff
        x = self.ff(self.norm3(x)) + x

        return x



# cross attn between condition img and noisy img
class BasicTransformerBlock_vae(nn.Module):
    ATTENTION_MODES = {
        "softmax": CrossAttention,  # vanilla attention
        "softmax-xformers": MemoryEfficientCrossAttention
    }
    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None, gated_ff=True, checkpoint=True,
                 disable_self_attn=False, gate_mode='cross_attn', enable_gate_attn=False, pos_emb_dim=320):
        super().__init__()
        attn_mode = "softmax-xformers" if XFORMERS_IS_AVAILBLE else "softmax"

        assert attn_mode in self.ATTENTION_MODES
        attn_cls = self.ATTENTION_MODES[attn_mode]
        self.gate_mode = gate_mode
        self.disable_self_attn = disable_self_attn
        self.attn1 = attn_cls(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout,
                              context_dim=context_dim if self.disable_self_attn else None)  # is a self-attention if not self.disable_self_attn
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = attn_cls(query_dim=dim, context_dim=context_dim,
                              heads=n_heads, dim_head=d_head, dropout=dropout)  # is self-attn if context is none

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        
        self.checkpoint = checkpoint

        self.vae_proj = nn.Linear(pos_emb_dim, dim)

    def forward(self, x, context=None, pos_emb=None):
        return checkpoint(self._forward, (x, context, pos_emb), self.parameters(), self.checkpoint)

    def _forward(self, x, context=None, pos_emb=None):
        # divide cond and target
        batch_size = x.shape[0]
        x_cond = x[batch_size//2:,...]
        x = x[:batch_size//2,...]

        # cross attn between cond and target
        # x_cross = torch.cat([x_cond, pos_emb.repeat(1,1,x.shape[-1]//320)], dim=-2)
        x_cross = torch.cat([x_cond, self.vae_proj(pos_emb)], dim=-2)
        x = self.attn1(self.norm1(x), self.norm1(x_cross)) + x

        # text attn
        x = self.attn2(self.norm2(x), context=context) + x

        # reconcat
        x = torch.cat([x, x_cond], dim=0)

        # ff
        x = self.ff(self.norm3(x)) + x

        return x



class BasicTransformerBlock(nn.Module):
    ATTENTION_MODES = {
        "softmax": CrossAttention,  # vanilla attention
        "softmax-xformers": MemoryEfficientCrossAttention
    }
    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None, gated_ff=True, checkpoint=True,
                 disable_self_attn=False):
        super().__init__()
        attn_mode = "softmax-xformers" if XFORMERS_IS_AVAILBLE else "softmax"
        attn_mode = "softmax"
        assert attn_mode in self.ATTENTION_MODES
        attn_cls = self.ATTENTION_MODES[attn_mode]
        self.disable_self_attn = disable_self_attn
        self.attn1 = attn_cls(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout,
                              context_dim=context_dim if self.disable_self_attn else None)  # is a self-attention if not self.disable_self_attn
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = attn_cls(query_dim=dim, context_dim=context_dim,
                              heads=n_heads, dim_head=d_head, dropout=dropout)  # is self-attn if context is none
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint

    def forward(self, x, context=None):
        return checkpoint(self._forward, (x, context), self.parameters(), self.checkpoint)

    def _forward(self, x, context=None):
        x = self.attn1(self.norm1(x), context=context if self.disable_self_attn else None) + x
        x = self.attn2(self.norm2(x), context=context) + x
        x = self.ff(self.norm3(x)) + x
        return x


class SpatialTransformer(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    NEW: use_linear for more efficiency instead of the 1x1 convs
    """
    def __init__(self, in_channels, n_heads, d_head,
                 depth=1, dropout=0., context_dim=None,
                 disable_self_attn=False, use_linear=False,
                 use_checkpoint=True):
        super().__init__()
        if exists(context_dim) and not isinstance(context_dim, list):
            context_dim = [context_dim]
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = Normalize(in_channels)
        if not use_linear:
            self.proj_in = nn.Conv2d(in_channels,
                                     inner_dim,
                                     kernel_size=1,
                                     stride=1,
                                     padding=0)
        else:
            self.proj_in = nn.Linear(in_channels, inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim[d],
                                disable_self_attn=disable_self_attn, checkpoint=use_checkpoint)
                for d in range(depth)]
        )
        if not use_linear:
            self.proj_out = zero_module(nn.Conv2d(inner_dim,
                                                  in_channels,
                                                  kernel_size=1,
                                                  stride=1,
                                                  padding=0))
        else:
            self.proj_out = zero_module(nn.Linear(in_channels, inner_dim))
        self.use_linear = use_linear

    def forward(self, x, context=None):
        # note: if no context is given, cross-attention defaults to self-attention
        if not isinstance(context, list):
            context = [context]
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        if not self.use_linear:
            x = self.proj_in(x)
        x = rearrange(x, 'b c h w -> b (h w) c').contiguous()
        if self.use_linear:
            x = self.proj_in(x)
        for i, block in enumerate(self.transformer_blocks):
            x = block(x, context=context[i])
        if self.use_linear:
            x = self.proj_out(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w).contiguous()
        if not self.use_linear:
            x = self.proj_out(x)
        return x + x_in




class SpatialTransformer_gate(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    NEW: use_linear for more efficiency instead of the 1x1 convs
    """
    def __init__(self, in_channels, n_heads, d_head,
                 depth=1, dropout=0., context_dim=None,
                 disable_self_attn=False, use_linear=False,
                 use_checkpoint=True, enable_gate_attn=True, gate_mode='cross_attn',
                 temp_attn="SA", img_size=32):
        super().__init__()
        if exists(context_dim) and not isinstance(context_dim, list):
            context_dim = [context_dim]
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = Normalize(in_channels)
        if not use_linear:
            self.proj_in = nn.Conv2d(in_channels,
                                     inner_dim,
                                     kernel_size=1,
                                     stride=1,
                                     padding=0)
        else:
            self.proj_in = nn.Linear(in_channels, inner_dim)

        if enable_gate_attn:
            self.transformer_blocks = nn.ModuleList(
                [BasicTransformerBlock_gated(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim[d],
                                    disable_self_attn=disable_self_attn, checkpoint=use_checkpoint, gate_mode=gate_mode)
                    for d in range(depth)]
            )
        elif temp_attn == "CA":
            self.transformer_blocks = nn.ModuleList(
                [BasicTransformerBlock_temp(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim[d],
                                    disable_self_attn=disable_self_attn, checkpoint=use_checkpoint, gate_mode=gate_mode, enable_gate_attn=enable_gate_attn)
                    for d in range(depth)]
            )
        elif temp_attn == "CA_vae":
            self.transformer_blocks = nn.ModuleList(
                [BasicTransformerBlock_vae(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim[d],
                                    disable_self_attn=disable_self_attn, checkpoint=use_checkpoint, gate_mode=gate_mode, enable_gate_attn=enable_gate_attn)
                    for d in range(depth)]
            )

        else:
            self.transformer_blocks = nn.ModuleList(
                [BasicTransformerBlock_mid(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim[d],
                                    disable_self_attn=disable_self_attn, checkpoint=use_checkpoint, gate_mode=gate_mode, enable_gate_attn=enable_gate_attn)
                    for d in range(depth)]
            )

        if not use_linear:
            self.proj_out = zero_module(nn.Conv2d(inner_dim, in_channels, kernel_size=1, stride=1, padding=0))
        else:
            self.proj_out = zero_module(nn.Linear(in_channels, inner_dim))
        self.use_linear = use_linear
        self.temp_attn = temp_attn

    def forward(self, x, context=None, pos_emb=None):
        # note: if no context is given, cross-attention defaults to self-attention
        if not isinstance(context, list):
            context = [context]
        if not isinstance(pos_emb, list):
            pos_emb = [pos_emb]

        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        if not self.use_linear:
            x = self.proj_in(x)
        x = rearrange(x, 'b c h w -> b (h w) c').contiguous()
        if self.use_linear:
            x = self.proj_in(x)
        for i, block in enumerate(self.transformer_blocks):
            x = block(x, context=context[i], pos_emb=pos_emb[i])
        if self.use_linear:
            x = self.proj_out(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w).contiguous()
        if not self.use_linear:
            x = self.proj_out(x)

        return x + x_in

