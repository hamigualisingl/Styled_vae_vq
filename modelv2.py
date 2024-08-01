from collections import OrderedDict
from dataclasses import dataclass
import logging
import math
from einops import rearrange
from typing import Tuple, Union, Callable, Optional
import numpy as np
import torch
from typing import Mapping, Text, Tuple
from torch.nn.functional import cosine_similarity
from einops import reduce
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint
##下面的是stylegan2的生成器,我用来作为解码器的,现在弃用了
from torch import nn, einsum
from torch import distributed as dist
def swish(x):
    # swish
    return x*torch.sigmoid(x)
# from omegaconf import OmegaConf
def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))
class VectorQuantizer(torch.nn.Module):
    def __init__(self,
                 codebook_size: int = 1024,
                 token_size: int = 256,
                 commitment_cost: float = 0.25,
                 use_l2_norm: bool = False,
                 ):
        super().__init__()
        self.commitment_cost = commitment_cost

        self.embedding = torch.nn.Embedding(codebook_size, token_size)
        #self.embedding.weight.data.uniform_(-1.0 / codebook_size, 1.0 / codebook_size)
        #self.use_l2_norm = use_l2_norm#true

    # Ensure quantization is performed using f32
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, Mapping[Text, torch.Tensor]]:
        z = z.float()
        #z = rearrange(z, 'b c h w -> b h w c').contiguous()
        z_flattened = rearrange(z, 's b c -> (s b) c')
        #z_flattened = rearrange(z, 'b h w c -> (b h w) c')

        embedding = self.embedding.weight
        d = torch.sum(z_flattened**2, dim=1, keepdim=True) + \
            torch.sum(embedding**2, dim=1) - 2 * \
            torch.einsum('bd,dn->bn', z_flattened, embedding.T)

        min_encoding_indices = torch.argmin(d, dim=1) # num_ele
        z_quantized = self.get_codebook_entry(min_encoding_indices).view(z.shape)
        # compute loss for embedding
        #commitment_loss = self.commitment_cost * torch.mean((z_quantized.detach() - z) **2)
        #codebook_loss = torch.mean((z_quantized - z.detach()) **2)

        #loss = commitment_loss + codebook_loss
        z_quantized = z + (z_quantized - z).detach()
       
        # reshape back to match original input shape
        #z_quantized = rearrange(z_quantized, 'b h w c -> b c h w').contiguous()

        result_dict = dict(
            #quantizer_loss=loss,
            #commitment_loss=commitment_loss,
            #codebook_loss=codebook_loss,
            #min_encoding_indices=min_encoding_indices.view(z_quantized.shape[0], z_quantized.shape[2], z_quantized.shape[3])
        )

        return z_quantized, result_dict

    def get_codebook_entry(self, indices):
        if len(indices.shape) == 1:
            z_quantized = self.embedding(indices)
        elif len(indices.shape) == 2:
            z_quantized = torch.einsum('bd,dn->bn', indices, self.embedding.weight)
        else:
            raise NotImplementedError
        return z_quantized
class Conv2dSame(nn.Conv2d):
    def calc_same_pad(self, i: int, k: int, s: int, d: int) -> int:
        return max((math.ceil(i / s) - 1) * s + (k - 1) * d + 1 - i, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ih, iw = x.size()[-2:]

        pad_h = self.calc_same_pad(i=ih, k=self.kernel_size[0], s=self.stride[0], d=self.dilation[0])
        pad_w = self.calc_same_pad(i=iw, k=self.kernel_size[1], s=self.stride[1], d=self.dilation[1])

        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])
        return super().forward(x)

class ResnetBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int = None,
        dropout_prob: float = 0.0,
        num_groups: int = 32
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.out_channels_ = self.in_channels if self.out_channels is None else self.out_channels

        self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)
        self.conv1 = Conv2dSame(self.in_channels, self.out_channels_, kernel_size=3, bias=False)

        self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=self.out_channels_, eps=1e-6, affine=True)
        self.dropout = nn.Dropout(dropout_prob)
        self.conv2 = Conv2dSame(self.out_channels_, self.out_channels_, kernel_size=3, bias=False)

        if self.in_channels != self.out_channels_:
            self.nin_shortcut = Conv2dSame(self.out_channels_, self.out_channels_, kernel_size=1, bias=False)

    def forward(self, hidden_states):
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states = F.silu(hidden_states)
        hidden_states = self.conv1(hidden_states)

        hidden_states = self.norm2(hidden_states)
        hidden_states = F.silu(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states)

        if self.in_channels != self.out_channels_:
            residual = self.nin_shortcut(hidden_states)

        return hidden_states + residual
class FP32LayerNorm(nn.LayerNorm):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        origin_dtype = inputs.dtype
        return F.layer_norm(
            inputs.float(), self.normalized_shape, self.weight.float(), self.bias.float(), self.eps
        ).to(origin_dtype)

class AdaLayerNorm_stled(nn.Module):
    r"""
    Norm layer modified to incorporate timestep embeddings.

    Parameters:
        embedding_dim (`int`): The size of each embedding vector.
        num_embeddings (`int`): The size of the embeddings dictionary.
    """

    def __init__(self, embedding_dim: int, elementwise_affine=True, eps=1e-6):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, 2*embedding_dim)
        self.norm = FP32LayerNorm(embedding_dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        shift = self.linear(self.silu(emb.to(torch.float32)).to(emb.dtype))#    12 768*2
        alpha, beta = shift.chunk(2, dim=-1)  # Split into alpha and beta (12, 768) each
        
        # Normalize input x
        x = self.norm(x)  # (257, 12, 768)
        alpha = alpha.unsqueeze(0)  # Add batch dimension, shape becomes (1, 12, 768)
        beta = beta.unsqueeze(0)  # Add batch dimension, shape becomes (1, 12, 768)
        x = alpha * x + beta  # Broadcasting should work correctly
        return x
class LayerNorm(nn.LayerNorm):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        origin_dtype = inputs.dtype
        return F.layer_norm(
            inputs.float(), self.normalized_shape, self.weight.float(), self.bias.float(), self.eps
        ).to(origin_dtype)
class QuickGELU(nn.Module):
    # NOTE This is slower than nn.GELU or nn.SiLU and uses more GPU memory
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)
class Attention(nn.Module):
    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=True,
            scaled_cosine=False,
            scale_heads=False,
            logit_scale_max=math.log(1. / 0.01),
            attn_drop=0.,
            proj_drop=0.
    ):
        super().__init__()
        self.scaled_cosine = scaled_cosine
        self.scale_heads = scale_heads
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.logit_scale_max = logit_scale_max

        # keeping in_proj in this form (instead of nn.Linear) to match weight scheme of original
        self.in_proj_weight = nn.Parameter(torch.randn((dim * 3, dim)) * self.scale)
        if qkv_bias:
            self.in_proj_bias = nn.Parameter(torch.zeros(dim * 3))
        else:
            self.in_proj_bias = None

        if self.scaled_cosine:
            self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))))
        else:
            self.logit_scale = None
        self.attn_drop = nn.Dropout(attn_drop)
        if self.scale_heads:
            self.head_scale = nn.Parameter(torch.ones((num_heads, 1, 1)))
        else:
            self.head_scale = None
        self.out_proj = nn.Linear(dim, dim)
        self.out_drop = nn.Dropout(proj_drop)

    def forward(self, x, attn_mask: Optional[torch.Tensor] = None):
        L, N, C = x.shape
        q, k, v = F.linear(x, self.in_proj_weight, self.in_proj_bias).chunk(3, dim=-1)
        q = q.contiguous().view(L, N * self.num_heads, -1).transpose(0, 1)
        k = k.contiguous().view(L, N * self.num_heads, -1).transpose(0, 1)
        v = v.contiguous().view(L, N * self.num_heads, -1).transpose(0, 1)

        if self.logit_scale is not None:
            attn = torch.bmm(F.normalize(q, dim=-1), F.normalize(k, dim=-1).transpose(-1, -2))
            logit_scale = torch.clamp(self.logit_scale, max=self.logit_scale_max).exp()
            attn = attn.view(N, self.num_heads, L, L) * logit_scale
            attn = attn.view(-1, L, L)
        else:
            q = q * self.scale
            attn = torch.bmm(q, k.transpose(-1, -2))

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                new_attn_mask = torch.zeros_like(attn_mask, dtype=q.dtype)
                new_attn_mask.masked_fill_(attn_mask, float("-inf"))
                attn_mask = new_attn_mask
            attn += attn_mask

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = torch.bmm(attn, v)
        if self.head_scale is not None:
            x = x.view(N, self.num_heads, L, C) * self.head_scale
            x = x.view(-1, L, C)
        x = x.transpose(0, 1).reshape(L, N, C)
        x = self.out_proj(x)
        x = self.out_drop(x)
        return x
class ResidualAttentionBlock_expert(nn.Module):
    def __init__(
            self,
            d_model: int,
            n_head: int,
            mlp_ratio: float = 4.0,
            act_layer: Callable = nn.GELU,
            scale_cosine_attn: bool = False,
            scale_heads: bool = False,
            scale_attn: bool = False,
            scale_fc: bool = False,
    ):
        super().__init__()
        self.d_model=d_model
        self.ln_1 = LayerNorm(d_model)
        # FIXME torchscript issues need to be resolved for custom attention
        # if scale_cosine_attn or scale_heads:
        #     self.attn = Attention(
        #        d_model, n_head,
        #        scaled_cosine=scale_cosine_attn,
        #        scale_heads=scale_heads,
        #     )
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_attn = LayerNorm(d_model) if scale_attn else nn.Identity()

        self.ln_2 = LayerNorm(d_model)
        mlp_width = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, mlp_width)),
            ('ln', LayerNorm(mlp_width) if scale_fc else nn.Identity()),
            ("gelu", act_layer()),
            ("c_proj", nn.Linear(mlp_width, d_model))
        ]))
        self.layers = nn.ModuleList([
            nn.Sequential(OrderedDict([
                ("c_fc", nn.Linear(d_model, mlp_width)),
                ("gelu", act_layer()),
                ("c_proj", nn.Linear(mlp_width, d_model))
            ]))
            for _ in range(36)
        ])
    def attention(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        return self.attn(x, x, x, need_weights=False, attn_mask=attn_mask)[0]
        # FIXME torchscript issues need resolving for custom attention option to work
        # if self.use_torch_attn:
        #     return self.attn(x, x, x, need_weights=False, attn_mask=attn_mask)[0]
        # else:
        #     return self.attn(x, attn_mask=attn_mask)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        x = x + self.ln_attn(self.attention(self.ln_1(x), attn_mask=attn_mask))#([516, 516]) torch.float32
        #x = x + self.mlp(self.ln_2(x))
        tem=self.ln_2(x)
        before=self.mlp(tem[0:257])#形状为257 ，bs,512
        last = torch.empty([36, x.shape[1], self.d_model], dtype=x.dtype, device=x.device)  # 预分配空间
        for i in range(36):
           last[i] = self.layers[i](tem[i+257])
        # 合并结果 index 293 is out of bounds for dimension 0 with size 293
        #[36, 12, 768])
        x =x+ torch.cat([before, last], dim=0)
        return x
class ResidualAttentionBlock(nn.Module):
    def __init__(
            self,
            d_model: int,
            n_head: int,
            mlp_ratio: float = 4.0,
            act_layer: Callable = nn.GELU,
            scale_cosine_attn: bool = False,
            scale_heads: bool = False,
            scale_attn: bool = False,
            scale_fc: bool = False,
    ):
        super().__init__()

        self.ln_1 = LayerNorm(d_model)
        # FIXME torchscript issues need to be resolved for custom attention
        # if scale_cosine_attn or scale_heads:
        #     self.attn = Attention(
        #        d_model, n_head,
        #        scaled_cosine=scale_cosine_attn,
        #        scale_heads=scale_heads,
        #     )
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_attn = LayerNorm(d_model) if scale_attn else nn.Identity()

        self.ln_2 = LayerNorm(d_model)
        mlp_width = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, mlp_width)),
            ('ln', LayerNorm(mlp_width) if scale_fc else nn.Identity()),
            ("gelu", act_layer()),
            ("c_proj", nn.Linear(mlp_width, d_model))
        ]))

    def attention(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        return self.attn(x, x, x, need_weights=False, attn_mask=attn_mask)[0]
        # FIXME torchscript issues need resolving for custom attention option to work
        # if self.use_torch_attn:
        #     return self.attn(x, x, x, need_weights=False, attn_mask=attn_mask)[0]
        # else:
        #     return self.attn(x, attn_mask=attn_mask)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        x = x + self.ln_attn(self.attention(self.ln_1(x), attn_mask=attn_mask))#([516, 516]) torch.float32
        x = x + self.mlp(self.ln_2(x))
        return x
class Styled_New_18Qvae_Block_stled(nn.Module):
    def __init__(
            self,
            d_model: int,
            n_head: int,
            mlp_ratio: float = 4.0,
            act_layer: Callable = nn.GELU,
            scale_cosine_attn: bool = False,
            scale_heads: bool = False,
            norm_elementwise_affine: bool = True,
            norm_eps: float = 1e-6,
            scale_attn: bool = False,
            scale_fc: bool = False,
    ):
        super().__init__()
       #####注意为了节省模型大小，这里的一个块送入俩个条件，所以模型块也奇奇怪怪的，有条件可以选择标准的结构
        scale = d_model ** -0.5
        self.attn1 = nn.MultiheadAttention(d_model, n_head)#
        mlp_width = int(d_model * mlp_ratio)
        self.condition_linear = nn.Linear(d_model, d_model)
        self.silu = nn.SiLU()
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, mlp_width)),
            ('ln', LayerNorm(mlp_width) if scale_fc else nn.Identity()),
            ("gelu", act_layer()),
            ("c_proj", nn.Linear(mlp_width, d_model))
        ]))
        self.ln_1=FP32LayerNorm(d_model, norm_eps, norm_elementwise_affine)#参与norm
        self.ln_2=FP32LayerNorm(d_model, norm_eps, norm_elementwise_affine)
        self.ln_11=FP32LayerNorm(d_model, norm_eps, norm_elementwise_affine)#参与交互
        self.ln_22=FP32LayerNorm(d_model, norm_eps, norm_elementwise_affine)
        self.norm2=AdaLayerNorm_stled(d_model, elementwise_affine=norm_elementwise_affine, eps=norm_eps)
        self.norm1 = AdaLayerNorm_stled(d_model, elementwise_affine=norm_elementwise_affine, eps=norm_eps)#进行自适应归一化
    def attention(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        return self.attn(x, x, x, need_weights=False, attn_mask=attn_mask)[0]
       
    def forward(self, x: torch.Tensor,condation: torch.Tensor, index,attn_mask: Optional[torch.Tensor] = None):
        #condation的形状为2,bs，d_models,x形状为256，bs,d_models
        norm_hidden_states = self.norm1(x,self.ln_1(condation[0]))
        x=torch.cat([torch.zeros(1, *x.shape[1:], device=x.device, dtype=x.dtype),x], dim=0)##条件先norm，然后参与交互
        norm_hidden_states=torch.cat([self.ln_11(condation[0:1]),norm_hidden_states], dim=0)#条件先norm，然后参与交互
        x = x + self.attn1(norm_hidden_states, norm_hidden_states, norm_hidden_states, need_weights=False, attn_mask=attn_mask)[0]#([516, 516]) torch.float32
        x = x + self.mlp(self.norm2(x,self.ln_2(condation[1])))
        x=torch.cat([self.ln_22(condation[1:2]),x], dim=0)#
        return x
    

class vit_32768_decoder(nn.Module):
    def __init__(self, width: int, layers: int=36, heads: int=12,  mlp_ratio: float = 4.0,sd3: int=0, act_layer: Callable = nn.GELU):
        super().__init__()
        self.width = width
        self.layers = layers
        scale = width ** -0.5
        self.heads=width//64
        self.linear=nn.Linear(width,768)             
        self.positional_embedding = nn.Parameter(scale * torch.randn(256, width))
        self.resblocks = nn.ModuleList([
            Styled_New_18Qvae_Block_stled(width,self.heads, mlp_ratio, act_layer=act_layer)
            for _ in range(18)
        ])
        self.final_block=nn.ModuleList([
            ResidualAttentionBlock(width, self.heads, mlp_ratio, act_layer=act_layer)
            for _ in range(6)#加了6层
        ])#
        self.lin_post = LayerNorm(width)
        self.pre_post = LayerNorm(width)
        self.cov = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=17, stride=1, padding=8)  # padding为8
        self.ResnetBlock= ResnetBlock(32,32,num_groups=8)
        self.norm_out = nn.GroupNorm(num_groups=8, num_channels=32, eps=1e-6, affine=True)
        self.norm_in = nn.GroupNorm(num_groups=8, num_channels=32, eps=1e-6, affine=True)
        self.conv_out = Conv2dSame(32, 3, kernel_size=3)
    def forward(self, conditon: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        x=self.positional_embedding.unsqueeze(1).repeat(1, conditon.shape[1], 1).to(conditon.dtype)
        x=self.pre_post(x)
       #e([256, 12, 768])
        for index, r in enumerate(self.resblocks):
            x = checkpoint(r, x,conditon[index*2:(index+1)*2], index)
        for index, r in enumerate(self.final_block):
            x = checkpoint(r, x, attn_mask)####256 bs 768
        bs=x.shape[1]
        x=self.linear(self.lin_post(x[-256:,:,:])).reshape(256, bs, 3, 16, 16)   #256 bs  768 patch内像素级别的归位
        x=x.permute(1, 2, 0, 3, 4).reshape(bs, 3, 16, 16, 16, 16).permute(0, 1, 2, 4, 3, 5).reshape(bs, 3, 256, 256)
        #patch级别的归位 bs 3 256 16 16 
        x= self.norm_in(self.cov(x))#一个卷积层
        x=self.ResnetBlock(x)#里面有2个卷积层
        x = self.norm_out(x)####
        x = F.silu(x)
        x = self.conv_out(x)#一个卷积层
        return x

class VQVAE_Transformer_vit_sd3_hug_4096(nn.Module):
    def __init__(self, width: int, layers: int, heads: int=16,  mlp_ratio: float = 4.0, act_layer: Callable = nn.GELU,codebook:int=4096,decoder_dim:int=768,emb_dim:int=128):
        super().__init__()
        self.width = width
        self.layers = layers
        self.grad_checkpointing = False
        scale = width ** -0.5
        self.emb_dim=emb_dim   #设置信息瓶颈，使编码器和解码器不能顺利沟通，第二阶段也可以在这边量化时候。求近似值，可能也会方便些？误差会小
        self.positional_embedding = nn.Parameter(scale * torch.randn(257, width))
        self.query_embedding = nn.Parameter(scale * torch.randn(36, width))
        self.ln_ecoder_1 = LayerNorm(width)
        self.ln_before_project_ = LayerNorm(width)
        self.quantize = VectorQuantizer(194560,emb_dim)
        self.class_embedding = nn.Parameter(scale * torch.randn(1, self.width))
        self.resblocks = nn.ModuleList([
            ResidualAttentionBlock(width, heads, mlp_ratio, act_layer=act_layer)
            for _ in range(18)#加了12层
        ])
        self.resblocks_exper = nn.ModuleList([
            ResidualAttentionBlock_expert(width, heads, mlp_ratio, act_layer=act_layer)
            for _ in range(6)#加了12层
        ])
        self.decoder_dim=decoder_dim
        self.progject_mean= nn.Linear(width, self.emb_dim)##设置信息瓶颈,降低维度
        #self.progject_std= nn.Linear(width, self.emb_dim)
        self.afer_progject= nn.Linear(self.emb_dim, self.decoder_dim)##升维度，也可以作为连续值用作理解任务，经过升维后才送入解码器
        self.decoder=vit_32768_decoder(width=self.decoder_dim,sd3=1)
        self.V2 = nn.Parameter(scale * torch.randn(8192,emb_dim))######V2版本独有的
        self.ln_cosin = LayerNorm(emb_dim)
        self.patch_embed = nn.Conv2d(
            in_channels=3, out_channels=self.width,
              kernel_size=16, stride=16, bias=True)
    def encoder_forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        x = self.patch_embed(x)#ze([12, 768, 16, 16])
        x = x.reshape(x.shape[0], x.shape[1], -1)# ########[8, 32, 16, 16] #bs   widtn,16 16  后面的是序列12, 768, 256])
        x = x.permute(2, 0, 1)#256, 12, 768])
        x = torch.cat([self.class_embedding.unsqueeze(1).repeat(1, x.shape[1], 1).to(x.dtype), x], dim=0)#257, 12, 768])
        positional_embedding_expanded = self.positional_embedding.unsqueeze(1).repeat(1, x.shape[1], 1)
        x = x + positional_embedding_expanded.to(x.dtype)#Size([256, 4, 96])#加上位置编码
        x = torch.cat(
            [x,self.query_embedding.unsqueeze(1).repeat(1, x.shape[1], 1).to(x.dtype) 
             ], dim=0)   #[293, 12, 768])
        x=self.ln_ecoder_1(x)
        for r in self.resblocks:
            x = checkpoint(r, x, attn_mask)####### self.attn_mask
        for r in self.resblocks_exper:
            x = checkpoint(r, x, attn_mask)####### self.attn_mask
        x=self.ln_before_project_ (x)  #之前忘记加了
        origin_dtype = x.dtype      
        mu=self.progject_mean(x[257:]) #这边才是要量化的！！量化使用余弦距离
        ###############V2,这边是为了减轻量化损失影响的！
        mu_flattened = mu.view(-1, 128)
        similarity = cosine_similarity(mu_flattened.float().unsqueeze(1), self.V2.float(), dim=2)
        similarity= similarity/torch.sum(similarity, dim=-1, keepdim=True)#线性加权，未使用softmax暂时没想好温控策略,这边也可以减少误差的影响,但是操作不当会增加误差
        weighted_sum = self.ln_cosin(torch.matmul(similarity, self.V2))
        output = weighted_sum.view(mu.shape)
        ###############这边是为了减轻量化影响的, 保证含义是连续的,这样存在误差也无妨,本来就是总结图像，不指望他还原,目前追求还原质量是担心链路太长,每个环节都差一些,不好找原因
        return     output.to(origin_dtype)
    def decoder_forward(self, condtion: torch.Tensor):
        with torch.cuda.amp.autocast(True):
            x=self.decoder(condtion)
        #  这里面的形状应该为256 bs  width
        return x.float()
    def forward(self,img,quantize=False):
        with torch.cuda.amp.autocast(True):
            #with torch.no_grad():
                feature=self.encoder_forward(img)
        if quantize:
            feature,_=self.quantize(feature.float())
        with torch.cuda.amp.autocast(True):
            after_project = self.afer_progject(feature)#   36 bs  128        下面的需要修改
        return self.decoder_forward(after_project),0#sd是0.000001 #轻微的控制下方差
    # def forward(self,img):
    #     #抽取特征时候用的
    #     return self.encoder_forward(img)#sd是0.000001 #轻微的控制下方差
    def decoder_infer(self, condtion: torch.Tensor):
        x=self.decoder(condtion)
        #  这里面的形状应该为256 bs  width
        return x.float()
    def encoder_infer(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        x = self.patch_embed(x)#ze([12, 768, 16, 16])
        x = x.reshape(x.shape[0], x.shape[1], -1)# ########[8, 32, 16, 16] #bs   widtn,16 16  后面的是序列12, 768, 256])
        x = x.permute(2, 0, 1)#256, 12, 768])
        x = torch.cat([self.class_embedding.unsqueeze(1).repeat(1, x.shape[1], 1).to(x.dtype), x], dim=0)#257, 12, 768])
        positional_embedding_expanded = self.positional_embedding.unsqueeze(1).repeat(1, x.shape[1], 1)
        x = x + positional_embedding_expanded.to(x.dtype)#Size([256, 4, 96])#加上位置编码
        x = torch.cat(
            [x,self.query_embedding.unsqueeze(1).repeat(1, x.shape[1], 1).to(x.dtype) 
             ], dim=0)   #[293, 12, 768])
        x=self.ln_ecoder_1(x)
        for r in self.resblocks:
            x = checkpoint(r, x, attn_mask)####### self.attn_mask
        for r in self.resblocks_exper:
            x = checkpoint(r, x, attn_mask)####### self.attn_mask
        x=self.ln_before_project_ (x)  #之前忘记加了
        origin_dtype = x.dtype       
        mu=self.progject_mean(x[257:]) #这边才是要量化的！！量化使用余弦距离
        ###############V2,这边是为了减轻量化损失影响的！
        mu_flattened = mu.view(-1, 128)
        similarity = cosine_similarity(mu_flattened.float().unsqueeze(1), self.V2.float(), dim=2)
        similarity= similarity/torch.sum(similarity, dim=-1, keepdim=True)#线性加权，未使用softmax暂时没想好温控策略,这边也可以减少误差的影响,但是操作不当会增加
        weighted_sum = self.ln_cosin(torch.matmul(similarity, self.V2))
        output = weighted_sum.view(mu.shape)
        return output
    def infer(self,img,quantize=False):
        ######这一版是连续的
        feature=self.encoder_infer(img)#.permute(1, 0, 2)
        if quantize:
            feature=self.quantize(feature.float())
        
        after_project = self.afer_progject(feature)#.permute(1, 0, 2)   36 bs  128        下面的需要修改
        return self.decoder_infer(after_project)
