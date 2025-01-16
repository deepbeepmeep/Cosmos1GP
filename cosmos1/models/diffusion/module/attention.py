# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List, Optional

import numpy as np
import torch
from mmgp import offload

use_TE = offload.shared_state.get("TE", False)
if use_TE:
    import transformer_engine as te
from einops import rearrange, repeat
from torch import nn, einsum
from torch.utils.checkpoint import checkpoint

if use_TE:
    from transformer_engine.pytorch.attention import DotProductAttention, apply_rotary_pos_emb
else:
    try:
        from sageattention import sageattn
    except:
        sageattn = None
        pass

    try:
        from flash_attn import flash_attn_func
    except:
        pass

    BROKEN_XFORMERS = False
    try:
        import xformers
        import xformers.ops
        x_vers = xformers.__version__
        # XFormers bug confirmed on all versions from 0.0.21 to 0.0.26 (q with bs bigger than 65535 gives CUDA error)
        BROKEN_XFORMERS = x_vers.startswith("0.0.2") and not x_vers.startswith("0.0.20")
    except:
        pass


####################### Thanks to ComfyUI for the transformer_engine replacement code ###############################
#  https://github.com/comfyanonymous/ComfyUI/

    try:
        rms_norm_torch = torch.nn.functional.rms_norm
    except:
        rms_norm_torch = None

    def cast_to(weight, dtype=None, device=None, non_blocking=False, copy=False):
        if device is None or weight.device == device:
            if not copy:
                if dtype is None or weight.dtype == dtype:
                    return weight
            return weight.to(dtype=dtype, copy=copy)

        r = torch.empty_like(weight, dtype=dtype, device=device)
        r.copy_(weight, non_blocking=non_blocking)
        return r

    def cast_to_input(weight, input, non_blocking=False, copy=True):
        return cast_to(weight, input.dtype, input.device, non_blocking=non_blocking, copy=copy)

    def rms_norm(x, weight=None, eps=1e-6):
        if rms_norm_torch is not None and not (torch.jit.is_tracing() or torch.jit.is_scripting()):
            if weight is None:
                return rms_norm_torch(x, (x.shape[-1],), eps=eps)
            else:
                return rms_norm_torch(x, weight.shape, weight=cast_to(weight, dtype=x.dtype, device=x.device), eps=eps)
        else:
            r = x * torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + eps)
            if weight is None:
                return r
            else:
                return r * cast_to(weight, dtype=x.dtype, device=x.device)
            
    class RMSNorm(torch.nn.Module):
        def __init__(
            self, dim: int, elementwise_affine: bool = False, eps: float = 1e-6, device=None, dtype=None
        ):
            """
            Initialize the RMSNorm normalization layer.
            Args:
                dim (int): The dimension of the input tensor.
                eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.
            Attributes:
                eps (float): A small value added to the denominator for numerical stability.
                weight (nn.Parameter): Learnable scaling parameter.
            """
            super().__init__()
            self.eps = eps
            self.learnable_scale = elementwise_affine
            if self.learnable_scale:
                self.weight = nn.Parameter(torch.empty(dim, device=device, dtype=dtype))
            else:
                self.register_parameter("weight", None)

        def forward(self, x):
            return rms_norm(x, self.weight, self.eps)

    def apply_rotary_pos_emb(
        t: torch.Tensor,
        freqs: torch.Tensor,
    ) -> torch.Tensor:
        #   lambda t: rearrange(t, "s b (n c) -> b n s c", n=self.heads, c=self.dim_head),
        
        freqs = freqs
        t_ = t.reshape(*t.shape[:-1], 2, -1).movedim(-2, -1).unsqueeze(-2)
        t_out = freqs[..., 0] * t_[..., 0] + freqs[..., 1] * t_[..., 1]
        t_out = t_out.movedim(-1, -2)
        t_out = t_out.reshape(*t.shape).type_as(t)
        return t_out

    # def get_attn_precision(attn_precision):
    #     if args.dont_upcast_attention:
    #         return None
    #     if FORCE_UPCAST_ATTENTION_DTYPE is not None:
    #         return FORCE_UPCAST_ATTENTION_DTYPE
    #     return attn_precision

    def exists(val):
        return val is not None

    def attention_sage(q, k, v, heads, mask=None, attn_precision=None, skip_reshape=False, skip_output_reshape=False):
        if skip_reshape:
            b, _, _, dim_head = q.shape
            tensor_layout="HND"
        else:
            b, _, dim_head = q.shape
            dim_head //= heads
            q, k, v = map(
                lambda t: t.view(b, -1, heads, dim_head),
                (q, k, v),
            )
            tensor_layout="NHD"

        if mask is not None:
            # add a batch dimension if there isn't already one
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)
            # add a heads dimension if there isn't already one
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)

        out = sageattn(q, k, v, attn_mask=mask, is_causal=False, tensor_layout=tensor_layout)


        if tensor_layout == "HND":
            if not skip_output_reshape:
                out = (
                    out.transpose(1, 2).reshape(b, -1, heads * dim_head)
                )
        else:
            if skip_output_reshape:
                out = out.transpose(1, 2)
            else:
                out = out.reshape(b, -1, heads * dim_head)
        return out


    def attention_basic(q, k, v, heads, mask=None, attn_precision=None, skip_reshape=False, skip_output_reshape=False):
        attn_precision = None # torch.float32 # get_attn_precision(attn_precision)

        if skip_reshape:
            b, _, _, dim_head = q.shape
        else:
            b, _, dim_head = q.shape
            dim_head //= heads

        scale = dim_head ** -0.5

        h = heads
        if skip_reshape:
            q, k, v = map(
                lambda t: t.reshape(b * heads, -1, dim_head),
                (q, k, v),
            )
        else:
            q, k, v = map(
                lambda t: t.unsqueeze(3)
                .reshape(b, -1, heads, dim_head)
                .permute(0, 2, 1, 3)
                .reshape(b * heads, -1, dim_head)
                .contiguous(),
                (q, k, v),
            )

        # force cast to fp32 to avoid overflowing
        if attn_precision == torch.float32:
            sim = einsum('b i d, b j d -> b i j', q.float(), k.float()) * scale
        else:
            sim = einsum('b i d, b j d -> b i j', q, k) * scale

        del q, k

        if exists(mask):
            if mask.dtype == torch.bool:
                mask = rearrange(mask, 'b ... -> b (...)') #TODO: check if this bool part matches pytorch attention
                max_neg_value = -torch.finfo(sim.dtype).max
                mask = repeat(mask, 'b j -> (b h) () j', h=h)
                sim.masked_fill_(~mask, max_neg_value)
            else:
                if len(mask.shape) == 2:
                    bs = 1
                else:
                    bs = mask.shape[0]
                mask = mask.reshape(bs, -1, mask.shape[-2], mask.shape[-1]).expand(b, heads, -1, -1).reshape(-1, mask.shape[-2], mask.shape[-1])
                sim.add_(mask)

        # attention, what we cannot get enough of
        sim = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', sim.to(v.dtype), v)

        if skip_output_reshape:
            out = (
                out.unsqueeze(0)
                .reshape(b, heads, -1, dim_head)
            )
        else:
            out = (
                out.unsqueeze(0)
                .reshape(b, heads, -1, dim_head)
                .permute(0, 2, 1, 3)
                .reshape(b, -1, heads * dim_head)
            )
        return out
    

    def attention_xformers(q, k, v, heads, mask=None, attn_precision=None, skip_reshape=False, skip_output_reshape=False):
        b = q.shape[0]
        dim_head = q.shape[-1]
        # check to make sure xformers isn't broken
        disabled_xformers = False

        if BROKEN_XFORMERS:
            if b * heads > 65535:
                disabled_xformers = True

        if not disabled_xformers:
            if torch.jit.is_tracing() or torch.jit.is_scripting():
                disabled_xformers = True

        if disabled_xformers:
            return attention_basic(q, k, v, heads, mask, skip_reshape=skip_reshape)

        if skip_reshape:
            # b h k d -> b k h d
            q, k, v = map(
                lambda t: t.permute(0, 2, 1, 3),
                (q, k, v),
            )
        # actually do the reshaping
        else:
            dim_head //= heads
            q, k, v = map(
                lambda t: t.reshape(b, -1, heads, dim_head),
                (q, k, v),
            )

        if mask is not None:
            # add a singleton batch dimension
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)
            # add a singleton heads dimension
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)
            # pad to a multiple of 8
            pad = 8 - mask.shape[-1] % 8
            # the xformers docs says that it's allowed to have a mask of shape (1, Nq, Nk)
            # but when using separated heads, the shape has to be (B, H, Nq, Nk)
            # in flux, this matrix ends up being over 1GB
            # here, we create a mask with the same batch/head size as the input mask (potentially singleton or full)
            mask_out = torch.empty([mask.shape[0], mask.shape[1], q.shape[1], mask.shape[-1] + pad], dtype=q.dtype, device=q.device)

            mask_out[..., :mask.shape[-1]] = mask
            # doesn't this remove the padding again??
            mask = mask_out[..., :mask.shape[-1]]
            mask = mask.expand(b, heads, -1, -1)

        out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=mask)

        if skip_output_reshape:
            out = out.permute(0, 2, 1, 3)
        else:
            out = (
                out.reshape(b, -1, heads * dim_head)
            )

        return out

    # if model_management.is_nvidia(): #pytorch 2.3 and up seem to have this issue.
        SDP_BATCH_LIMIT = 2**15
    # else:
    #     #TODO: other GPUs ?
    #     SDP_BATCH_LIMIT = 2**31


# ---------------------- Feed Forward Network -----------------------


class FeedForward(nn.Module):
    """
    Transformer FFN with optional gating

    Parameters:
        d_model (int): Dimensionality of input features.
        d_ff (int): Dimensionality of the hidden layer.
        dropout (float, optional): Dropout rate applied after the activation function. Defaults to 0.1.
        activation (callable, optional): The activation function applied after the first linear layer.
                                         Defaults to nn.ReLU().
        is_gated (bool, optional): If set to True, incorporates gating mechanism to the feed-forward layer.
                                   Defaults to False.
        bias (bool, optional): If set to True, adds a bias to the linear layers. Defaults to True.

    Example:
        >>> ff = FeedForward(d_model=512, d_ff=2048)
        >>> x = torch.randn(64, 10, 512)  # Example input tensor
        >>> output = ff(x)
        >>> print(output.shape)  # Expected shape: (64, 10, 512)
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1,
        activation=nn.ReLU(),
        is_gated: bool = False,
        bias: bool = False,
    ) -> None:
        super().__init__()

        self.layer1 = nn.Linear(d_model, d_ff, bias=bias)
        self.layer2 = nn.Linear(d_ff, d_model, bias=bias)

        self.dropout = nn.Dropout(dropout)
        self.activation = activation
        self.is_gated = is_gated
        if is_gated:
            self.linear_gate = nn.Linear(d_model, d_ff, bias=False)

    def forward(self, x: torch.Tensor):
        g = self.activation(self.layer1(x))
        if self.is_gated:
            x = g * self.linear_gate(x)
        else:
            x = g
        assert self.dropout.p == 0.0, "we skip dropout"
        return self.layer2(x)


class GPT2FeedForward(FeedForward):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1, bias: bool = False):
        super().__init__(
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout,
            activation=nn.GELU(),
            is_gated=False,
            bias=bias,
        )

    def forward(self, x: torch.Tensor):
        assert self.dropout.p == 0.0, "we skip dropout"

        x = self.layer1(x)

        def activation_layer2_forward(x):
            x = self.activation(x)
            x = self.layer2(x)
            return x

        x = checkpoint(activation_layer2_forward, x, use_reentrant=False)
        return x


# ---------------------- Normalization Layer -----------------------


def normalize(x: torch.Tensor, dim: Optional[List[int]] = None, eps: float = 0) -> torch.Tensor:
    """
    Normalizes the input tensor along specified dimensions such that the average square norm of elements is adjusted.

    Args:
        x (torch.Tensor): The input tensor to normalize.
        dim (list, optional): The dimensions over which to normalize. If None, normalizes over all dimensions except the first.
        eps (float, optional): A small constant to ensure numerical stability during division.

    Returns:
        torch.Tensor: The normalized tensor.
    """
    if dim is None:
        dim = list(range(1, x.ndim))
    norm = torch.linalg.vector_norm(x, dim=dim, keepdim=True, dtype=torch.float32)
    norm = torch.add(eps, norm, alpha=np.sqrt(norm.numel() / x.numel()))
    return x / norm.to(x.dtype)


def get_normalization(name: str, channels: int, kwargs = {}):
    if name == "I":
        return nn.Identity()
    elif name == "R":
        if use_TE:
            return te.pytorch.RMSNorm(channels, eps=1e-6)
        else:
            return RMSNorm(channels, elementwise_affine = True, eps=1e-6, **kwargs)

    else:
        raise ValueError(f"Normalization {name} not found")


class BaseAttentionOp(nn.Module):
    def __init__(self):
        super().__init__()


class Attention(nn.Module):
    """
    Generalized attention impl.

    Allowing for both self-attention and cross-attention configurations depending on whether a `context_dim` is provided.
    If `context_dim` is None, self-attention is assumed.

    Parameters:
        query_dim (int): Dimension of each query vector.
        context_dim (int, optional): Dimension of each context vector. If None, self-attention is assumed.
        heads (int, optional): Number of attention heads. Defaults to 8.
        dim_head (int, optional): Dimension of each head. Defaults to 64.
        dropout (float, optional): Dropout rate applied to the output of the attention block. Defaults to 0.0.
        attn_op (BaseAttentionOp, optional): Custom attention operation to be used instead of the default.
        qkv_bias (bool, optional): If True, adds a learnable bias to query, key, and value projections. Defaults to False.
        out_bias (bool, optional): If True, adds a learnable bias to the output projection. Defaults to False.
        qkv_norm (str, optional): A string representing normalization strategies for query, key, and value projections.
                                  Defaults to "SSI".
        qkv_norm_mode (str, optional): A string representing normalization mode for query, key, and value projections.
                                        Defaults to 'per_head'. Only support 'per_head'.

    Examples:
        >>> attn = Attention(query_dim=128, context_dim=256, heads=4, dim_head=32, dropout=0.1)
        >>> query = torch.randn(10, 128)  # Batch size of 10
        >>> context = torch.randn(10, 256)  # Batch size of 10
        >>> output = attn(query, context)  # Perform the attention operation

    Note:
        https://github.com/MatthieuTPHR/diffusers/blob/d80b531ff8060ec1ea982b65a1b8df70f73aa67c/src/diffusers/models/attention.py#L223
    """

    def __init__(
        self,
        query_dim: int,
        context_dim=None,
        heads=8,
        dim_head=64,
        dropout=0.0,
        attn_op: Optional[BaseAttentionOp] = None,
        qkv_bias: bool = False,
        out_bias: bool = False,
        qkv_norm: str = "SSI",
        qkv_norm_mode: str = "per_head",
        backend: str = "transformer_engine",
        qkv_format: str = "bshd",
    ) -> None:
        super().__init__()

        self.is_selfattn = context_dim is None  # self attention

        inner_dim = dim_head * heads
        context_dim = query_dim if context_dim is None else context_dim

        self.heads = heads
        self.dim_head = dim_head
        self.qkv_norm_mode = qkv_norm_mode
        self.qkv_format = qkv_format

        if self.qkv_norm_mode == "per_head":
            norm_dim = dim_head
        else:
            raise ValueError(f"Normalization mode {self.qkv_norm_mode} not found, only support 'per_head'")

        self.backend = backend

        self.to_q = nn.Sequential(
            nn.Linear(query_dim, inner_dim, bias=qkv_bias),
            get_normalization(qkv_norm[0], norm_dim),
        )
        self.to_k = nn.Sequential(
            nn.Linear(context_dim, inner_dim, bias=qkv_bias),
            get_normalization(qkv_norm[1], norm_dim),
        )
        self.to_v = nn.Sequential(
            nn.Linear(context_dim, inner_dim, bias=qkv_bias),
            get_normalization(qkv_norm[2], norm_dim),
        )

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim, bias=out_bias),
            nn.Dropout(dropout),
        )
        if use_TE:
            if attn_op:  # use what is given
                self.attn_op = attn_op
            elif self.backend == "transformer_engine":
                sequence_parallel = False
                self.attn_op: BaseAttentionOp = DotProductAttention(
                    self.heads,
                    self.dim_head,
                    num_gqa_groups=self.heads,
                    attention_dropout=0,
                    qkv_format=qkv_format,
                    attn_mask_type="no_mask",
                    tp_size=1,
                    tp_group=None,
                    sequence_parallel=sequence_parallel,
                )
            else:
                raise ValueError(f"Backend {backend} not found")

    def cal_qkv(
        self, x, context=None, mask=None, rope_emb=None, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        del kwargs

        """
        self.to_q, self.to_k, self.to_v are nn.Sequential with projection + normalization layers.
        Before 07/24/2024, these modules normalize across all heads.
        After 07/24/2024, to support tensor parallelism and follow the common practice in the community,
        we support to normalize per head.
        To keep the checkpoint copatibility with the previous code,
        we keep the nn.Sequential but call the projection and the normalization layers separately.
        We use a flag `self.qkv_norm_mode` to control the normalization behavior.
        The default value of `self.qkv_norm_mode` is "per_head", which means we normalize per head.
        """
        if self.qkv_norm_mode == "per_head":
            q = self.to_q[0](x)
            context = x if context is None else context
            k = self.to_k[0](context)
            v = self.to_v[0](context)
            if use_TE:
                q, k, v = map(
                    lambda t: rearrange(t, "b ... (n c) -> b ... n c", n=self.heads, c=self.dim_head),
                    (q, k, v),
                )

            else:
                q, k, v = map(
                    lambda t: rearrange(t, "s b (n c) -> b n s c", n=self.heads, c=self.dim_head),
                    (q, k, v),
                )
        else:
            raise ValueError(f"Normalization mode {self.qkv_norm_mode} not found, only support 'per_head'")

        q = self.to_q[1](q)
        k = self.to_k[1](k)
        v = self.to_v[1](v)
        if self.is_selfattn and rope_emb is not None:  # only apply to self-attention!
            if use_TE:
                q = apply_rotary_pos_emb(q, rope_emb, tensor_format=self.qkv_format, fused=True )
                k = apply_rotary_pos_emb(k, rope_emb, tensor_format=self.qkv_format, fused=True )
            else:
                q = apply_rotary_pos_emb(q, rope_emb ) 
                k = apply_rotary_pos_emb(k, rope_emb ) 
        return q, k, v

    def cal_attn(self, q, k, v, mask=None):
        if use_TE:
            if self.backend == "transformer_engine":
                seq_dim = self.qkv_format.index("s")
                assert (
                    q.shape[seq_dim] > 1 and k.shape[seq_dim] > 1
                ), "Seqlen must be larger than 1 for TE Attention starting with 1.8 TE version."
                out = self.attn_op(q, k, v, core_attention_bias_type="no_bias", core_attention_bias=None)  # [B, Mq, H, V]
                return self.to_out(out)
            elif self.backend == "torch":
                out = self.attn_op(q, k, v, mask=mask)  # [B, Mq, H, V]
                return self.to_out(rearrange(out, " b ... n c -> b ... (n c)"))
            else:
                raise ValueError(f"Backend {self.backend} not found")
        else:

            attention_mode = offload.shared_state.get("attention_mode", "basic")

            if attention_mode == "sage":
                attention_func = attention_basic
            elif attention_mode == "xformers":
                attention_func = attention_xformers
            else:
                attention_func = attention_basic
         
            out = attention_func(q, k, v, self.heads, skip_reshape=True, mask=mask, skip_output_reshape=True)

            out = rearrange(out, " b n s c -> s b (n c)")
        return self.to_out(out)    

    def forward(
        self,
        x,
        context=None,
        mask=None,
        rope_emb=None,
        **kwargs,
    ):
        """
        Args:
            x (Tensor): The query tensor of shape [B, Mq, K]
            context (Optional[Tensor]): The key tensor of shape [B, Mk, K] or use x as context [self attention] if None
        """
        q, k, v = self.cal_qkv(x, context, mask, rope_emb=rope_emb, **kwargs)
        return self.cal_attn(q, k, v, mask)
