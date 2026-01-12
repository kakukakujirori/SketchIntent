import math
import typing

import huggingface_hub
import omegaconf
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# Flags required to enable jit fusion kernels
torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)
torch._C._jit_override_can_fuse_on_cpu(True)
torch._C._jit_override_can_fuse_on_gpu(True)


def bias_dropout_add_scale(
    x: torch.Tensor,
    bias: typing.Optional[torch.Tensor],
    scale: torch.Tensor,
    residual: typing.Optional[torch.Tensor],
    prob: float,
    training: bool) -> torch.Tensor:
  if bias is not None:
    out = scale * F.dropout(x + bias, p=prob, training=training)
  else:
    out = scale * F.dropout(x, p=prob, training=training)

  if residual is not None:
    out = residual + out
  return out


def get_bias_dropout_add_scale(training):
  def _bias_dropout_add(x, bias, scale, residual, prob):
    return bias_dropout_add_scale(
      x, bias, scale, residual, prob, training)

  return _bias_dropout_add


# function overload
def modulate(x: torch.Tensor,
             shift: torch.Tensor,
             scale: torch.Tensor) -> torch.Tensor:
  return x * (1 + scale) + shift


@torch.jit.script
def bias_dropout_add_scale_fused_train(
    x: torch.Tensor,
    bias: typing.Optional[torch.Tensor],
    scale: torch.Tensor,
    residual: typing.Optional[torch.Tensor],
    prob: float) -> torch.Tensor:
  return bias_dropout_add_scale(
    x, bias, scale, residual, prob, True)


@torch.jit.script
def bias_dropout_add_scale_fused_inference(
    x: torch.Tensor,
    bias: typing.Optional[torch.Tensor],
    scale: torch.Tensor,
    residual: typing.Optional[torch.Tensor],
    prob: float) -> torch.Tensor:
  return bias_dropout_add_scale(
    x, bias, scale, residual, prob, False)


@torch.jit.script
def modulate_fused(x: torch.Tensor,
                   shift: torch.Tensor,
                   scale: torch.Tensor) -> torch.Tensor:
  return modulate(x, shift, scale)


class Rotary(torch.nn.Module):
  def __init__(self, dim, base=10_000):
    super().__init__()
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    self.register_buffer('inv_freq', inv_freq)
    self.seq_len_cached = None
    self.cos_cached = None
    self.sin_cached = None

  def forward(self, x, seq_dim=1):
    seq_len = x.shape[seq_dim]
    if seq_len != self.seq_len_cached:
      self.seq_len_cached = seq_len
      t = torch.arange(x.shape[seq_dim], device=x.device).type_as(self.inv_freq)
      freqs = torch.einsum("i,j->ij", t, self.inv_freq.clone())
      emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
      # dims are: batch, seq_len, qkv, head, dim
      self.cos_cached = emb.cos()[None, :, None, None, :].repeat(1,1,3,1,1)
      self.sin_cached = emb.sin()[None, :, None, None, :].repeat(1,1,3,1,1)
      # This makes the transformation on v an identity.
      self.cos_cached[:,:,2,:,:].fill_(1.)
      self.sin_cached[:,:,2,:,:].fill_(0.)

    return self.cos_cached, self.sin_cached


def rotate_half(x):
  x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
  return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(qkv, cos, sin):
  """Apply rotary position embeddings to QKV tensor.

  Args:
    qkv: [B, L, 3, H, D] - packed query, key, value
    cos: [1, L, 3, 1, D] - cosine embeddings from Rotary class
    sin: [1, L, 3, 1, D] - sine embeddings from Rotary class

  Returns:
    qkv with RoPE applied to Q and K (V unchanged): [B, L, 3, H, D]
  """
  # Slice cos/sin to [L, D/2]
  cos = cos[0, :, 0, 0, :cos.shape[-1]//2]
  sin = sin[0, :, 0, 0, :sin.shape[-1]//2]

  # Apply rotary to Q and K only (indices 0 and 1), not V (index 2)
  q = qkv[:, :, 0, :, :]  # [B, L, H, D]
  k = qkv[:, :, 1, :, :]
  v = qkv[:, :, 2, :, :]

  # Expand cos/sin for broadcasting: [L, D/2] -> [1, L, 1, D]
  cos_full = torch.cat([cos, cos], dim=-1)  # [L, D]
  sin_full = torch.cat([sin, sin], dim=-1)  # [L, D]
  cos_full = cos_full[None, :, None, :]  # [1, L, 1, D]
  sin_full = sin_full[None, :, None, :]  # [1, L, 1, D]

  q_rot = q * cos_full + rotate_half(q) * sin_full
  k_rot = k * cos_full + rotate_half(k) * sin_full

  # Stack back to [B, L, 3, H, D]
  return torch.stack([q_rot, k_rot, v], dim=2)


# function overload
def modulate(x, shift, scale):
  return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#################################################################################
#                                  Layers                                       #
#################################################################################
class LayerNorm(nn.Module):
  def __init__(self, dim):
    super().__init__()
    self.weight = nn.Parameter(torch.ones([dim]))
    self.dim = dim
  def forward(self, x):
    with torch.amp.autocast('cuda', enabled=False):
      x = F.layer_norm(x.float(), [self.dim])
    return x * self.weight[None,None,:]


def residual_linear(x, W, x_skip, residual_scale):
  """x_skip + residual_scale * W @ x"""
  dim_out, dim_in = W.shape[0], W.shape[1]
  return torch.addmm(
    x_skip.view(-1, dim_out),
    x.view(-1, dim_in),
    W.T,
    alpha=residual_scale).view(*x.shape[:-1], dim_out)


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################
class TimestepEmbedder(nn.Module):
  """
  Embeds scalar timesteps into vector representations.
  """
  def __init__(self, hidden_size, frequency_embedding_size=256):
    super().__init__()
    self.mlp = nn.Sequential(
      nn.Linear(frequency_embedding_size, hidden_size, bias=True),
      nn.SiLU(),
      nn.Linear(hidden_size, hidden_size, bias=True))
    self.frequency_embedding_size = frequency_embedding_size

  @staticmethod
  def timestep_embedding(t, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    :param t: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an (N, D) Tensor of positional embeddings.
    """
    # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
    half = dim // 2
    freqs = torch.exp(
      - math.log(max_period)
      * torch.arange(start=0, end=half, dtype=torch.float32)
      / half).to(device=t.device)
    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
      embedding = torch.cat(
        [embedding,
         torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

  def forward(self, t):
    t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
    t_emb = self.mlp(t_freq)
    return t_emb


class LabelEmbedder(nn.Module):
  """Embeds class labels into vector representations.

  Also handles label dropout for classifier-free guidance.
  """
  def __init__(self, num_classes, cond_size):
    super().__init__()
    self.embedding_table = nn.Embedding(num_classes + 1, cond_size)
    self.num_classes = num_classes

    # TODO think of initializing with 0.02 std deviation like in original DiT paper

  def forward(self, labels):
    embeddings = self.embedding_table(labels)
    return embeddings


#################################################################################
#                                 Core Model                                    #
#################################################################################


class DDiTPlainBlock(nn.Module):
  """Simple transformer block without conditioning.

  This is DDiTBlock without AdaLN modulation, for embedding primitives.
  """
  def __init__(self, dim, n_heads, mlp_ratio=4, dropout=0.1):
    super().__init__()
    self.n_heads = n_heads
    self.dropout = dropout

    self.norm1 = LayerNorm(dim)
    self.attn_qkv = nn.Linear(dim, 3 * dim, bias=False)
    self.attn_out = nn.Linear(dim, dim, bias=False)

    self.norm2 = LayerNorm(dim)
    self.mlp = nn.Sequential(
      nn.Linear(dim, mlp_ratio * dim, bias=True),
      nn.GELU(approximate='tanh'),
      nn.Linear(mlp_ratio * dim, dim, bias=True))

  def forward(self, x, rotary_cos_sin):
    batch_size, seq_len = x.shape[0], x.shape[1]

    # Self-attention
    x_skip = x
    x = self.norm1(x)

    qkv = self.attn_qkv(x)
    qkv = rearrange(qkv, 'b s (three h d) -> b s three h d', three=3, h=self.n_heads)

    with torch.amp.autocast('cuda', enabled=False):
      cos, sin = rotary_cos_sin
      qkv = apply_rotary_pos_emb(qkv, cos.to(qkv.dtype), sin.to(qkv.dtype))

    # Use SDPA for attention
    q = qkv[:, :, 0, :, :]  # [B, S, H, D]
    k = qkv[:, :, 1, :, :]
    v = qkv[:, :, 2, :, :]

    # Rearrange for SDPA: [B, H, S, D]
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout if self.training else 0.)

    # Rearrange back: [B, S, H*D]
    x = x.transpose(1, 2).reshape(batch_size, seq_len, -1)

    x = self.attn_out(x)
    x = F.dropout(x, p=self.dropout, training=self.training)
    x = x + x_skip

    # MLP
    x_skip = x
    x = self.norm2(x)
    x = self.mlp(x)
    x = F.dropout(x, p=self.dropout, training=self.training)
    x = x + x_skip

    return x


class DDiTBlock(nn.Module):
  def __init__(self, dim, n_heads, cond_dim, mlp_ratio=4, dropout=0.1):
    super().__init__()
    self.n_heads = n_heads

    self.norm1 = LayerNorm(dim)
    self.attn_qkv = nn.Linear(dim, 3 * dim, bias=False)
    self.attn_out = nn.Linear(dim, dim, bias=False)
    self.dropout1 = nn.Dropout(dropout)

    self.norm2 = LayerNorm(dim)
    self.mlp = nn.Sequential(
      nn.Linear(dim, mlp_ratio * dim, bias=True),
      nn.GELU(approximate='tanh'),
      nn.Linear(mlp_ratio * dim, dim, bias=True))
    self.dropout2 = nn.Dropout(dropout)
    self.dropout = dropout

    self.adaLN_modulation = nn.Linear(cond_dim, 6 * dim, bias=True)
    self.adaLN_modulation.weight.data.zero_()
    self.adaLN_modulation.bias.data.zero_()


  def _get_bias_dropout_scale(self):
    if self.training:
      return bias_dropout_add_scale_fused_train
    else:
      return bias_dropout_add_scale_fused_inference


  def forward(self, x, rotary_cos_sin, c, seqlens=None):
    batch_size, seq_len = x.shape[0], x.shape[1]

    bias_dropout_scale_fn = self._get_bias_dropout_scale()

    c_out = self.adaLN_modulation(c)
    split_size = c_out.shape[-1] // 6
    (shift_msa, scale_msa, gate_msa, shift_mlp,
     scale_mlp, gate_mlp) = c_out[:, None].split(split_size, dim=2)

    # attention operation
    x_skip = x
    x = modulate_fused(self.norm1(x), shift_msa, scale_msa)

    qkv = self.attn_qkv(x)
    qkv = rearrange(qkv,
                    'b s (three h d) -> b s three h d',
                    three=3,
                    h=self.n_heads)
    with torch.amp.autocast('cuda', enabled=False):
      cos, sin = rotary_cos_sin
      qkv = apply_rotary_pos_emb(
        qkv, cos.to(qkv.dtype), sin.to(qkv.dtype))

    # Use SDPA for attention (works on both CPU and GPU, auto-selects Flash backend when available)
    q = qkv[:, :, 0, :, :]  # [B, S, H, D]
    k = qkv[:, :, 1, :, :]
    v = qkv[:, :, 2, :, :]

    # Rearrange for SDPA: [B, H, S, D]
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout if self.training else 0.)
    x = x.transpose(1, 2).reshape(batch_size, seq_len, -1)

    x = bias_dropout_scale_fn(self.attn_out(x),
                              None,
                              gate_msa,
                              x_skip,
                              self.dropout)

    # mlp operation
    x = bias_dropout_scale_fn(
      self.mlp(modulate_fused(
        self.norm2(x), shift_mlp, scale_mlp)),
      None, gate_mlp, x, self.dropout)
    return x


class DDiTCrossBlock(nn.Module):
  """DDiT block with self-attention followed by cross-attention.

  Architecture per block:
    1. Self-attention on x (target sequence)
    2. Cross-attention: Q from x, K/V from memory (source sequence)
    3. Feed-forward network
  """
  def __init__(self, dim, n_heads, cond_dim, mlp_ratio=4, dropout=0.1):
    super().__init__()
    self.n_heads = n_heads
    self.head_dim = dim // n_heads
    self.dropout = dropout

    # Self-attention layers
    self.norm1 = LayerNorm(dim)
    self.self_attn_qkv = nn.Linear(dim, 3 * dim, bias=False)
    self.self_attn_out = nn.Linear(dim, dim, bias=False)

    # Cross-attention layers
    self.norm_cross = LayerNorm(dim)
    self.cross_attn_q = nn.Linear(dim, dim, bias=False)
    self.cross_attn_kv = nn.Linear(dim, 2 * dim, bias=False)
    self.cross_attn_out = nn.Linear(dim, dim, bias=False)

    # FFN layers
    self.norm2 = LayerNorm(dim)
    self.mlp = nn.Sequential(
      nn.Linear(dim, mlp_ratio * dim, bias=True),
      nn.GELU(approximate='tanh'),
      nn.Linear(mlp_ratio * dim, dim, bias=True))

    # AdaLN modulation: 9 * dim for (self-attn: shift, scale, gate) + (cross-attn: shift, scale, gate) + (mlp: shift, scale, gate)
    self.adaLN_modulation = nn.Linear(cond_dim, 9 * dim, bias=True)
    self.adaLN_modulation.weight.data.zero_()
    self.adaLN_modulation.bias.data.zero_()

  def _get_bias_dropout_scale(self):
    if self.training:
      return bias_dropout_add_scale_fused_train
    else:
      return bias_dropout_add_scale_fused_inference

  def forward(self, x, memory, rotary_cos_sin, c, seqlens=None):
    """
    Args:
      x: Target sequence [B, L_tgt, D] - constraint embeddings
      memory: Source sequence [B, L_src, D] - primitive embeddings
      rotary_cos_sin: Rotary position embeddings for self-attention
      c: Conditioning vector [B, D]
      seqlens: Optional sequence lengths
    """
    batch_size, tgt_len = x.shape[0], x.shape[1]
    src_len = memory.shape[1]

    bias_dropout_scale_fn = self._get_bias_dropout_scale()

    # AdaLN modulation parameters
    c_out = self.adaLN_modulation(c)
    split_size = c_out.shape[-1] // 9
    (shift_sa, scale_sa, gate_sa,
     shift_ca, scale_ca, gate_ca,
     shift_mlp, scale_mlp, gate_mlp) = c_out[:, None].split(split_size, dim=2)

    # 1. Self-attention on x
    x_skip = x
    x_normed = modulate_fused(self.norm1(x), shift_sa, scale_sa)

    qkv = self.self_attn_qkv(x_normed)
    qkv = rearrange(qkv, 'b s (three h d) -> b s three h d', three=3, h=self.n_heads)
    with torch.amp.autocast('cuda', enabled=False):
      cos, sin = rotary_cos_sin
      qkv = apply_rotary_pos_emb(qkv, cos.to(qkv.dtype), sin.to(qkv.dtype))

    # Use SDPA for attention (works on both CPU and GPU, auto-selects Flash backend when available)
    q = qkv[:, :, 0, :, :]  # [B, S, H, D]
    k = qkv[:, :, 1, :, :]
    v = qkv[:, :, 2, :, :]

    # Rearrange for SDPA: [B, H, S, D]
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout if self.training else 0.)
    x = x.transpose(1, 2).reshape(batch_size, tgt_len, -1)

    x = bias_dropout_scale_fn(self.self_attn_out(x), None, gate_sa, x_skip, self.dropout)

    # 2. Cross-attention: Q from x, K/V from memory
    x_skip = x
    x_normed = modulate_fused(self.norm_cross(x), shift_ca, scale_ca)

    q = self.cross_attn_q(x_normed)
    kv = self.cross_attn_kv(memory)

    q = rearrange(q, 'b s (h d) -> b h s d', h=self.n_heads)
    kv = rearrange(kv, 'b s (two h d) -> b h s two d', two=2, h=self.n_heads)
    k, v = kv[..., 0, :], kv[..., 1, :]

    # Standard attention (no flash for cross-attention with variable lengths to keep simple)
    attn_out = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout if self.training else 0.)
    attn_out = rearrange(attn_out, 'b h s d -> b s (h d)')

    x = bias_dropout_scale_fn(self.cross_attn_out(attn_out), None, gate_ca, x_skip, self.dropout)

    # 3. FFN
    x = bias_dropout_scale_fn(
      self.mlp(modulate_fused(self.norm2(x), shift_mlp, scale_mlp)),
      None, gate_mlp, x, self.dropout)

    return x



class EmbeddingLayer(nn.Module):
  def __init__(self, dim, vocab_dim):
    super().__init__()
    self.embedding = nn.Parameter(torch.empty((vocab_dim, dim)))
    torch.nn.init.kaiming_uniform_(self.embedding, a=math.sqrt(5))

  def forward(self, x):
    return self.embedding[x]


class DDitFinalLayer(nn.Module):
  def __init__(self, hidden_size, out_channels, cond_dim):
    super().__init__()
    self.norm_final = LayerNorm(hidden_size)
    self.linear = nn.Linear(hidden_size, out_channels)
    self.linear.weight.data.zero_()
    self.linear.bias.data.zero_()

    self.adaLN_modulation = nn.Linear(cond_dim,
                                      2 * hidden_size,
                                      bias=True)
    self.adaLN_modulation.weight.data.zero_()
    self.adaLN_modulation.bias.data.zero_()


  def forward(self, x, c):
    shift, scale = self.adaLN_modulation(c)[:, None].chunk(2, dim=2)
    x = modulate_fused(self.norm_final(x), shift, scale)
    x = self.linear(x)
    return x


class DIT(nn.Module, huggingface_hub.PyTorchModelHubMixin):
  def __init__(self, config, vocab_size: int):
    super().__init__()
    if type(config) == dict:
      config = omegaconf.OmegaConf.create(config)

    self.config = config
    self.vocab_size = vocab_size

    self.vocab_embed = EmbeddingLayer(config.model.hidden_size,
                                      vocab_size)
    self.sigma_map = TimestepEmbedder(config.model.cond_dim)
    self.rotary_emb = Rotary(
      config.model.hidden_size // config.model.n_heads)

    blocks = []
    for _ in range(config.model.n_blocks):
      blocks.append(DDiTBlock(config.model.hidden_size,
                              config.model.n_heads,
                              config.model.cond_dim,
                              dropout=config.model.dropout))
    self.blocks = nn.ModuleList(blocks)

    self.output_layer = DDitFinalLayer(
      config.model.hidden_size,
      vocab_size,
      config.model.cond_dim)
    self.scale_by_sigma = config.model.scale_by_sigma

  def _get_bias_dropout_scale(self):
    if self.training:
      return bias_dropout_add_scale_fused_train
    else:
      return  bias_dropout_add_scale_fused_inference

  def get_hidden_states(self, indices, sigma):
    """Get both the final output and hidden state before the final linear layer."""
    x = self.vocab_embed(indices)
    c = F.silu(self.sigma_map(sigma))

    rotary_cos_sin = self.rotary_emb(x)

    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
      for i in range(len(self.blocks)):
        x = self.blocks[i](x, rotary_cos_sin, c, seqlens=None)
      # hidden_state = self.output_layer.norm_final(x)
      final_output = self.output_layer(x, c)

    return final_output, x, c

  def forward(self, indices, sigma):
    x = self.vocab_embed(indices)
    c = F.silu(self.sigma_map(sigma))

    rotary_cos_sin = self.rotary_emb(x)

    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
      for i in range(len(self.blocks)):
        x = self.blocks[i](x, rotary_cos_sin, c, seqlens=None)
      x = self.output_layer(x, c)

    return x

class FeatureDIT(nn.Module, huggingface_hub.PyTorchModelHubMixin):
  '''
  DIT without embedding layer, gets feature inputs directly
  '''
  def __init__(self, config, output_size: int):
    super().__init__()
    if type(config) == dict:
      config = omegaconf.OmegaConf.create(config)

    self.config = config
    self.ouput_size = output_size
    self.rotary_emb = Rotary(
      config.model.hidden_size // config.model.n_heads)

    blocks = []
    for _ in range(config.model.n_blocks):
      blocks.append(DDiTBlock(config.model.hidden_size,
                              config.model.n_heads,
                              config.model.cond_dim,
                              dropout=config.model.dropout))
    self.blocks = nn.ModuleList(blocks)

    self.output_layer = DDitFinalLayer(
      config.model.hidden_size,
      output_size,
      config.model.cond_dim)
    self.scale_by_sigma = config.model.scale_by_sigma

  def _get_bias_dropout_scale(self):
    if self.training:
      return bias_dropout_add_scale_fused_train
    else:
      return  bias_dropout_add_scale_fused_inference

  def forward(self, x, c):
    rotary_cos_sin = self.rotary_emb(x)
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
      for i in range(len(self.blocks)):
        x = self.blocks[i](x, rotary_cos_sin, c, seqlens=None)
      x = self.output_layer(x, c)

    return x


class CrossAttentionFeatureDIT(nn.Module, huggingface_hub.PyTorchModelHubMixin):
  """DIT with cross-attention: target attends to memory (encoder-decoder style).

  Used for constraint generation conditioned on primitives:
    - x (target): constraint sequence
    - memory (source): primitive embeddings (fixed, used as K/V in cross-attention)
  """
  def __init__(self, config, output_size: int):
    super().__init__()
    if type(config) == dict:
      config = omegaconf.OmegaConf.create(config)

    self.config = config
    self.output_size = output_size
    self.rotary_emb = Rotary(
      config.model.hidden_size // config.model.n_heads)

    blocks = []
    for _ in range(config.model.n_blocks):
      blocks.append(DDiTCrossBlock(
        config.model.hidden_size,
        config.model.n_heads,
        config.model.cond_dim,
        dropout=config.model.dropout))
    self.blocks = nn.ModuleList(blocks)

    self.output_layer = DDitFinalLayer(
      config.model.hidden_size,
      output_size,
      config.model.cond_dim)
    self.scale_by_sigma = config.model.scale_by_sigma

  def forward(self, x, memory, c):
    """
    Args:
      x: Target sequence [B, L_tgt, D] - constraint embeddings (self-attention + cross-attention target)
      memory: Source sequence [B, L_src, D] - primitive embeddings (cross-attention source, K/V)
      c: Conditioning vector [B, D] - timestep embedding

    Returns:
      Output features [B, L_tgt, output_size]
    """
    rotary_cos_sin = self.rotary_emb(x)
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
      for block in self.blocks:
        x = block(x, memory, rotary_cos_sin, c, seqlens=None)
      x = self.output_layer(x, c)

    return x