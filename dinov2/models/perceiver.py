import torch
import torch.nn as nn
from einops import rearrange, repeat
from torch import einsum


def rearrange_many(tensors, pattern, **kwargs):
    return [rearrange(t, pattern, **kwargs) for t in tensors]

class FeedForward(nn.Module):
    """
    FeedForward module for Perceiver.
    """
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.ln = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim ,hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
    def forward(self, x):
        return self.mlp(self.ln(x))


class Attention(nn.Module):
    """
    Attention module for Perceiver.
    """
    def __init__(self, dim, dim_head, num_heads):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.dim = dim
        self.dim_head = dim_head
        self.num_heads = num_heads
        self.hidden_dim = dim_head * num_heads

        self.to_q = nn.Linear(dim, self.hidden_dim, bias=False)
        self.to_kv = nn.Linear(dim, self.hidden_dim * 2, bias=False)
        self.to_out = nn.Linear(self.hidden_dim, dim)

        self.media_norm = nn.LayerNorm(dim)
        self.latent_norm = nn.LayerNorm(dim)

    def forward(self, x, latents, mask=None):

        num_heads = self.num_heads
        n_batch, n_features, d = x.shape
        n_queries = latents.shape[1]

        #x and latents come from different domains, so we need to normalize them separately
        x = self.media_norm(x)
        latents = self.latent_norm(latents)

        n_heads = self.num_heads

        #compute queries from the latents
        q = self.to_q(latents)
        q = rearrange(q, 'b q (h d) -> b h q d', h=num_heads)
        q = q * self.scale
        assert q.shape == torch.Size([n_batch, num_heads, n_queries, self.dim_head])

        #compute keys and values from the media
        #i think in the paper, they concatenate media and latents together 
        kv_input = torch.cat([x, latents], dim=-2)
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)
        k, v = rearrange_many((k, v), 'b f (h d) -> b h f d', h=num_heads)
        assert v.shape == torch.Size([n_batch, num_heads, n_features + n_queries, self.dim_head])

        #compute attention
        sim = einsum('b h q d, b h f d -> b h q f', q, k)

        #AI-generated masking logic. To verify
        # --- MASKING LOGIC START ---
        if mask is not None:
            # mask shape is [Batch, 256]. 
            # The 'Keys' (k) length is [256 + Num_Latents].
            # We want to mask the 256 patches, but NEVER mask the Latents.
            
            # Create a 'False' mask for the latents (visible)
            # shape: [Batch, Num_Latents]
            latent_mask = torch.zeros((n_batch, n_queries), dtype=torch.bool, device=mask.device)
            
            # Combine: [Mask for Patches, Mask for Latents]
            full_mask = torch.cat((mask, latent_mask), dim=1) 
            
            # Reshape to match attention matrix dimensions [B, 1, 1, Key_Len]
            attn_mask = rearrange(full_mask, 'b j -> b 1 1 j')
            
            # Fill masked spots with -Infinity (so Softmax makes them 0)
            # We use a very large negative number safe for mixed precision training
            max_neg_value = -torch.finfo(sim.dtype).max
            sim = sim.masked_fill(attn_mask, max_neg_value)
        # --- MASKING LOGIC END ---


        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        alphas = sim.softmax(dim=-1)
        
        out = einsum('b h q f, b h f v -> b h q v', alphas, v)
        out = rearrange(out, 'b h q v -> b q (h v)')
        
        #compute output
        out = self.to_out(out)
        return out

class PerceiverResampler(nn.Module):
    def __init__(self, dim, num_layers, num_heads, dim_head, num_latents, n_pos_embeddings):
        super().__init__()
        self.dim = dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dim_head = dim_head
        self.n_time_embeddings = n_pos_embeddings
        self.num_latents = num_latents


        self.latents = nn.Parameter(torch.randn(num_latents, dim))
        self.pos_emb = nn.Parameter(torch.randn(n_pos_embeddings, dim))
        self.layers = nn.ModuleList([])
        for _ in range(num_layers):
            self.layers.append(
                nn.ModuleList([
                    Attention(dim, dim_head, num_heads),
                    FeedForward(dim, dim * 4),
                ])
            )

        self.norm = nn.LayerNorm(dim)

    def forward(self, x, mask=None):
        #Takes in input media embeddings and outputs resampled features

        batch_size, n_features, d = x.shape
       
        #add time position embeddings to input media embeddings
        x = x + self.pos_emb.unsqueeze(0)

        #do I have to flatten input media embeddings?

        latents = repeat(self.latents, 'q d -> b q d', b=batch_size)

        for attn, ff in self.layers:
            latents = attn(x, latents, mask) + latents
            latents = ff(latents) + latents
        latents = self.norm(latents)
        return latents

    