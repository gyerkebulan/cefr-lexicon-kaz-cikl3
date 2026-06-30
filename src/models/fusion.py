"""
Fusion modules for combining context and morphology embeddings.
"""

import torch
import torch.nn as nn


class ConcatFusion(nn.Module):
    """Simple concatenation fusion."""
    
    def __init__(self, context_dim: int, morph_dim: int, output_dim: int = None):
        super().__init__()
        self.output_dim = output_dim or (context_dim + morph_dim)
        
        if output_dim:
            self.projection = nn.Linear(context_dim + morph_dim, output_dim)
        else:
            self.projection = None
    
    def forward(
        self,
        context_emb: torch.Tensor,
        morph_emb: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            context_emb: [batch, context_dim]
            morph_emb: [batch, morph_dim]
            
        Returns:
            fused: [batch, output_dim]
        """
        combined = torch.cat([context_emb, morph_emb], dim=1)
        
        if self.projection:
            return self.projection(combined)
        return combined


class GatedFusion(nn.Module):
    """
    Gated fusion: learns to weight context vs morphology.
    
    gate = sigmoid(W * [context; morph])
    output = gate * context + (1 - gate) * morph
    """
    
    def __init__(self, context_dim: int, morph_dim: int, output_dim: int = None):
        super().__init__()
        
        # Project both to same dimension
        self.output_dim = output_dim or context_dim
        
        self.context_proj = nn.Linear(context_dim, self.output_dim)
        self.morph_proj = nn.Linear(morph_dim, self.output_dim)
        
        # Gate network
        self.gate = nn.Sequential(
            nn.Linear(context_dim + morph_dim, self.output_dim),
            nn.Sigmoid()
        )
    
    def forward(
        self,
        context_emb: torch.Tensor,
        morph_emb: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            context_emb: [batch, context_dim]
            morph_emb: [batch, morph_dim]
            
        Returns:
            fused: [batch, output_dim]
        """
        # Project to same dimension
        context_proj = self.context_proj(context_emb)
        morph_proj = self.morph_proj(morph_emb)
        
        # Compute gate
        combined = torch.cat([context_emb, morph_emb], dim=1)
        g = self.gate(combined)
        
        # Gated combination
        return g * context_proj + (1 - g) * morph_proj


class ResidualGatedFusion(nn.Module):
    """
    Residual gated fusion: 
    gate = sigmoid(W * [context; morph])
    output = context + gate * project(morph)
    """
    
    def __init__(self, context_dim: int, morph_dim: int, output_dim: int = None):
        super().__init__()
        self.output_dim = output_dim or context_dim
        
        if context_dim != self.output_dim:
            self.context_proj = nn.Linear(context_dim, self.output_dim)
        else:
            self.context_proj = nn.Identity()
            
        self.morph_proj = nn.Linear(morph_dim, self.output_dim)
        
        self.gate = nn.Sequential(
            nn.Linear(context_dim + morph_dim, self.output_dim),
            nn.Sigmoid()
        )
        
    def forward(
        self,
        context_emb: torch.Tensor,
        morph_emb: torch.Tensor
    ) -> torch.Tensor:
        context_proj = self.context_proj(context_emb)
        morph_proj = self.morph_proj(morph_emb)
        
        combined = torch.cat([context_emb, morph_emb], dim=1)
        g = self.gate(combined)
        
        return context_proj + g * morph_proj


class CrossAttentionFusion(nn.Module):
    """
    Cross-Attention Fusion: Context acts as Query, Morphology as Key/Value.
    """
    
    def __init__(self, context_dim: int, morph_dim: int, output_dim: int = None, num_heads: int = 4):
        super().__init__()
        self.output_dim = output_dim or context_dim
        
        # Ensure attn_dim is divisible by num_heads
        self.attn_dim = max(context_dim, morph_dim, self.output_dim)
        if self.attn_dim % num_heads != 0:
            self.attn_dim = self.attn_dim + (num_heads - (self.attn_dim % num_heads))
            
        self.context_proj = nn.Linear(context_dim, self.attn_dim)
        self.morph_proj = nn.Linear(morph_dim, self.attn_dim)
        
        self.attn = nn.MultiheadAttention(embed_dim=self.attn_dim, num_heads=num_heads, batch_first=True)
        self.output_proj = nn.Linear(self.attn_dim, self.output_dim)
        
    def forward(
        self,
        context_emb: torch.Tensor,
        morph_emb: torch.Tensor
    ) -> torch.Tensor:
        # Add sequence dimension for attention computation [batch, 1, dim]
        q = self.context_proj(context_emb).unsqueeze(1)
        k = self.morph_proj(morph_emb).unsqueeze(1)
        v = k
        
        attn_out, _ = self.attn(q, k, v)
        
        # Residual connection
        out = attn_out.squeeze(1) + q.squeeze(1)
        
        return self.output_proj(out)


def get_fusion_module(
    fusion_type: str,
    context_dim: int,
    morph_dim: int,
    output_dim: int = None
) -> nn.Module:
    """Factory function for fusion modules."""
    if fusion_type == "concat":
        return ConcatFusion(context_dim, morph_dim, output_dim)
    elif fusion_type == "gated":
        return GatedFusion(context_dim, morph_dim, output_dim)
    elif fusion_type == "residual_gated":
        return ResidualGatedFusion(context_dim, morph_dim, output_dim)
    elif fusion_type == "cross_attention":
        return CrossAttentionFusion(context_dim, morph_dim, output_dim)
    else:
        raise ValueError(f"Unknown fusion type: {fusion_type}")
