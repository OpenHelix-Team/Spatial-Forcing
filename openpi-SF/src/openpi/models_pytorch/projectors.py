"""Implementation of additional projectors for additional inputs to the VLA models."""
import torch
import torch.nn as nn
import openpi.models.gemma as _gemma

class AlignProjector(nn.Module):
    """
    calculate the alignment between LLM and VGGT embeddings.
    """
    def __init__(
            self,
            llm_dim: int, 
            vggt_dim: int,
            use_vlm_norm: bool = False,
        ) -> None:
        super().__init__()

        self.llm_dim = llm_dim
        self.vggt_dim = vggt_dim

        self.fc1 = nn.Linear(self.llm_dim, 2 * self.vggt_dim, bias=True)
        self.fc2 = nn.Linear(2 * self.vggt_dim, 2 * self.vggt_dim, bias=True)
        self.act_fn1 = nn.GELU()
        
        self.vlm_norm = nn.LayerNorm(llm_dim) if use_vlm_norm else None

        self.initialize_weights()
    
    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

    def align_dimension(self, LLM_embedding: torch.Tensor = None) -> torch.Tensor:
        if self.vlm_norm is not None:
            LLM_embedding = self.vlm_norm(LLM_embedding)
        projected_features = self.fc1(LLM_embedding)
        projected_features = self.act_fn1(projected_features)
        projected_features = self.fc2(projected_features)
        return projected_features
    
    def compute_align_loss_cosine(self, vision_hidden, vggt_hidden, align_mask):
        # vision_hidden has a shape of (bs, N, D)
        def mean_flat(x):
            return torch.mean(x, dim=list(range(1, len(x.size()))))
        align_loss = 0
        bsz = vision_hidden.shape[0]
        for _vision, _vggt, _mask in zip(vision_hidden, vggt_hidden, align_mask):
            _vision = torch.nn.functional.normalize(_vision, dim=-1)
            _vggt = torch.nn.functional.normalize(_vggt, dim=-1)
            # align_loss += 1 - torch.mean(vision_hidden * vggt_hidden).sum(dim=-1).mean()
            align_loss += 1 - mean_flat((_vision * _vggt)[_mask].sum(dim=-1))  # Cosine similarity loss
        align_loss /= bsz  # Average over batch size
        return align_loss
    
    def forward(self, LLM_emb, target_emb, align_mask):
        # project vla dimension and calculate align loss
        LLM_emb = self.align_dimension(LLM_emb)
        align_loss = self.compute_align_loss_cosine(LLM_emb, target_emb, align_mask).mean()  # mean for sequence length
        return align_loss
