import torch
import torch.nn as nn
import torch.nn.functional as F

from run_timer import TIMER


class ChessViT(nn.Module):
    def __init__(self, 
                 in_channels=18, 
                 img_size=8, 
                 patch_size=1, 
                 embed_dim=128, 
                 depth=6, 
                 num_heads=4, 
                 mlp_ratio=4.0, 
                 num_policy_classes=4672):
        super().__init__()
        
        assert img_size % patch_size == 0, "Image size must be divisible by patch size"
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        
        # Patch embedding: 1x1 conv if patch_size=1
        self.patch_embed = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        
        # Learnable classification token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # Learnable position embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=0.1)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim, 
                nhead=num_heads, 
                dim_feedforward=int(embed_dim * mlp_ratio), 
                dropout=0.1, 
                activation='gelu',
                batch_first=True  # for [B, N, C] input
            )
            for _ in range(depth)
        ])
        
        # Layer norm before heads
        self.norm = nn.LayerNorm(embed_dim)
        
        # Policy head (high-dimensional)
        self.policy_head = nn.Linear(embed_dim, num_policy_classes)
        # Value head (scalar)
        self.value_head = nn.Linear(embed_dim, 1)
        
        # Initialize weights
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
    
    def forward(self, x):
        """
        x: [B, C, H, W] = [batch_size, 18, 8, 8]
        """
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # [B, embed_dim, H/patch, W/patch]
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]
        
        # Append CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, embed_dim]
        x = torch.cat((cls_tokens, x), dim=1)  # [B, 1 + num_patches, embed_dim]
        
        # Add positional embedding
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Layer norm
        x = self.norm(x)
        
        # Extract CLS token for heads
        cls_token_final = x[:, 0]  # [B, embed_dim]
        
        # Policy and value heads
        policy_logits = self.policy_head(cls_token_final)  # [B, 4600]
        value = self.value_head(cls_token_final).squeeze(-1)  # [B]
        
        return policy_logits, value


if __name__ == "__main__":
    model = ChessViT()
    x = torch.randn(64, 18, 8, 8)
    TIMER.start("forward pass")
    policy_logits, value = model(x)
    TIMER.stop("forward pass")
    print("Policy logits shape:", policy_logits.shape)
    print("Value shape:", value.shape)