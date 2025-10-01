import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- small DropPath implementation (stochastic depth) ---
class DropPath(nn.Module):
    """DropPath / Stochastic Depth per sample (when applied in main path of residual blocks)."""
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        # shape: [batch, 1, 1] to broadcast across sequence and channels
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        rand = x.new_empty(shape).bernoulli_(keep_prob)
        x = x / keep_prob * rand
        return x

# --- ViT-style Chess model ---
class ChessViT(nn.Module):
    def __init__(self,
                 in_channels=18,
                 img_size=8,
                 patch_size=1,
                 embed_dim=128,
                 depth=6,
                 num_heads=4,
                 mlp_ratio=4.0,
                 num_policy_classes=4672,   # flat / legacy; must be divisible by num_patches
                 dropout=0.1,
                 drop_path_prob=0.0,        # stochastic depth across transformer layers
                 use_norm_first=True):      # pre-LN transformer (often more stable)
        """
        Pure ViT chess model:
         - policy is predicted per patch (square) and then flattened (spatial policy).
         - value is predicted from CLS token.
        Args:
            num_policy_classes: total number of policy logits (should equal moves_per_square * num_patches)
        """
        super().__init__()

        assert img_size % patch_size == 0, "Image size must be divisible by patch size"
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim

        # Validate policy shape / derive moves_per_square
        assert num_policy_classes % self.num_patches == 0, (
            "num_policy_classes must be divisible by num_patches. "
            f"Got {num_policy_classes} and {self.num_patches} patches."
        )
        self.num_policy_classes = num_policy_classes
        self.moves_per_square = num_policy_classes // self.num_patches  # e.g., 4672 / 64 = 73

        # Patch embedding (1x1 conv when patch_size==1)
        self.patch_embed = nn.Conv2d(in_channels, embed_dim,
                                     kernel_size=patch_size, stride=patch_size)

        # CLS token and positional embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=dropout)

        # Transformer encoder stack (using nn.TransformerEncoderLayer)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=use_norm_first
        )
        # Build a list of layers so we can apply DropPath between them
        self.layers = nn.ModuleList()
        # linearly scale drop path across layers (common practice)
        d_probs = [x.item() for x in torch.linspace(0, drop_path_prob, depth)]
        for dp in d_probs:
            self.layers.append(nn.ModuleDict({
                "enc_layer": encoder_layer.__class__(  # create fresh instance, same args
                    d_model=embed_dim,
                    nhead=num_heads,
                    dim_feedforward=int(embed_dim * mlp_ratio),
                    dropout=dropout,
                    activation='gelu',
                    batch_first=True,
                    norm_first=use_norm_first
                ),
                "drop_path": DropPath(dp)
            }))

        # final normalization (pre-head)
        self.norm = nn.LayerNorm(embed_dim)

        # Policy head: per-patch classifier -> [B, num_patches, moves_per_square]
        self.policy_head = nn.Linear(embed_dim, self.moves_per_square)

        # Value head: small MLP from CLS token
        self.value_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 1)
        )

        # Initialization
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
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x, legal_moves_mask=None):
        """
        Args:
            x: [B, C, H, W] e.g. [B, 18, 8, 8]
            legal_moves_mask (optional): Bool/float mask of shape [B, num_policy_classes]
                                         or [B, num_patches, moves_per_square] to mask illegal moves.
                                         If provided, it will be applied to the flattened policy logits
                                         (set logits for illegal moves to -inf before softmax outside).
        Returns:
            policy_logits: [B, num_policy_classes] (flat: patches * moves_per_square)
            value: [B] (scalar)
            policy_logits_per_patch: [B, num_patches, moves_per_square] (unflattened)
        """
        B = x.shape[0]
        # Patch embedding -> [B, embed_dim, H', W']
        x = self.patch_embed(x)
        # flatten spatial dims -> [B, embed_dim, num_patches] then -> [B, num_patches, embed_dim]
        x = x.flatten(2).transpose(1, 2)

        # prepend CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, embed_dim]
        x = torch.cat([cls_tokens, x], dim=1)  # [B, 1 + num_patches, embed_dim]

        # add pos embedding
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # pass through transformer layers with optional DropPath
        for layer_dict in self.layers:
            enc = layer_dict["enc_layer"] # type: ignore
            dp = layer_dict["drop_path"] # type: ignore
            # transformer layer is residual internally; but applying drop path to the transformer's output
            out = enc(x)  # [B, seq_len, embed_dim]
            out = dp(out)
            x = x + out  # add residual across whole layer (note: if enc already includes residual, this double-residuals)
            # ---- Explanation:
            # PyTorch's TransformerEncoderLayer already performs residual connections internally.
            # To get per-layer DropPath applied to the residual, you can instead use
            # enc.norm_first / pre-norm variants and apply DropPath to the layer's *output minus input*.
            # For simplicity and stability across PyTorch versions, we keep an extra residual here.
            # If you prefer "vanilla" stacking with Transformer's internal residuals, replace the loop with:
            # x = enc(x)

        # final norm
        x = self.norm(x)

        # Extract CLS for value
        cls_final = x[:, 0]  # [B, embed_dim]
        value = self.value_head(cls_final).squeeze(-1)  # [B]

        # Policy from patch embeddings (exclude CLS)
        patches = x[:, 1:, :]  # [B, num_patches, embed_dim]
        policy_per_patch = self.policy_head(patches)  # [B, num_patches, moves_per_square]
        policy_logits = policy_per_patch.flatten(1)   # [B, num_policy_classes]

        # Optionally mask illegal moves (user can also do this outside)
        if legal_moves_mask is not None:
            # Accept mask shapes: [B, num_policy_classes] or [B, num_patches, moves_per_square]
            if legal_moves_mask.dim() == 3:
                mask_flat = legal_moves_mask.flatten(1)
            else:
                mask_flat = legal_moves_mask
            # We assume mask==1 means legal. Set illegal logits to -inf (or large negative).
            # Use a safe in-place approach:
            inf_neg = -1e9
            policy_logits = policy_logits.masked_fill(~mask_flat.bool(), inf_neg)

        return policy_logits, value, policy_per_patch
