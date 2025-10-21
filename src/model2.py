import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------
# Utilities: DropPath (stochastic depth)
# -------------------------
class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x / keep_prob * random_tensor
        return output

# -------------------------
# Small MLP / FeedForward
# -------------------------
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0, activation=nn.GELU):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            activation(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.net(x)

# -------------------------
# Transformer Block (explicit attention so we can add relative bias)
# -------------------------
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.0, drop_path=0.0, layer_scale_init_value=1e-5):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        assert dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        # qkv projection
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.out_proj = nn.Linear(dim, dim)

        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        mlp_hidden = int(dim * mlp_ratio)
        self.ffn = FeedForward(dim, mlp_hidden, dropout=dropout)

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        # LayerScale
        if layer_scale_init_value > 0:
            self.gamma_1 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            self.gamma_2 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        else:
            self.gamma_1 = None
            self.gamma_2 = None

    def forward(self, x, rel_pos_bias=None):
        """
        x: [B, N, C]
        rel_pos_bias: [num_heads, N, N] or None
        """
        # ---- attention ----
        y = self.norm1(x)
        B, N, C = y.shape

        qkv = self.qkv(y)  # [B, N, 3C]
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        # qkv: [3, B, heads, N, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]  # each [B, heads, N, head_dim]

        # scaled dot-product
        scale = 1.0 / math.sqrt(self.head_dim)
        attn = torch.einsum("bhid,bhjd->bhij", q, k) * scale  # [B, heads, N, N]
        if rel_pos_bias is not None:
            # rel_pos_bias is [num_heads, N, N] -> broadcast to [B, heads, N, N]
            attn = attn + rel_pos_bias.unsqueeze(0)

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        out = torch.einsum("bhij,bhjd->bhid", attn, v)  # [B, heads, N, head_dim]
        out = out.permute(0, 2, 1, 3).reshape(B, N, C)
        out = self.out_proj(out)
        out = self.proj_dropout(out)

        if self.gamma_1 is not None:
            x = x + self.drop_path(out * self.gamma_1.unsqueeze(0).unsqueeze(0))
        else:
            x = x + self.drop_path(out)

        # ---- ffn ----
        y2 = self.norm2(x)
        ffn_out = self.ffn(y2)
        if self.gamma_2 is not None:
            x = x + self.drop_path(ffn_out * self.gamma_2.unsqueeze(0).unsqueeze(0))
        else:
            x = x + self.drop_path(ffn_out)
        return x

# -------------------------
# Relative 2D bias helper (computes a parameter lookup for pairwise offsets)
# -------------------------
class RelativePosition2D(nn.Module):
    def __init__(self, num_heads, height=8, width=8):
        super().__init__()
        self.num_heads = num_heads
        self.h = height
        self.w = width
        # relative coordinates range: [-h+1, h-1], [-w+1, w-1] -> shift by (h-1) and (w-1) for indexing
        self.relative_bias_table = nn.Parameter(torch.zeros(num_heads, (2 * height - 1) * (2 * width - 1)))
        # precompute index map
        coords_h = torch.arange(height)
        coords_w = torch.arange(width)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))  # [2, H, W]
        coords_flat = coords.reshape(2, -1)  # [2, N]
        relative_coords = coords_flat[:, :, None] - coords_flat[:, None, :]  # [2, N, N]
        relative_coords[0] += height - 1
        relative_coords[1] += width - 1
        relative_coords[0] *= 2 * width - 1
        relative_index = relative_coords[0] + relative_coords[1]  # shape [N, N]
        self.register_buffer("relative_index", relative_index.long())

        nn.init.trunc_normal_(self.relative_bias_table, std=0.02)

    def forward(self):
        # return [num_heads, N, N]
        # indexing into table
        bias = self.relative_bias_table[:, self.relative_index.view(-1)].view(self.num_heads, -1, -1) # type: ignore
        return bias  # [heads, N, N]

# -------------------------
# Conv stem: small conv stack to provide local bias
# -------------------------
class ConvStem(nn.Module):
    def __init__(self, in_ch, out_ch=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch // 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch // 2, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.net(x)

# -------------------------
# ChessViTV2 main module
# -------------------------
class ChessViTv2(nn.Module):
    def __init__(
        self,
        in_channels,
        img_size=8,
        embed_dim=256,
        depth=10,
        num_heads=8,
        mlp_ratio=4.0,
        num_policy_classes=4672,
        num_dest_per_src=73,   # default alpha-zero style
        dropout=0.1,
        drop_path_rate=0.1,
        conv_stem_channels=64,
        use_relative_bias=True,
        layer_scale_init_value=1e-5,
    ):
        super().__init__()
        assert img_size == 8, "This implementation assumes 8x8 chessboard"
        self.img_size = img_size
        self.num_patches = img_size * img_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_dest_per_src = num_dest_per_src

        # Conv stem to add local bias
        self.conv_stem = ConvStem(in_channels, out_ch=conv_stem_channels)

        # Project conv features to embedding dim per square (1x1 conv)
        self.proj = nn.Conv2d(conv_stem_channels, embed_dim, kernel_size=1, stride=1)

        # CLS token and absolute positional embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        # Relative bias
        self.use_relative_bias = use_relative_bias
        if use_relative_bias:
            self.rel_pos = RelativePosition2D(num_heads=num_heads, height=img_size, width=img_size)
        else:
            self.rel_pos = None

        # Transformer blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                drop_path=dpr[i],
                layer_scale_init_value=layer_scale_init_value,
            )
            for i in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        # Policy heads (dual head: source square + per-source destinations)
        self.src_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, self.num_patches)  # 64 source logits
        )
        # per-source destination head: map each square token to num_dest_per_src logits
        self.dest_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, num_dest_per_src)  # per-source dest logits
        )

        # Value head: conv pool over token map
        # We'll reshape per-square tokens -> [B, C, 8, 8] and run small convs
        self.value_conv = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),  # global pool => [B, C, 1, 1]
        )
        self.value_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, 1),
        )
        # Outcome distribution head (win/draw/loss) as auxiliary
        self.outcome_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, 3)  # logits for [win, draw, loss]
        )

        # init
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
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        x: [B, C, 8, 8]
        returns:
            policy_logits_flat: [B, 4672]   # not yet masked
            value_scalar: [B]
            outcome_logits: [B, 3]
            extra dict with src_logits [B,64] and dest_logits [B,64,num_dest_per_src]
        """
        B = x.shape[0]

        # conv stem -> [B, conv_stem_channels, 8, 8]
        x_conv = self.conv_stem(x)

        # project to embed dim (per-square)
        x_proj = self.proj(x_conv)  # [B, embed_dim, 8, 8]
        tokens = x_proj.flatten(2).transpose(1, 2)  # [B, 64, embed_dim]

        # prepend CLS
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B,1,dim]
        x_tokens = torch.cat((cls_tokens, tokens), dim=1)  # [B, 65, dim]

        # add absolute pos embedding
        x_tokens = x_tokens + self.pos_embed
        x_tokens = self.pos_drop(x_tokens)

        # compute relative bias once per forward if used
        rel_bias = None
        if self.use_relative_bias:
            if self.rel_pos is None:
                raise Exception("Relative position bias used but not initialized")
            rel_bias = self.rel_pos()  # [heads, N, N] for N=64
            # need to expand to include CLS token -> bias shape should be [heads, N+1, N+1]
            # We will pad zeros on rows/cols corresponding to cls token so attention involving CLS has no relative bias
            pad = torch.zeros(self.num_heads, 1, rel_bias.shape[-1], device=rel_bias.device, dtype=rel_bias.dtype)
            rel_bias = torch.cat([pad, torch.cat([pad.transpose(1,2), rel_bias], dim=2)], dim=1)
            # After above ops rel_bias is [heads, N+1, N+1] where first row/col = zeros for cls token

        # transformer blocks
        for blk in self.blocks:
            x_tokens = blk(x_tokens, rel_pos_bias=rel_bias)

        x_tokens = self.norm(x_tokens)  # [B, 65, dim]

        # separate CLS and board tokens
        cls_final = x_tokens[:, 0, :]  # [B, dim]
        board_tokens = x_tokens[:, 1:, :]  # [B, 64, dim]

        # --- Policy heads (dual) ---
        src_logits = self.src_head(board_tokens.mean(dim=1))  # [B, 64]
        # dest logits per source: apply dest_head on each square token
        # board_tokens -> [B*64, dim] -> dest logits -> [B,64,num_dest]
        dest_logits = self.dest_head(board_tokens.reshape(-1, self.embed_dim)).reshape(B, self.num_patches, self.num_dest_per_src)

        # flatten to single policy vector (AlphaZero style mapping of 64 x num_dest into 4672)
        policy_logits_flat = dest_logits.reshape(B, -1)  # [B, 64 * num_dest_per_src]
        # Note: some implementations multiply by src_probs etc. Here we will concatenate or combine via src distribution:
        # To compute final probability over moves, one typical approach is:
        #   final_logits = log_softmax(src_logits) + per-source dest logits (apply broadcasting)
        # We will leave both outputs for training flexibility and also provide a convenience function to combine them.

        # --- Value head ---
        # reshape board tokens into map [B, C, 8, 8] for conv pooling
        board_map = board_tokens.transpose(1, 2).reshape(B, self.embed_dim, self.img_size, self.img_size)
        v = self.value_conv(board_map).view(B, self.embed_dim)  # after AdaptiveAvgPool2d(1) => [B, C]
        value_scalar = self.value_mlp(v).squeeze(-1)  # [B]

        # outcome distribution
        outcome_logits = self.outcome_head(cls_final)  # [B, 3]

        extras = {
            "cls": cls_final,
            "src_logits": src_logits,                    # [B, 64]
            "dest_logits": dest_logits,                  # [B, 64, num_dest]
            "policy_logits_flat_unmasked": policy_logits_flat,  # [B, 64*num_dest]
        }

        return policy_logits_flat, value_scalar, outcome_logits, extras

    # -------------
    # Helper: combine src + dest into final flat logits (log-space) and apply legal mask
    # -------------
    def combine_src_dest_logits(self, src_logits, dest_logits, legal_mask=None, eps=1e-9):
        """
        Combine src logits and dest logits into final flattened policy logits.
        src_logits: [B, 64]
        dest_logits: [B, 64, D]
        legal_mask: [B, 64*D] boolean mask of legal moves (or None)
        returns: combined_logits [B, 64*D] (un-normalized logits)
        Implementation: log-softmax(src) + dest_logits (logits) combined in log-space:
            combined_logprob = log_softmax(src) expanded + dest_logits
        """
        B = src_logits.shape[0]
        # log-softmax over sources
        src_logprob = F.log_softmax(src_logits, dim=1).unsqueeze(-1)  # [B, 64, 1]
        combined = dest_logits + src_logprob  # [B, 64, D]
        flat = combined.reshape(B, -1)  # [B, 64*D]
        if legal_mask is not None:
            # mask out illegal moves by setting logits to a very large negative number
            flat = flat.masked_fill(~legal_mask, float("-1e9"))
        return flat

    # ----------------
    # Convenience: apply legal move mask to flat logits, return probabilities
    # ----------------
    def final_policy_probs(self, src_logits, dest_logits, legal_mask=None):
        flat = self.combine_src_dest_logits(src_logits, dest_logits, legal_mask=legal_mask)
        probs = F.softmax(flat, dim=-1)
        return probs

# -------------------------
# Example usage
# -------------------------
if __name__ == "__main__":
    # quick smoke test
    B = 2
    C = 18  # example channel count
    x = torch.randn(B, C, 8, 8)
    model = ChessViTv2(in_channels=C)
    policy_flat, value, outcome, extras = model(x)
    print("policy_flat:", policy_flat.shape)       # [B, 64*73] -> 4672 if D=73
    print("value:", value.shape)
    print("outcome:", outcome.shape)
    print("src_logits:", extras["src_logits"].shape)
    print("dest_logits:", extras["dest_logits"].shape)
