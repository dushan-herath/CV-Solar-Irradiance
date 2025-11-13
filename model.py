from typing import Optional
import torch
from torch import nn
import timm


class ImageEncoder(nn.Module):
    def __init__(self, model_name: str = 'vit_base_patch16_224', pretrained: bool = True, freeze: bool = True):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0, global_pool='avg')
        self.out_dim = self.backbone.num_features
        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) -> (B, img_embed_dim)
        return self.backbone(x)


class TS_Encoder(nn.Module):
    def __init__(self, ts_feat_dim: int, ts_embed_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(ts_feat_dim, ts_embed_dim),
            nn.GELU(),
            nn.LayerNorm(ts_embed_dim),
            nn.Dropout(dropout),
        )
        self.out_dim = ts_embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F) -> (B, T, ts_embed_dim)
        return self.proj(x)



class CrossAttentionFusion(nn.Module):
    """
    Fuse last N image embeddings with full TS embeddings using cross-attention.
    Each image attends to all numeric history **up to its timestamp** (causal).
    """
    def __init__(self, img_dim: int, ts_dim: int, fused_dim: int, nhead: int = 4, dropout: float = 0.1):
        super().__init__()
        self.q_proj = nn.Linear(img_dim, fused_dim)
        self.k_proj = nn.Linear(ts_dim, fused_dim)
        self.v_proj = nn.Linear(ts_dim, fused_dim)
        self.attn = nn.MultiheadAttention(embed_dim=fused_dim, num_heads=nhead, batch_first=True, dropout=dropout)
        self.out_proj = nn.Linear(fused_dim, fused_dim)
    
    def forward(self, img_feats: torch.Tensor, ts_feats: torch.Tensor):
        """
        img_feats: (B, T_img, img_dim)  # last N images
        ts_feats:  (B, T_ts, ts_dim)    # full numeric history
        Returns: (B, T_img, fused_dim)
        """
        B, T_img, _ = img_feats.shape
        _, T_ts, _ = ts_feats.shape

        # Project to Q, K, V
        Q = self.q_proj(img_feats)  # (B, T_img, fused_dim)
        K = self.k_proj(ts_feats)   # (B, T_ts, fused_dim)
        V = self.v_proj(ts_feats)   # (B, T_ts, fused_dim)

        # Build causal mask for last-N images
        mask = torch.ones(T_img, T_ts, device=img_feats.device).bool()  # True = masked
        ts_start = T_ts - T_img  # index of TS step corresponding to the first image
        for i in range(T_img):
            # Image i can attend to all TS steps up to its timestamp
            img_ts_index = ts_start + i
            mask[i, :img_ts_index + 1] = False  # unmask allowed steps

        # Apply cross-attention
        fused, _ = self.attn(Q, K, V, attn_mask=mask)

        # Linear projection
        fused = self.out_proj(fused)  # (B, T_img, fused_dim)
        return fused



class FusionTransformer(nn.Module):
    def __init__(self, input_dim: int, d_model: int = 256, nhead: int = 8, num_layers: int = 3, dim_feedforward: int = 512, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        x = self.transformer(x)
        return x



class MultimodalForecaster(nn.Module):
    def __init__(
        self,
        img_encoder: ImageEncoder,
        ts_feat_dim: int,
        img_embed_dim: Optional[int] = None,
        ts_embed_dim: int = 64,
        fused_dim: int = 256,
        d_model: int = 256,
        num_layers: int = 3,
        nhead: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        horizon: int = 25,
        target_dim: int = 3,
    ):
        super().__init__()
        self.img_encoder = img_encoder
        self.img_embed_dim = img_encoder.out_dim if img_embed_dim is None else img_embed_dim
        self.ts_encoder = TS_Encoder(ts_feat_dim=ts_feat_dim, ts_embed_dim=ts_embed_dim, dropout=dropout)
        self.cross_attn = CrossAttentionFusion(self.img_embed_dim, ts_embed_dim, fused_dim=fused_dim, nhead=nhead, dropout=dropout)
        self.temporal = FusionTransformer(input_dim=fused_dim, d_model=d_model, nhead=nhead, num_layers=num_layers, dim_feedforward=dim_feedforward, dropout=dropout)
        self.horizon = horizon
        self.target_dim = target_dim

        # regression head
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, horizon * target_dim)
        )

    def forward(self, imgs: torch.Tensor, ts: torch.Tensor) -> torch.Tensor:
        """
        imgs: (B, T_img, C, H, W)
        ts:   (B, T_ts, F)
        Returns: (B, horizon, target_dim)
        """
        B, T_img, C, H, W = imgs.shape
        T_ts = ts.shape[1]

        # Encode images
        imgs_flat = imgs.view(B * T_img, C, H, W)
        img_feats_flat = self.img_encoder(imgs_flat)
        img_feats = img_feats_flat.view(B, T_img, -1)  # (B, T_img, img_embed_dim)

        # Encode TS
        ts_feats = self.ts_encoder(ts)  # (B, T_ts, ts_embed_dim)

        # Cross-attention fusion
        fused_feats = self.cross_attn(img_feats, ts_feats)  # (B, T_img, fused_dim)
        fused_feats = img_feats

        # Temporal transformer over fused tokens
        out_seq = self.temporal(fused_feats)

        # Use last token for prediction
        last = out_seq[:, -1, :]
        out = self.head(last)
        out = out.view(B, self.horizon, self.target_dim)
        return out


# -------------------------------
# Smoke test
# -------------------------------
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vit = ImageEncoder(model_name='vit_small_patch16_224', pretrained=False, freeze=True)
    model = MultimodalForecaster(img_encoder=vit, ts_feat_dim=5, ts_embed_dim=64, fused_dim=128, d_model=128, num_layers=2, nhead=4, horizon=6, target_dim=3).to(device)

    B = 2
    T_img = 5
    T_ts = 30
    imgs = torch.randn(B, T_img, 3, 224, 224).to(device)
    ts = torch.randn(B, T_ts, 5).to(device)

    preds = model(imgs, ts)
    print("preds.shape:", preds.shape)  # (B, horizon, target_dim)
