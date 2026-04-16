"""
Feature Extraction: CNN + Transformer Hybrid Encoder

[수정 사항 - 분석 보고서 반영]
1. PalmprintRecognitionModel: CenterLoss / HardTripletLoss 제거 → ArcFace 단독
   - 이전 실험에서 ArcFace only + scale=32 + margin=0.3 로 Rank-1 84% 달성
   - CenterLoss + Triplet 조합이 score collapse를 유발함을 실험으로 확인
2. ArcFace: scale=32, margin=0.3 (소규모 데이터셋 권장값)
3. freeze_encoder_layers(): Phase1 로드 시 부분 동결 지원
4. QualityGuidedFusion: Sigmoid gate 유지 (독립적 branch 가중치)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# ==================================================================
# 1. Local Texture Branch (CNN)
# ==================================================================

class DepthwiseBlock(nn.Module):
    """Depthwise-Separable Convolution."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.dw = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, stride=stride, padding=1, groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.Hardswish(inplace=True),
        )
        self.pw = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.Hardswish(inplace=True),
        )

    def forward(self, x):
        return self.pw(self.dw(x))


class LocalTextureBranch(nn.Module):
    """
    MobileNet-style CNN for local palmprint texture.
    Input : (B, 3, 128, 128)
    Output: (B, local_dim)
    """

    def __init__(self, local_dim: int = 256):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, 3, 2, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.Hardswish(True),
        )
        self.layers = nn.Sequential(
            DepthwiseBlock(16,  32,  1),
            DepthwiseBlock(32,  64,  2),
            DepthwiseBlock(64,  64,  1),
            DepthwiseBlock(64,  128, 2),
            DepthwiseBlock(128, 128, 1),
            DepthwiseBlock(128, 256, 2),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(256, local_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layers(x)
        x = self.pool(x).flatten(1)
        return self.proj(x)


# ==================================================================
# 2. Global Structure Branch (Lightweight Transformer)
# ==================================================================

class PatchEmbedding(nn.Module):
    def __init__(self, in_ch: int = 3, embed_dim: int = 128, patch_size: int = 8):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, embed_dim, patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        B, C, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)
        return self.norm(x)


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int = 4, mlp_ratio: float = 2.0,
                 dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, dim), nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n = self.norm1(x)
        a, _ = self.attn(n, n, n)
        x = x + a
        x = x + self.mlp(self.norm2(x))
        return x


class GlobalStructureBranch(nn.Module):
    """
    MobileViT-style Transformer for global palm structure.
    Input : (B, 3, 128, 128)
    Output: (B, global_dim)
    """

    def __init__(self, global_dim: int = 256, embed_dim: int = 128,
                 patch_size: int = 8, num_layers: int = 4, num_heads: int = 4):
        super().__init__()
        self.patch_embed = PatchEmbedding(3, embed_dim, patch_size)
        self.cls_token   = nn.Parameter(torch.zeros(1, 1, embed_dim))
        n_patches        = (128 // patch_size) ** 2 + 1
        self.pos_embed   = nn.Parameter(torch.zeros(1, n_patches, embed_dim))
        self.blocks      = nn.Sequential(
            *[TransformerBlock(embed_dim, num_heads) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.proj = nn.Linear(embed_dim, global_dim)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B      = x.size(0)
        tokens = self.patch_embed(x)
        cls    = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1) + self.pos_embed
        tokens = self.blocks(tokens)
        tokens = self.norm(tokens)
        return self.proj(tokens[:, 0])


# ==================================================================
# 3. Quality-Guided Fusion  (Sigmoid gate 유지)
# ==================================================================

class QualityGuidedFusion(nn.Module):
    """
    local_feat + global_feat 을 ROI 품질 점수 또는
    학습 가능한 독립 Sigmoid gate로 융합.
    """

    def __init__(self, feat_dim: int = 256, fused_dim: int = 512):
        super().__init__()
        self.local_gate = nn.Sequential(
            nn.Linear(feat_dim * 2, feat_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim // 2, 1),
            nn.Sigmoid(),
        )
        self.global_gate = nn.Sequential(
            nn.Linear(feat_dim * 2, feat_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim // 2, 1),
            nn.Sigmoid(),
        )
        self.proj = nn.Linear(feat_dim * 2, fused_dim)
        self.norm = nn.LayerNorm(fused_dim)

    def forward(
        self,
        local_feat:  torch.Tensor,
        global_feat: torch.Tensor,
        local_w:  Optional[torch.Tensor] = None,
        global_w: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        concat = torch.cat([local_feat, global_feat], dim=-1)

        if local_w is not None and global_w is not None:
            lw = local_w.view(-1, 1)
            gw = global_w.view(-1, 1)
        else:
            lw = self.local_gate(concat)
            gw = self.global_gate(concat)

        weighted = lw * local_feat + gw * global_feat
        residual = local_feat + global_feat
        fused    = torch.cat([weighted, residual], dim=-1)
        return self.norm(self.proj(fused))


# ==================================================================
# 4. Hybrid Encoder
# ==================================================================

class HybridPalmprintEncoder(nn.Module):
    """
    Input : (B, 3, 128, 128)
    Output: (B, embed_dim)
    """

    def __init__(self, local_dim: int = 256, global_dim: int = 256,
                 embed_dim: int = 512):
        super().__init__()
        self.local_branch  = LocalTextureBranch(local_dim)
        self.global_branch = GlobalStructureBranch(global_dim)
        self.fusion = QualityGuidedFusion(
            feat_dim=max(local_dim, global_dim), fused_dim=embed_dim
        )
        self.embed_dim  = embed_dim
        self.local_proj = (nn.Identity() if local_dim == global_dim
                           else nn.Linear(local_dim, global_dim))

    def forward(self, x: torch.Tensor,
                local_w:  Optional[torch.Tensor] = None,
                global_w: Optional[torch.Tensor] = None) -> torch.Tensor:
        local_feat  = self.local_proj(self.local_branch(x))
        global_feat = self.global_branch(x)
        return self.fusion(local_feat, global_feat, local_w, global_w)


# ==================================================================
# 5. ArcFace Head
# ==================================================================

class ArcFaceHead(nn.Module):
    """
    Additive Angular Margin Loss.
    소규모 데이터셋 권장: scale=32, margin=0.3
    """

    def __init__(self, embed_dim: int, num_classes: int,
                 margin: float = 0.3, scale: float = 32.0):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_classes, embed_dim))
        nn.init.xavier_uniform_(self.weight)
        self.margin = margin
        self.scale  = scale
        self.cos_m  = math.cos(margin)
        self.sin_m  = math.sin(margin)
        self.th     = math.cos(math.pi - margin)
        self.mm     = math.sin(math.pi - margin) * margin

    def forward(self, feat: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        w         = F.normalize(self.weight, dim=1)
        cos_theta = torch.matmul(feat, w.t()).clamp(-1 + 1e-6, 1 - 1e-6)
        sin_theta = torch.sqrt(1.0 - cos_theta ** 2)

        cos_theta_m = cos_theta * self.cos_m - sin_theta * self.sin_m
        cos_theta_m = torch.where(cos_theta > self.th, cos_theta_m,
                                  cos_theta - self.mm)

        one_hot = torch.zeros_like(cos_theta)
        one_hot.scatter_(1, labels.unsqueeze(1), 1.0)
        logits = one_hot * cos_theta_m + (1 - one_hot) * cos_theta
        return logits * self.scale


# ==================================================================
# 6. Recognition Model  — ArcFace 단독
#
# [핵심 수정]
# - CenterLoss / HardTripletLoss 완전 제거
#   → score collapse의 근본 원인 제거
#   → 이전 실험(ArcFace only)에서 Rank-1 84.47% 달성한 구조로 복원
# - arc_margin=0.3, arc_scale=32 (소규모 데이터셋 권장값)
# - freeze_encoder_layers(): Phase1 encoder 로드 후 부분 동결 지원
# ==================================================================

class PalmprintRecognitionModel(nn.Module):
    """
    End-to-end Palmprint Recognition.
    Loss: ArcFace 단독 (CenterLoss / TripletLoss 제거)
    """

    def __init__(
        self,
        num_classes:  int,
        embed_dim:    int   = 512,
        arc_margin:   float = 0.3,
        arc_scale:    float = 32.0,
    ):
        super().__init__()
        self.encoder = HybridPalmprintEncoder(embed_dim=embed_dim)
        self.arcface = ArcFaceHead(embed_dim, num_classes,
                                   margin=arc_margin, scale=arc_scale)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor,
                labels:  Optional[torch.Tensor] = None,
                local_w: Optional[torch.Tensor] = None,
                global_w:Optional[torch.Tensor] = None) -> dict:

        feat      = self.encoder(x, local_w, global_w)
        feat_norm = F.normalize(feat, dim=1)

        out = {"embedding": feat_norm}

        if labels is not None:
            logits   = self.arcface(feat_norm, labels)
            arc_loss = F.cross_entropy(logits, labels)
            out.update({
                "logits":     logits,
                "arc_loss":   arc_loss,
                "total_loss": arc_loss,
            })

        return out

    # ------------------------------------------------------------------
    # 추론용 embedding 추출
    # ------------------------------------------------------------------
    def get_embedding(self, x: torch.Tensor,
                      local_w: float = 0.5, global_w: float = 0.5) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            lw = torch.tensor([local_w],  dtype=torch.float32, device=x.device)
            gw = torch.tensor([global_w], dtype=torch.float32, device=x.device)
            return self.forward(x, local_w=lw, global_w=gw)["embedding"]

    # ------------------------------------------------------------------
    # [수정 1] Phase1 encoder 부분 동결
    # freeze_ratio=0.5 → 하위 50% 레이어 동결, 상위 50% fine-tune
    # ------------------------------------------------------------------
    def freeze_encoder_layers(self, freeze_ratio: float = 0.5):
        """
        Phase1 pretrained encoder를 로드한 뒤 호출.
        하위 레이어(저수준 특징)는 동결하고 상위 레이어만 fine-tune.

        Args:
            freeze_ratio: 0.0 = 전체 학습, 1.0 = 전체 동결, 0.5 = 하위 절반 동결
        """
        params   = list(self.encoder.parameters())
        n_freeze = int(len(params) * freeze_ratio)
        frozen, trainable = 0, 0
        for i, p in enumerate(params):
            if i < n_freeze:
                p.requires_grad = False
                frozen += p.numel()
            else:
                p.requires_grad = True
                trainable += p.numel()
        print(
            f"  [Encoder Freeze] ratio={freeze_ratio:.1f} | "
            f"frozen={frozen:,} params | trainable={trainable:,} params"
        )

    def unfreeze_encoder(self):
        """모든 encoder 파라미터를 다시 학습 가능하게."""
        for p in self.encoder.parameters():
            p.requires_grad = True
        print("  [Encoder] 전체 파라미터 학습 가능으로 전환")
