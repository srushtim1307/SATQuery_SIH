"""
Optical + SAR Cross-Modal Fusion Model Calibration & Training Script.

Trains a dual-stream cross-attention neural network (CrossModalOpticalSARNet)
on simulated co-registered Optical multispectral reflectance and SAR C-band backscatter pairs.
Learns complementary feature alignment, spatial cross-modal attention, and synergy mapping.

Outputs:
- backend/models/weights/optical_sar_fusion.pth
- backend/models/weights/optical_sar_training_log.json
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np

# ============================================================================
# Neural Architecture Definition
# ============================================================================

class ConvBlock(nn.Module):
    """Convolution + BatchNorm + GELU block with optional residual projection."""
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(x) + self.shortcut(x))


class SpatialCrossAttention(nn.Module):
    """
    Bidirectional Spatial Cross-Modal Attention.
    Enables Optical queries to query SAR structural keys,
    and SAR queries to query Optical spectral keys.
    """
    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads

        self.q_opt = nn.Conv2d(channels, channels, kernel_size=1)
        self.k_sar = nn.Conv2d(channels, channels, kernel_size=1)
        self.v_sar = nn.Conv2d(channels, channels, kernel_size=1)

        self.q_sar = nn.Conv2d(channels, channels, kernel_size=1)
        self.k_opt = nn.Conv2d(channels, channels, kernel_size=1)
        self.v_opt = nn.Conv2d(channels, channels, kernel_size=1)

        self.out_opt = nn.Conv2d(channels, channels, kernel_size=1)
        self.out_sar = nn.Conv2d(channels, channels, kernel_size=1)
        self.scale = self.head_dim ** -0.5

    def _attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, C, H, W = q.shape
        N = H * W
        q = q.view(B, self.num_heads, self.head_dim, N).permute(0, 1, 3, 2)
        k = k.view(B, self.num_heads, self.head_dim, N).permute(0, 1, 3, 2)
        v = v.view(B, self.num_heads, self.head_dim, N).permute(0, 1, 3, 2)

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_probs = F.softmax(attn_weights, dim=-1)

        out = torch.matmul(attn_probs, v)
        out = out.permute(0, 1, 3, 2).contiguous().view(B, C, H, W)
        return out, attn_probs

    def forward(self, f_opt: torch.Tensor, f_sar: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q_o = self.q_opt(f_opt)
        k_s = self.k_sar(f_sar)
        v_s = self.v_sar(f_sar)
        opt_attended, attn_opt = self._attention(q_o, k_s, v_s)
        opt_attended = self.out_opt(opt_attended)

        q_s = self.q_sar(f_sar)
        k_o = self.k_opt(f_opt)
        v_o = self.v_opt(f_opt)
        sar_attended, attn_sar = self._attention(q_s, k_o, v_o)
        sar_attended = self.out_sar(sar_attended)

        coherence = (attn_opt.max(dim=-1)[0].mean() + attn_sar.max(dim=-1)[0].mean()) * 0.5
        return opt_attended, sar_attended, coherence


class CrossModalOpticalSARNet(nn.Module):
    """
    Dual-stream Cross-Modal Optical + SAR Fusion Network.
    Combines optical spectral features and SAR dielectric backscatter features.
    """
    def __init__(self, feature_dim: int = 64):
        super().__init__()
        # Optical encoder: 3 channels (RGB/multispectral) -> feature_dim
        self.opt_stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            ConvBlock(32, feature_dim, stride=2)
        )

        # SAR encoder: 1 or 3 channels (radar backscatter) -> feature_dim
        self.sar_stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            ConvBlock(32, feature_dim, stride=2)
        )

        # Spatial Cross Attention
        self.cross_attn = SpatialCrossAttention(feature_dim, num_heads=4)

        # Channel Attention & Saliency Gate
        self.saliency_gate = nn.Sequential(
            nn.Conv2d(feature_dim * 4, feature_dim, kernel_size=1),
            nn.BatchNorm2d(feature_dim),
            nn.GELU(),
            nn.Conv2d(feature_dim, 2, kernel_size=1),
            nn.Softmax(dim=1)
        )

        # Fusion decoder: maps fused representations back to image space (3-channel false-color composite)
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            ConvBlock(feature_dim * 2, feature_dim, stride=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            ConvBlock(feature_dim, 32, stride=1),
            nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

        # Complementarity / synergy head: outputs 1-channel synergy map
        self.synergy_head = nn.Sequential(
            nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True),
            nn.Conv2d(2, 16, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x_opt: torch.Tensor, x_sar: torch.Tensor) -> Dict[str, torch.Tensor]:
        if x_sar.shape[1] == 1:
            x_sar = x_sar.repeat(1, 3, 1, 1)

        # 1. Feature extraction
        f_opt = self.opt_stem(x_opt)
        f_sar = self.sar_stem(x_sar)

        # 2. Spatial Cross-Attention
        att_opt, att_sar, coherence = self.cross_attn(f_opt, f_sar)

        # 3. Dynamic Saliency Weighting
        concat_feats = torch.cat([f_opt, att_opt, f_sar, att_sar], dim=1)
        gates = self.saliency_gate(concat_feats)
        w_opt, w_sar = gates[:, 0:1], gates[:, 1:2]

        fused_opt = (f_opt + att_opt) * w_opt
        fused_sar = (f_sar + att_sar) * w_sar
        fused_latent = torch.cat([fused_opt, fused_sar], dim=1)

        # 4. Decoder outputs
        fused_composite = self.decoder(fused_latent)
        synergy_map = self.synergy_head(gates)

        return {
            "fused_composite": fused_composite,
            "synergy_map": synergy_map,
            "coherence": coherence,
            "opt_gate": w_opt,
            "sar_gate": w_sar
        }


# ============================================================================
# Simulated Multi-Sensor Dataset
# ============================================================================

class MultiSensorOpticalSARDataset(Dataset):
    """
    Synthesizes realistic co-registered Optical + SAR pairs:
    1. Optical: Multispectral spectral bands with vegetation chlorophyll, water absorption, urban materials, and cloud haze.
    2. SAR: Microwave backscatter with dielectric specular reflection (water appears dark), surface roughness (vegetation), and corner reflectors (urban appears very bright).
    """
    def __init__(self, num_samples: int = 120, img_size: int = 128):
        self.num_samples = num_samples
        self.img_size = img_size
        self.data = []

        np.random.seed(42)
        for _ in range(num_samples):
            H, W = img_size, img_size
            opt = np.zeros((H, W, 3), dtype=np.float32)
            sar = np.zeros((H, W, 3), dtype=np.float32)

            # 1. Vegetation base
            opt[:, :] = [0.15, 0.45, 0.12]
            sar[:, :] = 0.35

            # 2. Water body
            water_mask = np.zeros((H, W), dtype=bool)
            y_curve = np.sin(np.linspace(0, 3 * np.pi, W)) * 20 + H // 2
            for x in range(W):
                y_center = int(y_curve[x])
                y_min = max(0, y_center - 12)
                y_max = min(H, y_center + 12)
                water_mask[y_min:y_max, x] = True

            opt[water_mask] = [0.05, 0.18, 0.55]
            sar[water_mask] = 0.05

            # 3. Urban clusters
            ux, uy = np.random.randint(10, W - 40), np.random.randint(10, H - 40)
            urban_mask = np.zeros((H, W), dtype=bool)
            urban_mask[uy:uy+25, ux:ux+25] = True
            opt[urban_mask] = [0.65, 0.62, 0.60]
            sar[urban_mask] = 0.92

            # 4. Optional cloud occlusions over optical
            if np.random.rand() > 0.5:
                cx, cy = np.random.randint(10, W - 30), np.random.randint(10, H - 30)
                cloud_mask = np.zeros((H, W), dtype=bool)
                cloud_mask[cy:cy+20, cx:cx+30] = True
                opt[cloud_mask] = [0.90, 0.92, 0.95]

            # Sensor noise
            opt += np.random.normal(0, 0.02, opt.shape).astype(np.float32)
            sar += np.random.normal(0, 0.04, sar.shape).astype(np.float32)

            opt = np.clip(opt, 0.0, 1.0)
            sar = np.clip(sar, 0.0, 1.0)

            target = 0.6 * opt + 0.4 * sar
            target = np.clip(target, 0.0, 1.0)

            self.data.append((
                torch.from_numpy(opt).permute(2, 0, 1),
                torch.from_numpy(sar).permute(2, 0, 1),
                torch.from_numpy(target).permute(2, 0, 1)
            ))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# ============================================================================
# Training Pipeline
# ============================================================================

def train_optical_sar_adaptation():
    print("=================================================================")
    print("SatQuery AI — Phase 5E: Optical + SAR Model Calibration & Training")
    print("=================================================================")

    weights_dir = Path(__file__).resolve().parent.parent / "models" / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    weights_path = weights_dir / "optical_sar_fusion.pth"
    log_path = weights_dir / "optical_sar_training_log.json"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[*] Training device: {device}")

    model = CrossModalOpticalSARNet(feature_dim=64).to(device)
    print(f"[*] Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    dataset = MultiSensorOpticalSARDataset(num_samples=128, img_size=128)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion_recon = nn.L1Loss()
    criterion_ssim_approx = nn.MSELoss()

    epochs = 5
    training_log = {
        "model": "CrossModalOpticalSARNet",
        "device": str(device),
        "parameters": sum(p.numel() for p in model.parameters()),
        "epochs": epochs,
        "history": []
    }

    start_time = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_recon = 0.0
        total_coherence = 0.0

        for opt_imgs, sar_imgs, targets in loader:
            opt_imgs = opt_imgs.to(device)
            sar_imgs = sar_imgs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(opt_imgs, sar_imgs)

            loss_recon = criterion_recon(outputs["fused_composite"], targets)
            loss_mse = criterion_ssim_approx(outputs["fused_composite"], targets)
            loss_coherence = -0.1 * outputs["coherence"]

            loss = loss_recon + 0.5 * loss_mse + loss_coherence
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_recon += loss_recon.item()
            total_coherence += outputs["coherence"].item()

        avg_loss = total_loss / len(loader)
        avg_recon = total_recon / len(loader)
        avg_coherence = total_coherence / len(loader)

        print(f"Epoch [{epoch}/{epochs}] — Loss: {avg_loss:.4f} | Recon L1: {avg_recon:.4f} | Coherence: {avg_coherence:.4f}")
        training_log["history"].append({
            "epoch": epoch,
            "total_loss": round(avg_loss, 4),
            "reconstruction_loss": round(avg_recon, 4),
            "coherence_score": round(avg_coherence, 4)
        })

    elapsed = time.time() - start_time
    training_log["elapsed_seconds"] = round(elapsed, 2)
    training_log["final_loss"] = round(avg_loss, 4)

    # Save weights checkpoint
    torch.save(model.state_dict(), weights_path)
    print(f"[+] Saved calibrated weights to: {weights_path} ({weights_path.stat().st_size / (1024*1024):.2f} MB)")

    # Save training log
    with open(log_path, "w") as f:
        json.dump(training_log, f, indent=2)
    print(f"[+] Saved training log to: {log_path}")
    print("[+] Phase 5E Optical + SAR model training completed successfully.")


if __name__ == "__main__":
    train_optical_sar_adaptation()
