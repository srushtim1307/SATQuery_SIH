"""
Real Neural Change Detection Specialist Model (TinyCD).

Paper: "TINYCD: A (Not So) Deep Learning Model For Change Detection"
Weights: HZDR-FWGEL/UCD-MNCD256-TinyCD (safetensors format)
Architecture: Sliced EfficientNet-B4 + MixingMaskAttentionBlock + UpMask Decoder
"""

import os
import uuid
import logging
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
import numpy as np
from PIL import Image

import torch
import torchvision
from torch import nn, Tensor
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download
from scipy.ndimage import label, find_objects

from app.core.config import PROCESSED_DIR

logger = logging.getLogger(__name__)

HF_REPO_ID = "HZDR-FWGEL/UCD-MNCD256-TinyCD"
HF_FILENAME = "model.safetensors"


# ============================================================================
# TinyCD Architecture Definition
# ============================================================================

class PixelwiseLinear(nn.Module):
    """Pointwise 1x1 convolution with non-linear activation."""
    def __init__(self, fin: List[int], fout: List[int], last_activation: Optional[nn.Module] = None):
        super().__init__()
        assert len(fout) == len(fin)
        n = len(fin)
        self._linears = nn.Sequential(*[
            nn.Sequential(
                nn.Conv2d(fin[i], fout[i], kernel_size=1, bias=True),
                nn.PReLU() if (i < n - 1 or last_activation is None) else last_activation,
            )
            for i in range(n)
        ])

    def forward(self, x: Tensor) -> Tensor:
        return self._linears(x)


class MixingBlock(nn.Module):
    """Depthwise-separable mixing convolution for bi-temporal channel interleave."""
    def __init__(self, ch_in: int, ch_out: int):
        super().__init__()
        self._convmix = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, groups=ch_out, padding=1),
            nn.PReLU(),
            nn.InstanceNorm2d(ch_out),
        )

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        mixed = torch.stack((x, y), dim=2)
        mixed = torch.reshape(mixed, (x.shape[0], -1, x.shape[2], x.shape[3]))
        return self._convmix(mixed)


class MixingMaskAttentionBlock(nn.Module):
    """Space-semantic attention mechanism merging embeddings from Siamese passes."""
    def __init__(self, ch_in: int, ch_out: int, fin: List[int], fout: List[int], generate_masked: bool = False):
        super().__init__()
        self._mixing = MixingBlock(ch_in, ch_out)
        self._linear = PixelwiseLinear(fin, fout)
        self._final_normalization = nn.InstanceNorm2d(ch_out) if generate_masked else None
        self._mixing_out = MixingBlock(ch_in, ch_out) if generate_masked else None

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        z_mix = self._mixing(x, y)
        z = self._linear(z_mix)
        z_mix_out = 0 if self._mixing_out is None else self._mixing_out(x, y)
        return z if self._final_normalization is None else self._final_normalization(z_mix_out * z)


class UpMask(nn.Module):
    """Upsampling block with skip connections and residual convolutions."""
    def __init__(self, scale_factor: float, nin: int, nout: int):
        super().__init__()
        self._upsample = nn.Upsample(scale_factor=scale_factor, mode="bilinear", align_corners=True)
        self._convolution = nn.Sequential(
            nn.Conv2d(nin, nin, kernel_size=3, stride=1, groups=nin, padding=1),
            nn.PReLU(),
            nn.InstanceNorm2d(nin),
            nn.Conv2d(nin, nout, kernel_size=1, stride=1),
            nn.PReLU(),
            nn.InstanceNorm2d(nout),
        )

    def forward(self, x: Tensor, y: Optional[Tensor] = None) -> Tensor:
        x = self._upsample(x)
        if y is not None:
            x = x * y
        return self._convolution(x)


def _get_sliced_backbone(bkbn_name: str = "efficientnet_b4", output_layer_bkbn: str = "3") -> nn.ModuleList:
    """Extracts features up to layer 3 of EfficientNet-B4 without downloading ImageNet classifier."""
    entire_model = getattr(torchvision.models, bkbn_name)(weights=None).features
    derived_model = nn.ModuleList([])
    for name, layer in entire_model.named_children():
        derived_model.append(layer)
        if name == output_layer_bkbn:
            break
    return derived_model


class TinyCD(nn.Module):
    """
    Complete TinyCD Network.
    Computes dense binary change probability map between two co-registered rasters.
    """
    def __init__(self, bkbn_name: str = "efficientnet_b4", output_layer_bkbn: str = "3"):
        super().__init__()
        self._backbone = _get_sliced_backbone(bkbn_name, output_layer_bkbn)

        # Initial mixing layer on raw inputs (3 channels each -> 6 channels interleaved)
        self._first_mix = MixingMaskAttentionBlock(6, 3, [3, 10, 5], [10, 5, 1])

        # Intermediate attention blocks across backbone hierarchies
        self._mixing_mask = nn.ModuleList([
            MixingMaskAttentionBlock(48, 24, [24, 12, 6], [12, 6, 1]),
            MixingMaskAttentionBlock(64, 32, [32, 16, 8], [16, 8, 1]),
            MixingBlock(112, 56),
        ])

        # Progressive decoder upsampling
        self._up = nn.ModuleList([
            UpMask(2, 56, 64),
            UpMask(2, 64, 64),
            UpMask(2, 64, 32),
        ])

        # Final pixelwise classification (outputs 1-channel probability in [0, 1])
        self._classify = PixelwiseLinear([32, 16, 8], [16, 8, 1], nn.Sigmoid())

    def forward(self, ref: Tensor, test: Tensor) -> Tensor:
        features = [self._first_mix(ref, test)]
        for num, layer in enumerate(self._backbone):
            ref, test = layer(ref), layer(test)
            if num != 0:
                features.append(self._mixing_mask[num - 1](ref, test))

        upping = features[-1]
        for i, j in enumerate(range(-2, -5, -1)):
            upping = self._up[i](upping, features[j])

        return self._classify(upping)


# ============================================================================
# Model Loading & Weights Management
# ============================================================================

def load_tinycd_model(device: str = "cpu") -> TinyCD:
    """
    Downloads and caches pre-trained TinyCD weights from Hugging Face Hub,
    instantiates the architecture, and prepares it for inference.
    """
    logger.info(f"Loading TinyCD model from {HF_REPO_ID} on device={device}...")
    weights_path = hf_hub_download(repo_id=HF_REPO_ID, filename=HF_FILENAME)
    raw_weights = load_file(weights_path)
    
    # Strip CD_model. prefix if present in the checkpoint
    clean_weights = {k.replace("CD_model.", ""): v for k, v in raw_weights.items()}

    model = TinyCD()
    model.load_state_dict(clean_weights, strict=False)
    model.to(device)
    model.eval()
    logger.info("TinyCD model successfully loaded and ready for evaluation.")
    return model


# ============================================================================
# Inference & Analysis Engine
# ============================================================================

def preprocess_image_pair(
    t1_path: str,
    t2_path: str,
    target_size: Tuple[int, int] = (256, 256),
    device: str = "cpu"
) -> Tuple[Tensor, Tensor, Image.Image, Image.Image]:
    """
    Opens T1 and T2 images from disk, converts to RGB, resizes, and produces normalized tensors.
    """
    img1 = Image.open(t1_path).convert("RGB")
    img2 = Image.open(t2_path).convert("RGB")

    resized_1 = img1.resize(target_size, Image.Resampling.BILINEAR)
    resized_2 = img2.resize(target_size, Image.Resampling.BILINEAR)

    # Convert to float array in [0, 1]
    arr1 = np.array(resized_1, dtype=np.float32) / 255.0
    arr2 = np.array(resized_2, dtype=np.float32) / 255.0

    # Standard ImageNet mean and std
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    norm1 = (arr1 - mean) / std
    norm2 = (arr2 - mean) / std

    # Transpose to (C, H, W) and add batch dimension
    t1_tensor = torch.from_numpy(norm1.transpose(2, 0, 1)).unsqueeze(0).to(device)
    t2_tensor = torch.from_numpy(norm2.transpose(2, 0, 1)).unsqueeze(0).to(device)

    return t1_tensor, t2_tensor, resized_1, resized_2


def generate_change_artifacts(
    prob_map: np.ndarray,
    mask: np.ndarray,
    t2_image: Image.Image,
    prefix_id: str
) -> Tuple[str, str, str, str]:
    """
    Generates binary mask and overlay visualization images, saving them into PROCESSED_DIR.
    Returns:
        (mask_rel_url, overlay_rel_url, mask_disk_path, overlay_disk_path)
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Binary Mask (Grayscale 0/255)
    mask_arr = (mask * 255).astype(np.uint8)
    mask_img = Image.fromarray(mask_arr, mode="L")
    mask_filename = f"change_mask_{prefix_id}.png"
    mask_disk_path = str(PROCESSED_DIR / mask_filename)
    mask_img.save(mask_disk_path, format="PNG")

    # 2. Change Overlay: Blends rose/crimson highlight onto T2 image
    t2_arr = np.array(t2_image).astype(np.float32)
    overlay_arr = t2_arr.copy()
    
    # Highlight changed pixels with vivid coral-red (#F43F5E -> [244, 63, 94])
    changed_mask = mask > 0
    highlight_color = np.array([244, 63, 94], dtype=np.float32)
    alpha = 0.55
    overlay_arr[changed_mask] = (1.0 - alpha) * overlay_arr[changed_mask] + alpha * highlight_color
    
    overlay_img = Image.fromarray(np.clip(overlay_arr, 0, 255).astype(np.uint8), mode="RGB")
    overlay_filename = f"change_overlay_{prefix_id}.png"
    overlay_disk_path = str(PROCESSED_DIR / overlay_filename)
    overlay_img.save(overlay_disk_path, format="PNG")

    mask_rel_url = f"/static/processed/{mask_filename}"
    overlay_rel_url = f"/static/processed/{overlay_filename}"

    return mask_rel_url, overlay_rel_url, mask_disk_path, overlay_disk_path


def extract_change_bounding_boxes(
    mask: np.ndarray,
    prob_map: np.ndarray,
    min_area_pixels: int = 16
) -> List[Dict[str, Any]]:
    """
    Finds contiguous changed regions and returns normalized bounding boxes.
    """
    labeled_mask, num_features = label(mask)
    if num_features == 0:
        return []

    slices = find_objects(labeled_mask)
    boxes = []
    box_counter = 1

    h, w = mask.shape

    for i, slice_tuple in enumerate(slices):
        if slice_tuple is None:
            continue
        y_slice, x_slice = slice_tuple
        comp_mask = labeled_mask[y_slice, x_slice] == (i + 1)
        area = int(np.sum(comp_mask))
        if area < min_area_pixels:
            continue

        top = round(float(y_slice.start / h) * 100, 1)
        left = round(float(x_slice.start / w) * 100, 1)
        width = round(float((x_slice.stop - x_slice.start) / w) * 100, 1)
        height = round(float((y_slice.stop - y_slice.start) / h) * 100, 1)

        comp_probs = prob_map[y_slice, x_slice][comp_mask]
        conf = round(float(np.mean(comp_probs)), 3) if len(comp_probs) > 0 else 0.5

        boxes.append({
            "id": f"box-change-{box_counter}",
            "label": f"Change Cluster #{box_counter}",
            "top": top,
            "left": left,
            "width": width,
            "height": height,
            "confidence": conf,
            "pixel_area": area,
            "is_demo": False
        })
        box_counter += 1

    # Sort boxes by largest area first
    boxes.sort(key=lambda b: b["pixel_area"], reverse=True)
    return boxes


def run_tinycd_inference(
    model: TinyCD,
    t1_path: str,
    t2_path: str,
    device: str = "cpu",
    threshold: float = 0.35
) -> Dict[str, Any]:
    """
    Runs full neural change detection pipeline on a pair of images.
    Returns real metrics, change mask, overlay, and bounding coordinates.
    """
    t1_tensor, t2_tensor, img1_resized, img2_resized = preprocess_image_pair(
        t1_path, t2_path, target_size=(256, 256), device=device
    )

    with torch.no_grad():
        out_tensor = model(t1_tensor, t2_tensor)  # Shape: (1, 1, 256, 256)

    prob_map = out_tensor.squeeze().cpu().numpy().astype(np.float32)
    binary_mask = (prob_map >= threshold).astype(np.uint8)

    total_pixels = 256 * 256
    changed_pixels = int(np.sum(binary_mask > 0))
    change_percentage = round(float((changed_pixels / total_pixels) * 100), 2)

    # Real confidence computed from probability distribution
    if changed_pixels > 0:
        confidence = round(float(np.mean(prob_map[binary_mask > 0])), 4)
    else:
        # High confidence that no significant change occurred
        confidence = round(float(1.0 - np.mean(prob_map)), 4)

    # Generate visual artifacts
    run_id = uuid.uuid4().hex[:8]
    mask_url, overlay_url, mask_path, overlay_path = generate_change_artifacts(
        prob_map=prob_map,
        mask=binary_mask,
        t2_image=img2_resized,
        prefix_id=run_id
    )

    # Extract bounding boxes
    bounding_boxes = extract_change_bounding_boxes(binary_mask, prob_map)

    return {
        "change_percentage": change_percentage,
        "changed_pixels": changed_pixels,
        "total_pixels": total_pixels,
        "confidence": confidence,
        "mask_url": mask_url,
        "overlay_url": overlay_url,
        "mask_path": mask_path,
        "overlay_path": overlay_path,
        "bounding_boxes": bounding_boxes,
        "model_name": "TinyCD (EfficientNet-B4 + MAMB)",
        "backend": "real"
    }
