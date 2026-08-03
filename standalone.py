#!/usr/bin/env python3
"""
Standalone Evaluation Script for Semiconductor Image Restoration
Location: C:\\Users\\MOKSH\\OneDrive\\Desktop\\ai image enhancer\\eval.py

Accepts:
  - (a) Path to test images directory (--input_dir / -i)
  - (b) Path to output directory (--output_dir / -o)
  - (c) Optional path to weights checkpoint (--weights / -w)

Usage:
  python eval.py --input_dir /path/to/test_images --output_dir /path/to/restored_output
"""

import os
import sys
import glob
import time
import json
import argparse
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# 1. Model Architecture (NAFNetGrayscale with Sub-Pixel Convolution)
# ─────────────────────────────────────────────────────────────────────────────

class LayerNorm2d(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        var = (x - mean).pow(2).mean(dim=1, keepdim=True)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return x_norm * self.weight + self.bias


class SimpleGate(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class SimpleChannelAttention(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn = self.fc(self.pool(x))
        return x * attn


class NAFBlock(nn.Module):
    def __init__(self, c: int, dw_expand: int = 2, ffn_expand: int = 2):
        super().__init__()
        dw_channel = c * dw_expand
        self.conv1 = nn.Conv2d(c, dw_channel, kernel_size=1, padding=0, stride=1, bias=True)
        self.conv2 = nn.Conv2d(dw_channel, dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel, bias=True)
        self.sg1 = SimpleGate()
        self.sca = SimpleChannelAttention(dw_channel // 2)
        self.conv3 = nn.Conv2d(dw_channel // 2, c, kernel_size=1, padding=0, stride=1, bias=True)
        self.norm1 = LayerNorm2d(c)

        ffn_channel = c * ffn_expand
        self.conv4 = nn.Conv2d(c, ffn_channel, kernel_size=1, padding=0, stride=1, bias=True)
        self.sg2 = SimpleGate()
        self.conv5 = nn.Conv2d(ffn_channel // 2, c, kernel_size=1, padding=0, stride=1, bias=True)
        self.norm2 = LayerNorm2d(c)

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm1(x)
        y = self.conv1(y)
        y = self.conv2(y)
        y = self.sg1(y)
        y = self.sca(y)
        y = self.conv3(y)
        x = x + y * self.beta

        y = self.norm2(x)
        y = self.conv4(y)
        y = self.sg2(y)
        y = self.conv5(y)
        return x + y * self.gamma


class NAFNetGrayscale(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 1, width: int = 128, 
                 enc_blk_nums: list = [2, 2], dec_blk_nums: list = [2, 2], scale: int = 2):
        super().__init__()
        self.scale = scale
        self.intro = nn.Conv2d(in_channels, width, kernel_size=3, padding=1, stride=1, bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.upsamplers = nn.ModuleList()
        self.downsamplers = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))
            self.downsamplers.append(nn.Conv2d(chan, chan * 2, kernel_size=2, stride=2))
            chan = chan * 2

        self.middle_blks = nn.Sequential(*[NAFBlock(chan) for _ in range(2)])

        for num in dec_blk_nums:
            self.upsamplers.append(nn.ConvTranspose2d(chan, chan // 2, kernel_size=2, stride=2))
            chan = chan // 2
            self.decoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))

        self.sr_head = nn.Sequential(
            nn.Conv2d(width, width * (scale ** 2), kernel_size=3, padding=1, bias=True),
            nn.PixelShuffle(scale),
            NAFBlock(width),
            nn.Conv2d(width, out_channels, kernel_size=3, padding=1, bias=True)
        )

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        orig_h, orig_w = inp.shape[2], inp.shape[3]
        pad_h = (4 - orig_h % 4) % 4
        pad_w = (4 - orig_w % 4) % 4
        inp_padded = F.pad(inp, (0, pad_w, 0, pad_h), mode='reflect') if (pad_h > 0 or pad_w > 0) else inp

        inp_rescale = F.interpolate(inp_padded, scale_factor=self.scale, mode='bicubic', align_corners=False)
        x = self.intro(inp_padded)
        
        skips = []
        for encoder, downsampler in zip(self.encoders, self.downsamplers):
            x = encoder(x)
            skips.append(x)
            x = downsampler(x)

        x = self.middle_blks(x)

        for decoder, upsampler, skip in zip(self.decoders, self.upsamplers, reversed(skips)):
            x = upsampler(x)
            x = x + skip
            x = decoder(x)

        out = torch.clamp(self.sr_head(x) + inp_rescale, 0.0, 1.0)
        target_h, target_w = orig_h * self.scale, orig_w * self.scale
        return out[:, :, :target_h, :target_w]


# ─────────────────────────────────────────────────────────────────────────────
# 2. Image IO & Pre/Post-Processing Utilities
# ─────────────────────────────────────────────────────────────────────────────

def normalize_intensity(img_np: np.ndarray) -> np.ndarray:
    img = img_np.astype(np.float32)
    min_val, max_val = img.min(), img.max()
    if max_val > min_val:
        return (img - min_val) / (max_val - min_val)
    return np.zeros_like(img, dtype=np.float32)


def load_image_as_tensor(file_path: str) -> torch.Tensor:
    if file_path.endswith('.npy'):
        raw = np.load(file_path, allow_pickle=True).astype(np.float32)
        if raw.ndim == 3:
            raw = raw.squeeze()
        norm = normalize_intensity(raw)
    else:
        img_bgr = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if img_bgr is None:
            raise ValueError(f"Unable to read image file: {file_path}")
        norm = img_bgr.astype(np.float32) / 255.0

    tensor = torch.from_numpy(norm).unsqueeze(0).unsqueeze(0).float()
    return tensor


def tensor_to_uint8(tensor: torch.Tensor) -> np.ndarray:
    arr = tensor.squeeze().detach().cpu().numpy()
    arr = np.clip(arr, 0.0, 1.0)
    return (arr * 255.0).astype(np.uint8)


def _soft_highlight_compression(img_float: np.ndarray, knee: float = 0.82) -> np.ndarray:
    out = img_float.copy()
    mask = out > knee
    if np.any(mask):
        excess = out[mask] - knee
        scale = 1.0 - knee
        out[mask] = knee + scale * np.tanh(excess / scale)
    return out


def _halo_free_edge_enhance(img_uint8: np.ndarray, gain: float = 0.12) -> np.ndarray:
    blurred = cv2.GaussianBlur(img_uint8, (3, 3), 0.8)
    detail = img_uint8.astype(np.float32) - blurred.astype(np.float32)

    sobelx = cv2.Sobel(img_uint8, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img_uint8, cv2.CV_32F, 0, 1, ksize=3)
    edge_mag = np.sqrt(sobelx**2 + sobely**2) / 255.0
    edge_weight = 1.0 / (1.0 + 5.0 * edge_mag)

    enhanced = img_uint8.astype(np.float32) + gain * detail * edge_weight
    return np.clip(enhanced, 0, 255).astype(np.uint8)


def balanced_post_process(restored_uint8: np.ndarray) -> np.ndarray:
    img_f = restored_uint8.astype(np.float32) / 255.0
    compressed_f = _soft_highlight_compression(img_f, knee=0.82)
    compressed_u8 = (compressed_f * 255.0).astype(np.uint8)

    enhanced_u8 = _halo_free_edge_enhance(compressed_u8, gain=0.10)
    return enhanced_u8


def remap_checkpoint_keys(state_dict: dict) -> dict:
    mapping = [
        ('enc.',         'encoders.'),
        ('dec.',         'decoders.'),
        ('up.',          'upsamplers.'),
        ('down.',        'downsamplers.'),
        ('.mid.',        '.middle_blks.'),
        ('mid.',         'middle_blks.'),
        ('head.',        'sr_head.'),
        ('.c1.',         '.conv1.'),
        ('.c2.',         '.conv2.'),
        ('.c3.',         '.conv3.'),
        ('.c4.',         '.conv4.'),
        ('.c5.',         '.conv5.'),
        ('.g1',          '.sg1'),
        ('.g2',          '.sg2'),
        ('.n1.w',        '.norm1.weight'),
        ('.n1.b',        '.norm1.bias'),
        ('.n2.w',        '.norm2.weight'),
        ('.n2.b',        '.norm2.bias'),
        ('.sca.f.',      '.sca.fc.'),
    ]
    new_sd = {}
    for k, v in state_dict.items():
        new_k = k
        for old, new in mapping:
            new_k = new_k.replace(old, new)
        new_sd[new_k] = v
    return new_sd


# ─────────────────────────────────────────────────────────────────────────────
# 3. Main Standalone Evaluation Entrypoint
# ─────────────────────────────────────────────────────────────────────────────

def run_evaluation(input_dir: str, output_dir: str, weights_path: str = None):
    input_dir = input_dir.strip().strip('"\'')
    output_dir = output_dir.strip().strip('"\'')
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[EVAL] Running evaluation on device: {device}")

    # Automatic weights resolution
    candidate_weights = [
        weights_path if weights_path else "",
        os.path.join(os.path.dirname(__file__), "weights", "best_model.pth"),
        os.path.join(os.path.dirname(__file__), "best_model.pth"),
        r"c:\Users\MOKSH\OneDrive\Desktop\semicon hackathon\ml_backend\weights\best_model.pth",
    ]

    resolved_weights = None
    for cand in candidate_weights:
        if cand and os.path.exists(cand):
            resolved_weights = cand
            break

    # Build model (width=128 first, width=32 fallback)
    model = NAFNetGrayscale(in_channels=1, out_channels=1, width=128, scale=2).to(device)

    if resolved_weights:
        print(f"[EVAL] Loading trained weights from: {resolved_weights}")
        raw_sd = torch.load(resolved_weights, map_location=device)
        try:
            try:
                model.load_state_dict(raw_sd)
            except RuntimeError:
                model.load_state_dict(remap_checkpoint_keys(raw_sd))
            print("[EVAL] Loaded model (width=128) successfully.")
        except Exception as e:
            print(f"[EVAL WARNING] width=128 load failed ({e}). Trying width=32 fallback...")
            model = NAFNetGrayscale(in_channels=1, out_channels=1, width=32, scale=2).to(device)
            try:
                model.load_state_dict(raw_sd)
            except RuntimeError:
                model.load_state_dict(remap_checkpoint_keys(raw_sd))
            print("[EVAL] Loaded model (width=32 fallback) successfully.")
    else:
        print("[WARNING] No weights file found. Operating with initialized model weights.")

    model.eval()

    # Find degraded test images (.npy, .png, .jpg, .jpeg)
    input_files = sorted(
        glob.glob(os.path.join(input_dir, "*.npy")) +
        glob.glob(os.path.join(input_dir, "*.png")) +
        glob.glob(os.path.join(input_dir, "*.jpg")) +
        glob.glob(os.path.join(input_dir, "*.jpeg"))
    )

    if not input_files:
        print(f"[ERROR] No image or .npy files found in input directory: {input_dir}")
        sys.exit(1)

    print(f"[EVAL] Processing {len(input_files)} degraded test samples...")
    results = []
    total_time = 0.0

    with torch.no_grad():
        for file_path in input_files:
            filename = os.path.basename(file_path)
            png_filename = filename.rsplit('.', 1)[0] + ".png"
            output_file_path = os.path.join(output_dir, png_filename)

            inp_tensor = load_image_as_tensor(file_path).to(device)

            start_t = time.perf_counter()
            out_tensor = model(inp_tensor)
            end_t = time.perf_counter()

            infer_ms = (end_t - start_t) * 1000.0
            total_time += infer_ms

            restored_np = tensor_to_uint8(out_tensor)
            final_np = balanced_post_process(restored_np)
            cv2.imwrite(output_file_path, final_np)

            results.append({
                "file": filename,
                "input_path": file_path,
                "output_path": output_file_path,
                "inference_time_ms": round(infer_ms, 2)
            })

    avg_time_ms = total_time / len(input_files)
    summary = {
        "total_images": len(input_files),
        "avg_inference_time_ms": round(avg_time_ms, 2),
        "device": str(device),
        "results": results
    }

    summary_file = os.path.join(output_dir, "evaluation_summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[EVAL COMPLETE] Processed {len(input_files)} images in {round(total_time/1000.0, 2)}s (Avg: {round(avg_time_ms, 2)} ms/image)")
    print(f"[EVAL COMPLETE] Summary report written to: {summary_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standalone Evaluation Script for AI Image Restoration")
    parser.add_argument("--input_dir", "-i", type=str, default=None, help="Path to test images directory")
    parser.add_argument("--output_dir", "-o", type=str, default=None, help="Path to output directory")
    parser.add_argument("--weights", "-w", type=str, default=None, help="Optional path to trained model weights checkpoint")
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir

    if not input_dir:
        input_dir = input("Enter the location of the INPUT folder: ").strip().strip('"\'')
    if not output_dir:
        output_dir = input("Enter the location of the OUTPUT folder: ").strip().strip('"\'')

    run_evaluation(input_dir, output_dir, args.weights)
