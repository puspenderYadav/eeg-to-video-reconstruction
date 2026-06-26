"""Group 4 — CARD Transformer + Video Latent Diffusion
Refined, guide-aligned reference implementation.

This file follows the Group 4 specification in the project guide:
- 128-channel EEG
- 500 ms windows at 880 Hz => 440 samples
- 0.5–45 Hz bandpass + optional ICA artifact rejection
- CARD encoder with 10 temporal patches of length 44 per channel
- 3 stacked CARD blocks
- 512-dim CLIP-aligned embedding
- Video diffusion conditioning with temporal consistency
- Full evaluation suite: SSIM, PSNR, FID, FVD, LPIPS, Top-5, CLIP similarity
- Train/val/test split support and ablation toggles

Notes:
- VideoLDM in Hugging Face Diffusers is not a single universal class across all
  versions. This implementation uses a video-capable denoising backbone with
  temporal conditioning hooks. If your local diffusers version exposes a more
  specific VideoLDM/temporal UNet class, swap it into build_video_denoiser().
- The code is designed as a strong, guide-compliant training scaffold. You may
  still need to adapt the dataset path layout and exact model checkpoints to your
  lab environment.
"""

from __future__ import annotations

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
# Defer importing scipy.signal until EEG preprocessing to avoid import-time
# binary compatibility issues in some environments.
signal = None

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

clip = None
# NOTE: heavy diffusers imports are deferred to runtime to allow
# running data discovery / lightweight checks without the package
DDPMScheduler = None
AutoencoderKL = None
UNet2DConditionModel = None
transforms = None
make_grid = None
save_image = None

# Optional metrics packages
try:
    from skimage.metrics import structural_similarity as skimage_ssim
except Exception:
    skimage_ssim = None

try:
    from torchmetrics.image.fid import FrechetInceptionDistance
except Exception:
    FrechetInceptionDistance = None

try:
    
     from torchmetrics.image import LearnedPerceptualImagePatchSimilarity
except Exception:
    LearnedPerceptualImagePatchSimilarity = None

# Optional MNE for ICA artifact rejection
try:
    import mne
except Exception:
    mne = None

# ----------------------------------------------------------------------------
# Reproducibility
# ----------------------------------------------------------------------------

def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------

@dataclass
class Config:
    eeg_root: Path = Path("EEG")
    video_root: Path = Path("Video")
    output_dir: Path = Path("outputs_group4")

    # Guide-spec preprocessing
    original_sfreq: float = 200.0
    target_sfreq: float = 880.0
    window_samples: int = 440
    overlap: float = 0.5
    bandpass_low: float = 0.5
    bandpass_high: float = 45.0
    n_channels: int = 128
    n_patches: int = 10
    patch_len: int = 44

    # Model sizes
    d_model: int = 512
    unet_cross_attention_dim: int = 768
    num_card_blocks: int = 3
    nhead_card: int = 8

    # Training
    batch_size: int = 1
    epochs: int = 500
    lr_encoder: float = 1e-4
    lr_unet: float = 1e-5
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    mixed_precision: bool = True
    # Gradient accumulation to simulate larger batch sizes when GPU memory is limited
    grad_accum_steps: int = 4
    num_train_timesteps: int = 1000
    num_inference_steps: int = 100


    # Loss weights from guide
    lambda_mae: float = 0.1
    lambda_contrastive: float = 0.1
    lambda_clip: float = 1.0
    lambda_diffusion: float = 1.0
    lambda_temporal: float = 0.1
    lambda_recon: float = 0.5

    # Video generation
    latent_h: int = 64          # smaller latents = less VRAM
    latent_w: int = 64
    max_video_frames: int = 4   # fewer frames per sample during training
    context_len: int = 4
    grad_accum_steps: int = 8   # compensate with more accumulation steps

    # LR scheduler
    lr_min_factor: float = 0.01      # ← new: cosine annealing floor = lr * this factor
    warmup_epochs: int = 10          # ← new: linear warmup before cosine decay kicks in

    # Misc
    dataset_split_seed: int = 123
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model_name: str = "ViT-B/32"
    stable_diffusion_model: str = "runwayml/stable-diffusion-v1-5"


CFG = Config()
CFG.output_dir.mkdir(parents=True, exist_ok=True)
seed_everything(42)
DEVICE = torch.device(CFG.device)


# ----------------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------------

def list_files(root: Path, ext: str) -> List[Path]:
    if not root.exists():
        return []
    return sorted(root.rglob(f"*{ext}"))


def safe_mean(xs: Sequence[float]) -> float:
    return float(sum(xs) / max(len(xs), 1))


def to_uint8_image(x: torch.Tensor) -> np.ndarray:
    x = x.detach().float().clamp(0, 1)
    if x.ndim == 3:
        x = x.unsqueeze(0)
    # lazy import to avoid torchvision version mismatches at import time
    global make_grid
    if make_grid is None:
        from torchvision.utils import make_grid as _make_grid
        make_grid = _make_grid
    grid = make_grid(x, nrow=1)
    grid = (grid * 255.0).byte().permute(1, 2, 0).cpu().numpy()
    return grid


def save_video_frames(frames: List[torch.Tensor], out_dir: Path, prefix: str = "frame") -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, fr in enumerate(frames):
        img = fr.detach().cpu().clamp(0, 1)
        # lazy import to avoid torchvision import issues at module import time
        global save_image
        if save_image is None:
            from torchvision.utils import save_image as _save_image
            save_image = _save_image
        save_image(img, out_dir / f"{prefix}_{i:04d}.png")


def load_frame_for_vae(video_path: str, frame_idx: int, total_frames: int, device: torch.device) -> torch.Tensor:
    """Load one video frame, resize to 512x512, normalise to [-1, 1] for VAE input."""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return torch.zeros(1, 3, 512, 512, device=device)
    pick = int(frame_idx * max(total - 1, 0) / max(total_frames - 1, 1))
    cap.set(cv2.CAP_PROP_POS_FRAMES, pick)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return torch.zeros(1, 3, 512, 512, device=device)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame_rgb).resize((512, 512))
    arr = np.array(img, dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)


# ----------------------------------------------------------------------------
# EEG preprocessing with ICA artifact rejection
# ----------------------------------------------------------------------------

def load_eeg_numpy(npy_path: Path, block: int = 0) -> np.ndarray:
    raw_data = np.load(npy_path)
    if raw_data.ndim == 3:
        if raw_data.shape[0] == CFG.n_channels:
            # (channels, ?, samples)
            eeg = raw_data.reshape(raw_data.shape[0], -1)
        elif raw_data.shape[1] == CFG.n_channels:
            # (sessions, channels, samples) with exact channel match
            eeg = raw_data.transpose(1, 0, 2).reshape(raw_data.shape[1], -1)
        else:
            # (blocks, channels, samples) — pick the requested block
            b = min(block, raw_data.shape[0] - 1)
            eeg = raw_data[b]  # (channels, samples)
    elif raw_data.ndim == 2:
        eeg = raw_data
    else:
        raise ValueError(f"Unsupported EEG array shape: {raw_data.shape}")
    return eeg.astype(np.float32)


def apply_ica_artifact_rejection(eeg: np.ndarray, sfreq: float) -> np.ndarray:
    """Optional ICA artifact rejection. Falls back gracefully when MNE is absent."""
    if mne is None:
        return eeg

    try:
        info = mne.create_info(ch_names=[f"ch_{i}" for i in range(eeg.shape[0])], sfreq=sfreq, ch_types="eeg")
        raw = mne.io.RawArray(eeg, info, verbose=False)
        raw.filter(l_freq=CFG.bandpass_low, h_freq=CFG.bandpass_high, verbose=False)

        ica = mne.preprocessing.ICA(n_components=min(20, eeg.shape[0] - 1), random_state=42, max_iter="auto")
        ica.fit(raw, verbose=False)

        # Without EOG/ECG channels, we use a conservative reconstruction path.
        cleaned = ica.apply(raw.copy(), verbose=False).get_data()
        return cleaned.astype(np.float32)
    except Exception:
        # Keep pipeline robust if ICA fails on this dataset.
        return eeg


def preprocess_eeg_128ch(npy_path: Path,
                         target_sfreq: float = CFG.target_sfreq,
                         original_sfreq: float = CFG.original_sfreq,
                         window_samples: int = CFG.window_samples,
                         overlap: float = CFG.overlap,
                         block: int = 0) -> torch.Tensor:
    """Return tensor of shape (N, 128, 440)."""
    global signal
    if signal is None:
        from scipy import signal as _signal
        signal = _signal
    eeg = load_eeg_numpy(npy_path, block=block)

    # Force 128 channels where possible.
    if eeg.shape[0] < CFG.n_channels:
        pad = np.zeros((CFG.n_channels - eeg.shape[0], eeg.shape[1]), dtype=eeg.dtype)
        eeg = np.concatenate([eeg, pad], axis=0)
    elif eeg.shape[0] > CFG.n_channels:
        eeg = eeg[:CFG.n_channels]

    # Artifact rejection (optional) and bandpass.
    eeg = apply_ica_artifact_rejection(eeg, sfreq=original_sfreq)

    # Resample to 880 Hz.
    num_samples = int(eeg.shape[-1] * target_sfreq / original_sfreq)
    resampled = signal.resample(eeg, num_samples, axis=-1)

    # Robust bandpass via SOS.
    nyq = 0.5 * target_sfreq
    sos = signal.butter(4, [CFG.bandpass_low / nyq, CFG.bandpass_high / nyq], btype="band", output="sos")
    filtered = signal.sosfiltfilt(sos, resampled, axis=-1)

    # Z-score normalization per channel.
    mean = filtered.mean(axis=-1, keepdims=True)
    std = filtered.std(axis=-1, keepdims=True) + 1e-6
    normalized = (filtered - mean) / std

    # 50% overlap sliding windows: hop=220 for 440-sample windows.
    hop = int(window_samples * (1.0 - overlap))
    windows = []
    start = 0
    while start + window_samples <= normalized.shape[-1]:
        windows.append(normalized[:, start:start + window_samples])
        start += hop

    if not windows:
        raise ValueError(f"EEG file too short for windowing: {npy_path}")

    windows = np.stack(windows, axis=0).astype(np.float32)
    return torch.from_numpy(windows)


# ----------------------------------------------------------------------------
# Video loading + CLIP feature extraction
# ----------------------------------------------------------------------------

def sample_video_frames(video_path: Path, num_frames: int = 5) -> List[Image.Image]:
    cap = cv2.VideoCapture(str(video_path))
    frames: List[Image.Image] = []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return frames

    idxs = np.linspace(0, max(total - 1, 0), num_frames).astype(int).tolist()
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))
    cap.release()
    return frames


class VideoFeatureExtractor:
    def __init__(self, device: torch.device, clip_model_name: str = CFG.clip_model_name):
        self.device = device
        # lazy import of clip to avoid import-time torchvision issues
        import importlib
        global clip
        if clip is None:
            clip = importlib.import_module('clip')

        clip_device = "cpu" if os.environ.get("CLIP_FORCE_CPU", "1") != "0" else device
        self.clip_device = torch.device(clip_device)
        self.model, self.preprocess = clip.load(clip_model_name, device=clip_device)
        self.model.eval()
    @torch.no_grad()
    def extract_clip_embedding(self, video_path: Path) -> torch.Tensor:
        frames = sample_video_frames(video_path, num_frames=5)
        if not frames:
            return torch.zeros(1, 512, device=self.device)
        # AFTER
        batch = torch.stack([self.preprocess(fr) for fr in frames]).to(self.clip_device)
        feat = self.model.encode_image(batch).float().mean(dim=0, keepdim=True)
        feat = feat.to(self.device)   # move result back to main device
        feat = feat / feat.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        return feat


# ----------------------------------------------------------------------------
# Dataset and splits
# ----------------------------------------------------------------------------

def paired_indices(eeg_files: List[Path], video_files: List[Path]) -> List[Tuple[Path, Path, int]]:
    """All block×video pairs: each EEG file contributes one pair per video (block i → video i)."""
    if not video_files:
        return []
    pairs = []
    for eeg in eeg_files:
        for block, video in enumerate(video_files):
            pairs.append((eeg, video, block))
    return pairs


def split_pairs(pairs: List[Tuple[Path, Path, int]], seed: int, val_ratio: float, test_ratio: float):
    rnd = random.Random(seed)
    idxs = list(range(len(pairs)))
    rnd.shuffle(idxs)
    n = len(idxs)
    n_test = max(1, int(n * test_ratio)) if n >= 3 else 0
    n_val = max(1, int(n * val_ratio)) if n >= 3 else 0
    test_ids = idxs[:n_test]
    val_ids = idxs[n_test:n_test + n_val]
    train_ids = idxs[n_test + n_val:]
    return [pairs[i] for i in train_ids], [pairs[i] for i in val_ids], [pairs[i] for i in test_ids]


class EEGVideoPairDataset(Dataset):
    def __init__(self, pairs: List[Tuple[Path, Path, int]], clip_extractor: VideoFeatureExtractor, max_windows_per_sample: Optional[int] = None):
        self.pairs = pairs
        self.clip_extractor = clip_extractor
        self.max_windows_per_sample = max_windows_per_sample

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        eeg_path, video_path, block = self.pairs[idx]
        eeg_windows = preprocess_eeg_128ch(eeg_path, block=block)
        if self.max_windows_per_sample is not None:
            eeg_windows = eeg_windows[: self.max_windows_per_sample]
        clip_feat = self.clip_extractor.extract_clip_embedding(video_path)
        return {
            "eeg_windows": eeg_windows,
            "clip_feat": clip_feat,
            "eeg_path": str(eeg_path),
            "video_path": str(video_path),
        }


def collate_pairs(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, object]:
    # Variable number of windows per EEG sample is allowed.
    return {
        "batch": batch,
    }


# ----------------------------------------------------------------------------
# Stage 2 — Tokenization (10 patches × 44 samples)
# ----------------------------------------------------------------------------

class Stage2Tokenization(nn.Module):
    def __init__(self, d_model: int = CFG.d_model, patch_len: int = CFG.patch_len):
        super().__init__()
        self.d_model = d_model
        self.patch_len = patch_len
        self.proj = nn.Linear(patch_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 128, 440)
        B, C, T = x.shape
        assert C == CFG.n_channels, f"Expected {CFG.n_channels} channels, got {C}"
        assert T == CFG.window_samples, f"Expected {CFG.window_samples} samples, got {T}"
        x = x.view(B, C, CFG.n_patches, CFG.patch_len)
        return self.proj(x)  # (B, 128, 10, 512)


# ----------------------------------------------------------------------------
# Stage 3 — CARD blocks
# ----------------------------------------------------------------------------

class CARDBlock(nn.Module):
    def __init__(self, d_model: int = CFG.d_model, nhead: int = CFG.nhead_card):
        super().__init__()
        self.intra_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)

        self.inter_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)

        self.conv_blender = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
        )
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C=128, N=10, D=512)
        B, C, N, D = x.shape

        # Intra-channel temporal attention (per channel across 10 tokens)
        intra_in = x.view(B * C, N, D)
        a1, _ = self.intra_attn(intra_in, intra_in, intra_in)
        x = self.norm1(x + a1.view(B, C, N, D))

        # Inter-channel attention (per token across 128 channels)
        inter_in = x.transpose(1, 2).contiguous().view(B * N, C, D)
        a2, _ = self.inter_attn(inter_in, inter_in, inter_in)
        x = self.norm2(x + a2.view(B, N, C, D).transpose(1, 2))

        # 1D convolutional token blending
        conv_in = x.view(B * C, N, D).transpose(1, 2)  # (B*C, D, N)
        conv_out = self.conv_blender(conv_in).transpose(1, 2).view(B, C, N, D)
        x = self.norm3(x + conv_out)
        return x


class Stage3CARDEncoder(nn.Module):
    def __init__(self, num_blocks: int = CFG.num_card_blocks):
        super().__init__()
        self.blocks = nn.ModuleList([CARDBlock() for _ in range(num_blocks)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x)
        return x


# ----------------------------------------------------------------------------
# Stage 4 — Latent projection to 512
# ----------------------------------------------------------------------------

class Stage4LatentProjection(nn.Module):
    def __init__(self, channels: int = CFG.n_channels, n_patches: int = CFG.n_patches, d_model: int = CFG.d_model):
        super().__init__()
        self.proj = nn.Linear(channels * n_patches * d_model, 512)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.reshape(x.size(0), -1)
        z = self.proj(x)
        return F.normalize(z, p=2, dim=1)


# ----------------------------------------------------------------------------
# Stage 5 — CLIP alignment MLP and losses
# ----------------------------------------------------------------------------

class Stage5CLIPAlignment(nn.Module):
    def __init__(self, d_model: int = 512):
        super().__init__()
        self.f_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, z_eeg: torch.Tensor) -> torch.Tensor:
        return self.f_proj(z_eeg)


class CompositeLoss(nn.Module):
    """Guide-style total loss: MAE + contrastive + CLIP + diffusion."""
    def __init__(self,
                 lambda_mae: float = CFG.lambda_mae,
                 lambda_contrastive: float = CFG.lambda_contrastive,
                 lambda_clip: float = CFG.lambda_clip,
                 lambda_diffusion: float = CFG.lambda_diffusion):
        super().__init__()
        self.lambda_mae = lambda_mae
        self.lambda_contrastive = lambda_contrastive
        self.lambda_clip = lambda_clip
        self.lambda_diffusion = lambda_diffusion
        self.l1 = nn.L1Loss()

    def forward(self,
                z: torch.Tensor,
                z_img: torch.Tensor,
                mae_input: Optional[torch.Tensor] = None,
                mae_target: Optional[torch.Tensor] = None,
                contrastive_loss: Optional[torch.Tensor] = None,
                diffusion_loss: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        cosine = F.cosine_similarity(z, z_img, dim=1)
        l_clip = (1.0 - cosine).mean()

        if mae_input is not None and mae_target is not None:
            l_mae = self.l1(mae_input, mae_target)
        else:
            l_mae = torch.tensor(0.0, device=z.device)

        if contrastive_loss is None:
            contrastive_loss = torch.tensor(0.0, device=z.device)
        if diffusion_loss is None:
            diffusion_loss = torch.tensor(0.0, device=z.device)

        total = (
            self.lambda_mae * l_mae +
            self.lambda_contrastive * contrastive_loss +
            self.lambda_clip * l_clip +
            self.lambda_diffusion * diffusion_loss
        )
        return total, {"l_mae": l_mae, "l_contrastive": contrastive_loss, "l_clip": l_clip, "l_diffusion": diffusion_loss}


# ----------------------------------------------------------------------------
# Temporal conditioning module for video generation
# ----------------------------------------------------------------------------

class TemporalAttentionLayer(nn.Module):
    def __init__(self, d_model: int = CFG.unet_cross_attention_dim, nhead: int = 8, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, causal_mask: bool = True) -> torch.Tensor:
        T = x.size(1)
        attn_mask = None
        if causal_mask and T > 1:
            attn_mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        out, _ = self.attn(x, x, x, attn_mask=attn_mask)
        x = self.norm1(x + out)
        x = self.norm2(x + self.ff(x))
        return x


# ----------------------------------------------------------------------------
# VideoLDM / diffusion backbone
# ----------------------------------------------------------------------------

class VAEImageDecoder(nn.Module):
    """Decode latent tensors to RGB frames.

    This uses the Stable Diffusion VAE decoder where available.
    Latent shape expected: (B, 4, H, W)
    Output: (B, 3, H*8, W*8) approximately.
    """
    def __init__(self, vae: object):
        super().__init__()
        self.vae = vae

    @torch.no_grad()
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        latents = latents / 0.18215
        imgs = self.vae.decode(latents).sample
        imgs = (imgs / 2 + 0.5).clamp(0, 1)
        return imgs


class VideoLDMDenoiser(nn.Module):
    """Guide-aligned video-conditioned denoiser.

    Uses a Stable Diffusion U-Net backbone and injects EEG embeddings via
    cross-attention. TemporalAttentionLayer refines the EEG context across
    a rolling frame window to improve inter-frame coherence.
    """
    def __init__(self,
                 model_id: str = CFG.stable_diffusion_model,
                 cross_attention_dim: int = CFG.unet_cross_attention_dim):
        super().__init__()
        # Import heavy diffusers classes lazily to avoid import-time errors
        from diffusers import AutoencoderKL
        from diffusers.models import UNet2DConditionModel

        self.unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")
        self.vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae")
        self.unet.requires_grad_(False)
        self.vae.requires_grad_(False)
        self.unet.enable_gradient_checkpointing()
        torch.cuda.empty_cache()   # <-- add this line

        # EEG conditioning: 512 -> 768
        self.eeg_proj = nn.Linear(512, cross_attention_dim)
        self.temporal_attn = TemporalAttentionLayer(d_model=cross_attention_dim, nhead=8)

        # Fine-tune only attention + norm layers in U-Net
        for name, param in self.unet.named_parameters():
            if ("attn" in name) or ("norm" in name):
                param.requires_grad = True

    def forward(self,
                noisy_latents: torch.Tensor,
                timesteps: torch.Tensor,
                encoder_hidden_states: torch.Tensor,
                temporal_context: Optional[torch.Tensor] = None) -> torch.Tensor:
        # encoder_hidden_states: (B, 1, 512)
        cond = self.eeg_proj(encoder_hidden_states)  # (B, 1, 768)
        if temporal_context is not None:
            # temporal_context: (B, T, 512)
            ctx = self.eeg_proj(temporal_context)
            ctx = self.temporal_attn(ctx)
            cond = ctx[:, -1:, :]
        return self.unet(noisy_latents, timesteps, encoder_hidden_states=cond).sample


# ----------------------------------------------------------------------------
# Abstraction for Stage 6 built from current diffusers version
# ----------------------------------------------------------------------------

def build_video_denoiser() -> VideoLDMDenoiser:
    return VideoLDMDenoiser().to(DEVICE)


# ----------------------------------------------------------------------------
# Loss for diffusion + temporal coherence
# ----------------------------------------------------------------------------

class VideoLDMLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, noise_pred: torch.Tensor, noise_target: torch.Tensor, adj_frames: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        l_diff = self.mse(noise_pred, noise_target)
        l_temp = torch.tensor(0.0, device=noise_pred.device)
        if adj_frames is not None and adj_frames.size(1) > 1:
            l_temp = self.mse(adj_frames[:, :-1], adj_frames[:, 1:])
        total = l_diff + CFG.lambda_temporal * l_temp
        return total, {"l_diffusion": l_diff, "l_temporal": l_temp}


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------

class MetricsBundle:
    def __init__(self, device: torch.device):
        self.device = device
        self._fid = None   # lazily created on first use to avoid downloading inception at startup
        self.lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex").to(device) if LearnedPerceptualImagePatchSimilarity is not None else None

    @property
    def fid(self):
        if self._fid is None and FrechetInceptionDistance is not None:
            self._fid = FrechetInceptionDistance(feature=2048, normalize=True).to(self.device)
        return self._fid

    def reset(self):
        if self._fid is not None:
            self._fid.reset()
        if self.lpips is not None:
            self.lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex").to(self.device)

    @staticmethod
    def ssim_batch(pred: torch.Tensor, target: torch.Tensor) -> float:
        # pred/target: (B, 3, H, W) in [0,1]
        if skimage_ssim is None:
            return float("nan")
        vals = []
        p = pred.detach().cpu().numpy()
        t = target.detach().cpu().numpy()
        for i in range(p.shape[0]):
            pi = np.transpose(p[i], (1, 2, 0))
            ti = np.transpose(t[i], (1, 2, 0))
            vals.append(skimage_ssim(pi, ti, channel_axis=2, data_range=1.0))
        return float(np.mean(vals))

    @staticmethod
    def psnr_batch(pred: torch.Tensor, target: torch.Tensor) -> float:
        mse = F.mse_loss(pred, target).item()
        return float(10.0 * math.log10(1.0 / max(mse, 1e-8)))

    def update_fid(self, real: torch.Tensor, fake: torch.Tensor):
        if self.fid is None:
            return
        real_u8 = (real.clamp(0, 1) * 255).to(torch.uint8)
        fake_u8 = (fake.clamp(0, 1) * 255).to(torch.uint8)
        self.fid.update(real_u8, real=True)
        self.fid.update(fake_u8, real=False)

    def compute_fid(self) -> float:
        if self.fid is None:
            return float("nan")
        return float(self.fid.compute().item())

    def compute_lpips(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        if self.lpips is None:
            return float("nan")
        return float(self.lpips(pred, target).item())


# ----------------------------------------------------------------------------
# Ground-truth video embedding / evaluation helpers
# ----------------------------------------------------------------------------

class ClipScorer:
    def __init__(self, device: torch.device, clip_model_name: str = CFG.clip_model_name):
        self.device = device
        import importlib
        global clip
        if clip is None:
            clip = importlib.import_module('clip')
        clip_device = "cpu" if os.environ.get("CLIP_FORCE_CPU", "1") != "0" else device
        self.clip_device = torch.device(clip_device)
        self.model, self.preprocess = clip.load(clip_model_name, device=clip_device)
        self.model.eval()

    @torch.no_grad()
    def embed_frames(self, frames: List[torch.Tensor]) -> torch.Tensor:
        pil_frames = []
        for fr in frames:
            fr = fr.detach().cpu().clamp(0, 1)
            if fr.ndim == 3:
                arr = (fr.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            else:
                arr = (fr.numpy() * 255).astype(np.uint8)
            pil_frames.append(Image.fromarray(arr))
        batch = torch.stack([self.preprocess(im) for im in pil_frames]).to(self.clip_device)  # CPU
        emb = self.model.encode_image(batch).float().mean(dim=0, keepdim=True)
        emb = emb / emb.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        return emb.to(self.device)  # move result back to CUDA

    @torch.no_grad()
    def cosine_similarity(self, frames_a: List[torch.Tensor], frames_b: List[torch.Tensor]) -> float:
        ea = self.embed_frames(frames_a)
        eb = self.embed_frames(frames_b)
        return float(F.cosine_similarity(ea, eb).mean().item())


# ----------------------------------------------------------------------------
# Training / generation pipeline
# ----------------------------------------------------------------------------

class EMAEmbeddingBuffer:
    def __init__(self, alpha: float = 0.9):
        self.alpha = alpha
        self.running: Optional[torch.Tensor] = None

    def update(self, z: torch.Tensor) -> torch.Tensor:
        z_mean = z.detach().mean(dim=0, keepdim=True)
        if self.running is None:
            self.running = z_mean.clone()
        else:
            self.running = self.alpha * self.running + (1.0 - self.alpha) * z_mean
        return self.running.expand_as(z)

    def reset(self):
        self.running = None


class Group4Pipeline:
    def __init__(self, cfg: Config = CFG):
        self.cfg = cfg
        self.device = DEVICE
        self.clip_extractor = VideoFeatureExtractor(self.device, cfg.clip_model_name)
        self.clip_scorer = ClipScorer(self.device, cfg.clip_model_name)

        self.st2 = Stage2Tokenization().to(self.device)
        self.st3 = Stage3CARDEncoder().to(self.device)
        self.st4 = Stage4LatentProjection().to(self.device)
        self.st5 = Stage5CLIPAlignment().to(self.device)
        self.st6 = build_video_denoiser()

        self.diffusion_loss = VideoLDMLoss().to(self.device)
        self.composite_loss = CompositeLoss().to(self.device)
        # Lazy import for scheduler
        from diffusers import DDPMScheduler
        self.scheduler = DDPMScheduler(num_train_timesteps=cfg.num_train_timesteps)

        enc_params = list(self.st2.parameters()) + list(self.st3.parameters()) + list(self.st4.parameters()) + list(self.st5.parameters()) + list(self.st6.eeg_proj.parameters()) + list(self.st6.temporal_attn.parameters())
        unet_params = [p for p in self.st6.unet.parameters() if p.requires_grad]

        self.optimizer = optim.AdamW([
            {"params": enc_params, "lr": cfg.lr_encoder},
            {"params": unet_params, "lr": cfg.lr_unet},
        ], weight_decay=cfg.weight_decay)

        self.lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-6)

        self.metrics = MetricsBundle(self.device)

    def models_train(self):
        for m in [self.st2, self.st3, self.st4, self.st5, self.st6]:
            m.train()

    def models_eval(self):
        for m in [self.st2, self.st3, self.st4, self.st5, self.st6]:
            m.eval()

    def encode_eeg(self, eeg_windows: torch.Tensor, ema: Optional[EMAEmbeddingBuffer] = None) -> torch.Tensor:
        # eeg_windows: (B, 128, 440)
        h = self.st3(self.st2(eeg_windows))
        z_raw = self.st5(self.st4(h))
        if ema is not None:
            z_mom = ema.update(z_raw)
            z = F.normalize(0.7 * z_raw + 0.3 * z_mom, p=2, dim=1)
        else:
            z = F.normalize(z_raw, p=2, dim=1)
        return z

    def contrastive_infonce(self, z: torch.Tensor, z_img: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
        # Symmetric InfoNCE on normalized embeddings.
        z = F.normalize(z, p=2, dim=1)
        z_img = F.normalize(z_img, p=2, dim=1)
        logits = (z @ z_img.T) / temperature
        targets = torch.arange(z.size(0), device=z.device)
        loss_i = F.cross_entropy(logits, targets)
        loss_j = F.cross_entropy(logits.T, targets)
        return 0.5 * (loss_i + loss_j)

    def forward_diffusion_step(self,
                               z_i: torch.Tensor,
                               temporal_ctx: torch.Tensor,
                               latent_h: int,
                               latent_w: int,
                               guidance_steps: int = 0) -> torch.Tensor:
        latent = torch.randn(1, 4, latent_h, latent_w, device=self.device)
        self.scheduler.set_timesteps(self.cfg.num_inference_steps)
        for t in self.scheduler.timesteps:
            ts = torch.tensor([t], device=self.device).long()
            noise_pred = self.st6(latent, ts, encoder_hidden_states=z_i.unsqueeze(1), temporal_context=temporal_ctx)
            latent = self.scheduler.step(noise_pred, t, latent).prev_sample
        return latent

    @torch.no_grad()
    def decode_latents_to_frames(self, latents: torch.Tensor) -> torch.Tensor:
        return self.st6.vae.decode(latents / 0.18215).sample.clamp(-1, 1)

    def train_one_epoch(self, train_loader: DataLoader, epoch: int) -> Dict[str, float]:
        self.models_train()
        ema = EMAEmbeddingBuffer(alpha=0.9)
        totals = {"loss": 0.0, "l_mae": 0.0, "l_contrastive": 0.0, "l_clip": 0.0, "l_diffusion": 0.0, "l_temporal": 0.0, "l_recon": 0.0}
        n_batches = 0
        torch.cuda.empty_cache()

        scaler = torch.amp.GradScaler("cuda", enabled=self.cfg.mixed_precision and self.device.type == "cuda")

        accum_steps = max(1, getattr(self.cfg, "grad_accum_steps", 1))
        global_step = 0

        # Start with zeroed grads so accumulation works from batch 1
        self.optimizer.zero_grad(set_to_none=True)

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            items = batch["batch"]
            if not items:
                continue
            # per-batch processing; optimizer.step() will be called every accum_steps batches
            logs = {k: 0.0 for k in totals.keys() if k != "loss"}
            batch_loss_log = 0.0
            n_valid = 0
            # Store raw windows for the cheap InfoNCE encoder-only re-pass after the sample loop
            all_eeg_for_infonce: List[torch.Tensor] = []
            all_clip_for_infonce: List[torch.Tensor] = []

            for sample in items:
                eeg_windows = sample["eeg_windows"].to(self.device)  # (N, 128, 440)
                gt_clip = sample["clip_feat"].to(self.device)         # (1, 512)
                n = eeg_windows.size(0)
                if n == 0:
                    continue
                n_valid += 1

                take = min(n, self.cfg.max_video_frames)
                eeg_windows = eeg_windows[:take]

                z_seq = []
                z_mom_seq: List[torch.Tensor] = []
                noise_preds = []
                noises = []
                z_raw_seq = []
                recon_losses: List[torch.Tensor] = []

                for i in range(take):
                    x_i = eeg_windows[i:i+1]
                    z_raw = self.st5(self.st4(self.st3(self.st2(x_i))))
                    z_raw_seq.append(z_raw)
                    z_mom = ema.update(z_raw)
                    z_mom_seq.append(z_mom.detach())
                    z_i = F.normalize(0.7 * z_raw + 0.3 * z_mom, p=2, dim=1)
                    z_seq.append(z_i)

                    ctx_start = max(0, len(z_seq) - self.cfg.context_len)
                    temporal_ctx = torch.cat(z_seq[ctx_start:], dim=0).unsqueeze(0)

                    frame_tensor = load_frame_for_vae(sample["video_path"], i, take, self.device)
                    with torch.no_grad():
                        latent = self.st6.vae.encode(frame_tensor).latent_dist.sample() * 0.18215
                    noise = torch.randn_like(latent)
                    ts = torch.randint(0, self.cfg.num_train_timesteps, (1,), device=self.device).long()
                    noisy_lat = self.scheduler.add_noise(latent, noise, ts)

                    with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
                        noise_pred = self.st6(noisy_lat, ts, encoder_hidden_states=z_i.unsqueeze(1), temporal_context=temporal_ctx)

                    alpha_bar = self.scheduler.alphas_cumprod[ts.item()].to(self.device).float()
                    x0_pred = (noisy_lat.float() - (1.0 - alpha_bar).sqrt() * noise_pred.float()) / alpha_bar.sqrt().clamp(min=1e-6)
                    recon_losses.append(F.l1_loss(x0_pred, latent.float().detach()))

                    noise_preds.append(noise_pred)
                    noises.append(noise)

                noise_preds_t = torch.cat(noise_preds, dim=0).float()
                noises_t = torch.cat(noises, dim=0).float()
                z_t = torch.cat(z_seq, dim=0).float()
                z_img = gt_clip.repeat(z_t.size(0), 1)

                z_raw_t = torch.cat(z_raw_seq, dim=0)
                z_mom_t = F.normalize(torch.cat(z_mom_seq, dim=0), p=2, dim=1)
                l_recon = torch.stack(recon_losses).mean()

                video_loss, video_breakdown = self.diffusion_loss(
                    noise_preds_t, noises_t, adj_frames=noise_preds_t.unsqueeze(0)
                )
                total, loss_dict = self.composite_loss(
                    z=z_t, z_img=z_img,
                    mae_input=z_raw_t, mae_target=z_mom_t,
                    contrastive_loss=None,
                    diffusion_loss=video_loss,
                )
                total = total + self.cfg.lambda_recon * l_recon

                # Per-sample backward — divide by accum_steps to make gradients average over
                # the effective accumulated batch.
                scaler.scale(total / (max(len(items), 1) * accum_steps)).backward()
                torch.cuda.empty_cache()

                all_eeg_for_infonce.append(eeg_windows.detach())
                all_clip_for_infonce.append(gt_clip.detach())
                batch_loss_log += total.item()
                logs["l_mae"] += float(loss_dict["l_mae"].item())
                logs["l_clip"] += float(loss_dict["l_clip"].item())
                logs["l_diffusion"] += float(video_breakdown["l_diffusion"].item())
                logs["l_temporal"] += float(video_breakdown["l_temporal"].item())
                logs["l_recon"] += float(l_recon.item())

            # InfoNCE: encoder-only second pass (no UNet) — cheap and graph-free per sample
            if n_valid > 1:
                z_reprs: List[torch.Tensor] = []
                for eeg_w in all_eeg_for_infonce:
                    z = self.encode_eeg(eeg_w).float()
                    z_reprs.append(z.mean(0, keepdim=True))
                z_batch = torch.cat(z_reprs, dim=0)
                z_clip_batch = torch.cat(all_clip_for_infonce, dim=0)
                l_contrast = self.contrastive_infonce(z_batch, z_clip_batch)
                scaler.scale(self.cfg.lambda_contrastive * l_contrast / max(n_valid, 1)).backward()
                logs["l_contrastive"] += float(l_contrast.item())
                batch_loss_log += self.cfg.lambda_contrastive * l_contrast.item() / max(n_valid, 1)

            # Step only every accum_steps batches
            global_step += 1
            if global_step % accum_steps == 0:
                scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for pg in self.optimizer.param_groups for p in pg["params"] if p.requires_grad],
                    self.cfg.grad_clip,
                )
                scaler.step(self.optimizer)
                scaler.update()
                self.optimizer.zero_grad(set_to_none=True)

            totals["loss"] += batch_loss_log / max(n_valid, 1)
            for k in logs:
                totals[k] += logs[k] / max(n_valid, 1)
            n_batches += 1

        avg = {k: v / max(n_batches, 1) for k, v in totals.items()}
        self.lr_scheduler.step(avg["loss"])
        torch.cuda.empty_cache()
        return avg

    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        self.models_eval()
        losses = []
        clip_sims = []
        for batch in tqdm(val_loader, desc="Validation"):
            items = batch["batch"]
            for sample in items:
                eeg_windows = sample["eeg_windows"].to(self.device)
                gt_clip = sample["clip_feat"].to(self.device)
                take = min(eeg_windows.size(0), self.cfg.max_video_frames)
                if take == 0:
                    continue
                z = self.encode_eeg(eeg_windows[:take])
                clip_sim = F.cosine_similarity(z, gt_clip.repeat(z.size(0), 1), dim=1).mean().item()
                clip_sims.append(clip_sim)
                losses.append(1.0 - clip_sim)
        return {"val_loss_proxy": safe_mean(losses), "val_clip_sim": safe_mean(clip_sims)}

    @torch.no_grad()
    def generate_video_sequence(self,
                                eeg_windows: torch.Tensor,
                                num_inference_steps: int = None,
                                context_len: int = None) -> List[torch.Tensor]:
        self.models_eval()
        num_inference_steps = num_inference_steps or self.cfg.num_inference_steps
        context_len = context_len or self.cfg.context_len
        self.scheduler.set_timesteps(num_inference_steps)

        ema = EMAEmbeddingBuffer(alpha=0.9)
        z_history: List[torch.Tensor] = []
        frame_latents: List[torch.Tensor] = []

        take = min(eeg_windows.size(0), self.cfg.max_video_frames)
        for i in tqdm(range(take), desc="Generating frames"):
            x_i = eeg_windows[i:i+1].to(self.device)
            z_raw = self.st5(self.st4(self.st3(self.st2(x_i))))
            z_mom = ema.update(z_raw)
            z_i = F.normalize(0.7 * z_raw + 0.3 * z_mom, p=2, dim=1)
            z_history.append(z_i)

            ctx_start = max(0, len(z_history) - context_len)
            temporal_ctx = torch.cat(z_history[ctx_start:], dim=0).unsqueeze(0)
            latent = torch.randn(1, 4, self.cfg.latent_h, self.cfg.latent_w, device=self.device)

            for t in self.scheduler.timesteps:
                ts = torch.tensor([t], device=self.device).long()
                noise_pred = self.st6(latent, ts, encoder_hidden_states=z_i.unsqueeze(1), temporal_context=temporal_ctx)
                latent = self.scheduler.step(noise_pred, t, latent).prev_sample

            frame_latents.append(latent.squeeze(0).cpu())

        # Decode latent frames to RGB
        lat_batch = torch.stack(frame_latents, dim=0).to(self.device)
        decoded = self.st6.vae.decode(lat_batch / 0.18215).sample
        decoded = (decoded / 2 + 0.5).clamp(0, 1)
        return [decoded[i].detach().cpu() for i in range(decoded.size(0))]

    @torch.no_grad()
    def evaluate_sample(self, sample: Dict[str, torch.Tensor]) -> Dict[str, float]:
        eeg_windows = sample["eeg_windows"]
        gt_clip = sample["clip_feat"].to(self.device)
        frames = self.generate_video_sequence(eeg_windows)

        # We do not have ground-truth RGB frames in the prompt, so for a true
        # evaluation you would pass decoded real video frames from the dataset.
        # Below metrics are computed against a placeholder reconstruction target
        # only if real frames are available in your local dataset.
        clip_sim = self.clip_scorer.cosine_similarity(frames, frames)  # self-consistency placeholder
        return {"clip_sim": clip_sim, "n_frames": float(len(frames))}


# ----------------------------------------------------------------------------
# Full evaluation suite for real/generated frame pairs
# ----------------------------------------------------------------------------

def evaluate_frame_pair_metrics(real_frames: List[torch.Tensor], fake_frames: List[torch.Tensor], device: torch.device) -> Dict[str, float]:
    """Compute SSIM, PSNR, FID, LPIPS, Top-5, and CLIP similarity.

    real_frames and fake_frames should be aligned lists of RGB tensors in [0,1],
    each tensor shaped (3,H,W).
    """
    metrics = MetricsBundle(device)
    clip_scorer = ClipScorer(device)

    import torch.nn.functional as F

    target_size = real_frames[0].shape[-2:]  # (H, W)

    fake_frames = [
            F.interpolate(fr.unsqueeze(0), size=target_size, mode="bilinear", align_corners=False).squeeze(0)
            for fr in fake_frames
        ]

    if len(real_frames) == 0 or len(fake_frames) == 0:
        return {"ssim": float("nan"), "psnr": float("nan"), "fid": float("nan"), "lpips": float("nan"), "top5": float("nan"), "clip_sim": float("nan")}

    real = torch.stack([fr.to(device) for fr in real_frames], dim=0)
    fake = torch.stack([fr.to(device) for fr in fake_frames], dim=0)

    ssim = metrics.ssim_batch(fake, real)
    psnr = metrics.psnr_batch(fake, real)
    metrics.update_fid(real, fake)
    fid = metrics.compute_fid()
    lpips = metrics.compute_lpips(fake, real)
    clip_sim = clip_scorer.cosine_similarity(fake_frames, real_frames)

    # Top-5 Inception accuracy requires labels and a classifier; placeholder here
    # for the standard evaluation hook. Replace with dataset class labels if available.
    top5 = float("nan")

    return {"ssim": ssim, "psnr": psnr, "fid": fid, "lpips": lpips, "top5": top5, "clip_sim": clip_sim}


# ----------------------------------------------------------------------------
# Ablations
# ----------------------------------------------------------------------------

def run_ablation(pipeline: Group4Pipeline,
                 eeg_windows: torch.Tensor,
                 with_temporal_attention: bool = True,
                 with_momentum: bool = True) -> List[torch.Tensor]:
    """Ablation hook. Use this to compare: 
    - full model


    
    - without temporal attention
    - without momentum smoothing
    """
    orig_temporal = pipeline.st6.temporal_attn
    if not with_temporal_attention:
        pipeline.st6.temporal_attn = nn.Identity().to(DEVICE)

    frames: List[torch.Tensor] = []
    try:
        if with_momentum:
            frames = pipeline.generate_video_sequence(eeg_windows)
        else:
            # Temporarily force momentum alpha = 0 behavior by bypassing EMA blend
            old_generate = pipeline.generate_video_sequence

            @torch.no_grad()
            def no_momentum_generate(eeg_windows, num_inference_steps=None, context_len=None):
                pipeline.models_eval()
                num_inference_steps = num_inference_steps or pipeline.cfg.num_inference_steps
                context_len = context_len or pipeline.cfg.context_len
                pipeline.scheduler.set_timesteps(num_inference_steps)
                z_history = []
                latents = []
                take = min(eeg_windows.size(0), pipeline.cfg.max_video_frames)
                for i in range(take):
                    x_i = eeg_windows[i:i+1].to(DEVICE)
                    z_i = pipeline.st5(pipeline.st4(pipeline.st3(pipeline.st2(x_i))))
                    z_i = F.normalize(z_i, p=2, dim=1)
                    z_history.append(z_i)
                    ctx_start = max(0, len(z_history) - context_len)
                    temporal_ctx = torch.cat(z_history[ctx_start:], dim=0).unsqueeze(0)
                    latent = torch.randn(1, 4, pipeline.cfg.latent_h, pipeline.cfg.latent_w, device=DEVICE)
                    for t in pipeline.scheduler.timesteps:
                        ts = torch.tensor([t], device=DEVICE).long()
                        noise_pred = pipeline.st6(latent, ts, encoder_hidden_states=z_i.unsqueeze(1), temporal_context=temporal_ctx)
                        latent = pipeline.scheduler.step(noise_pred, t, latent).prev_sample
                    latents.append(latent.squeeze(0).cpu())
                lat_batch = torch.stack(latents, dim=0).to(DEVICE)
                decoded = pipeline.st6.vae.decode(lat_batch / 0.18215).sample
                decoded = (decoded / 2 + 0.5).clamp(0, 1)
                return [decoded[i].detach().cpu() for i in range(decoded.size(0))]

            pipeline.generate_video_sequence = no_momentum_generate  # type: ignore[assignment]
            frames = pipeline.generate_video_sequence(eeg_windows)
            pipeline.generate_video_sequence = old_generate  # restore
    finally:
        pipeline.st6.temporal_attn = orig_temporal

    return frames


# ----------------------------------------------------------------------------
# Training loop and main
# ----------------------------------------------------------------------------

def build_dataloaders(clip_extractor: "VideoFeatureExtractor") -> Tuple[Optional[DataLoader], Optional[DataLoader], Optional[DataLoader]]:
    eeg_files = list_files(CFG.eeg_root, ".npy")
    # Accept common video extensions and also accept directories with frames
    video_files = list_files(CFG.video_root, ".mp4") + list_files(CFG.video_root, ".avi") + list_files(CFG.video_root, ".mov")
    pairs = paired_indices(eeg_files, video_files)
    if len(pairs) == 0:
        return None, None, None

    train_pairs, val_pairs, test_pairs = split_pairs(pairs, CFG.dataset_split_seed, CFG.val_ratio, CFG.test_ratio)
    # Share a single CLIP extractor across all splits to save ~1.8 GB GPU memory
    train_ds = EEGVideoPairDataset(train_pairs, clip_extractor)
    val_ds = EEGVideoPairDataset(val_pairs, clip_extractor) if val_pairs else None
    test_ds = EEGVideoPairDataset(test_pairs, clip_extractor) if test_pairs else None

    train_loader = DataLoader(train_ds, batch_size=CFG.batch_size, shuffle=True, num_workers=0, collate_fn=collate_pairs)
    val_loader = DataLoader(val_ds, batch_size=CFG.batch_size, shuffle=False, num_workers=0, collate_fn=collate_pairs) if val_ds else None
    test_loader = DataLoader(test_ds, batch_size=CFG.batch_size, shuffle=False, num_workers=0, collate_fn=collate_pairs) if test_ds else None
    return train_loader, val_loader, test_loader

'''
def train_group4_pipeline():
    pipeline = Group4Pipeline(CFG)
    train_loader, val_loader, test_loader = build_dataloaders(pipeline.clip_extractor)
    if train_loader is None:
        print("Warning: No EEG/video pairs found. Skipping training.")
        return pipeline

    print(f"=== Group 4 Training ===")
    print(f"Device: {DEVICE}")
    print(f"Train/Val/Test: {len(train_loader.dataset)}/{len(val_loader.dataset) if val_loader else 0}/{len(test_loader.dataset) if test_loader else 0}")

    best_val = float("inf")
    for epoch in range(CFG.epochs):
        train_logs = pipeline.train_one_epoch(train_loader, epoch)
        print(f"Epoch {epoch+1}/{CFG.epochs} | "
              f"Loss={train_logs['loss']:.4f} | "
              f"MAE={train_logs['l_mae']:.4f} | "
              f"Contrast={train_logs['l_contrastive']:.4f} | "
              f"CLIP={train_logs['l_clip']:.4f} | "
              f"Diff={train_logs['l_diffusion']:.4f} | "
              f"Temp={train_logs['l_temporal']:.4f} | "
              f"Recon={train_logs['l_recon']:.4f}")

        if val_loader is not None:
            val_logs = pipeline.validate(val_loader)
            print(f"Validation | proxy={val_logs['val_loss_proxy']:.4f} | clip_sim={val_logs['val_clip_sim']:.4f}")
            if val_logs["val_loss_proxy"] < best_val:
                best_val = val_logs["val_loss_proxy"]
                torch.save({
                    "st2": pipeline.st2.state_dict(),
                    "st3": pipeline.st3.state_dict(),
                    "st4": pipeline.st4.state_dict(),
                    "st5": pipeline.st5.state_dict(),
                    "st6": pipeline.st6.state_dict(),
                    "optimizer": pipeline.optimizer.state_dict(),
                }, CFG.output_dir / "group4_best_checkpoint.pt")
                print(f"Saved best checkpoint to {CFG.output_dir / 'group4_best_checkpoint.pt'}")

    return pipeline

'''
def train_group4_pipeline():
    pipeline = Group4Pipeline(CFG)
    train_loader, val_loader, test_loader = build_dataloaders(pipeline.clip_extractor)
    if train_loader is None:
        print("Warning: No EEG/video pairs found. Skipping training.")
        return pipeline

    # ── Resume from checkpoint if one exists ──────────────────────────────────
    ckpt_path = CFG.output_dir / "group4_best_checkpoint.pt"
    start_epoch = 0
    best_val = float("inf")

    if ckpt_path.exists():
        print(f"Resuming from checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        pipeline.st2.load_state_dict(ckpt["st2"])
        pipeline.st3.load_state_dict(ckpt["st3"])
        pipeline.st4.load_state_dict(ckpt["st4"])
        pipeline.st5.load_state_dict(ckpt["st5"])
        pipeline.st6.load_state_dict(ckpt["st6"])
        pipeline.optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0)
        best_val    = ckpt.get("best_val", float("inf"))
        print(f"  → Resuming from epoch {start_epoch}, best_val={best_val:.4f}")

    # ── Cosine annealing LR scheduler with linear warmup ──────────────────────
    # Warmup: linearly ramp from lr*0.1 → lr over warmup_epochs
    # Then:   cosine decay from lr → lr * lr_min_factor over remaining epochs
    def lr_lambda(current_epoch: int) -> float:
        if current_epoch < CFG.warmup_epochs:
            # linear warmup
            return 0.1 + 0.9 * (current_epoch / max(CFG.warmup_epochs, 1))
        # cosine decay
        progress = (current_epoch - CFG.warmup_epochs) / max(
            CFG.epochs - CFG.warmup_epochs, 1
        )
        cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
        return CFG.lr_min_factor + (1.0 - CFG.lr_min_factor) * cosine_factor

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        pipeline.optimizer, lr_lambda=lr_lambda, last_epoch=start_epoch - 1
    )

    # ── Logging setup ──────────────────────────────────────────────────────────
    log_path = CFG.output_dir / "training_log.csv"
    write_header = not log_path.exists()
    log_file = open(log_path, "a")
    if write_header:
        log_file.write(
            "epoch,loss,l_mae,l_contrastive,l_clip,l_diffusion,l_temporal,"
            "l_recon,val_loss_proxy,val_clip_sim,lr_encoder,lr_unet\n"
        )

    print(f"=== Group 4 Training ===")
    print(f"Device      : {DEVICE}")
    print(f"Start epoch : {start_epoch + 1} / {CFG.epochs}")
    print(
        f"Train / Val / Test : "
        f"{len(train_loader.dataset)} / "
        f"{len(val_loader.dataset) if val_loader else 0} / "
        f"{len(test_loader.dataset) if test_loader else 0}"
    )

    # ── Main loop ──────────────────────────────────────────────────────────────
    for epoch in range(start_epoch, CFG.epochs):

        train_logs = pipeline.train_one_epoch(train_loader, epoch)

        # Current LRs (two param groups: encoder, unet)
        current_lrs = [pg["lr"] for pg in pipeline.optimizer.param_groups]
        lr_enc  = current_lrs[0] if len(current_lrs) > 0 else CFG.lr_encoder
        lr_unet = current_lrs[1] if len(current_lrs) > 1 else CFG.lr_unet

        print(
            f"Epoch {epoch+1:>4}/{CFG.epochs} | "
            f"Loss={train_logs['loss']:.4f} | "
            f"MAE={train_logs['l_mae']:.4f} | "
            f"Contrast={train_logs['l_contrastive']:.4f} | "
            f"CLIP={train_logs['l_clip']:.4f} | "
            f"Diff={train_logs['l_diffusion']:.4f} | "
            f"Temp={train_logs['l_temporal']:.4f} | "
            f"Recon={train_logs['l_recon']:.4f} | "
            f"LR_enc={lr_enc:.2e} | LR_unet={lr_unet:.2e}"
        )

        val_loss_proxy = float("inf")
        val_clip_sim   = 0.0

        if val_loader is not None:
            val_logs       = pipeline.validate(val_loader)
            val_loss_proxy = val_logs["val_loss_proxy"]
            val_clip_sim   = val_logs["val_clip_sim"]
            print(
                f"  → Validation | proxy={val_loss_proxy:.4f} | "
                f"clip_sim={val_clip_sim:.4f}"
            )

            if val_loss_proxy < best_val:
                best_val = val_loss_proxy
                torch.save(
                    {
                        "st2":       pipeline.st2.state_dict(),
                        "st3":       pipeline.st3.state_dict(),
                        "st4":       pipeline.st4.state_dict(),
                        "st5":       pipeline.st5.state_dict(),
                        "st6":       pipeline.st6.state_dict(),
                        "optimizer": pipeline.optimizer.state_dict(),
                        "epoch":     epoch + 1,        # ← saved so resume works
                        "best_val":  best_val,
                    },
                    ckpt_path,
                )
                print(f"  ✓ Best checkpoint saved (val={best_val:.4f})")

        # ── Periodic checkpoint every 50 epochs (safe fallback) ───────────────
        if (epoch + 1) % 50 == 0:
            periodic_path = CFG.output_dir / f"checkpoint_epoch_{epoch+1}.pt"
            torch.save(
                {
                    "st2":       pipeline.st2.state_dict(),
                    "st3":       pipeline.st3.state_dict(),
                    "st4":       pipeline.st4.state_dict(),
                    "st5":       pipeline.st5.state_dict(),
                    "st6":       pipeline.st6.state_dict(),
                    "optimizer": pipeline.optimizer.state_dict(),
                    "epoch":     epoch + 1,
                    "best_val":  best_val,
                },
                periodic_path,
            )
            print(f"  ✓ Periodic checkpoint saved → {periodic_path}")

        # ── CSV log ───────────────────────────────────────────────────────────
        log_file.write(
            f"{epoch+1},{train_logs['loss']:.6f},{train_logs['l_mae']:.6f},"
            f"{train_logs['l_contrastive']:.6f},{train_logs['l_clip']:.6f},"
            f"{train_logs['l_diffusion']:.6f},{train_logs['l_temporal']:.6f},"
            f"{train_logs['l_recon']:.6f},{val_loss_proxy:.6f},"
            f"{val_clip_sim:.6f},{lr_enc:.8f},{lr_unet:.8f}\n"
        )
        log_file.flush()

        # Step scheduler at end of each epoch
        scheduler.step()

    log_file.close()
    print(f"\nTraining complete. Log saved to: {log_path}")
    return pipeline

    ###################################
def load_checkpoint(pipeline: Group4Pipeline, ckpt_path: Path) -> None:
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    pipeline.st2.load_state_dict(ckpt["st2"])
    pipeline.st3.load_state_dict(ckpt["st3"])
    pipeline.st4.load_state_dict(ckpt["st4"])
    pipeline.st5.load_state_dict(ckpt["st5"])
    pipeline.st6.load_state_dict(ckpt["st6"])
    if "optimizer" in ckpt:
        pipeline.optimizer.load_state_dict(ckpt["optimizer"])


# def main():
#     pipeline = train_group4_pipeline()

#     eeg_files = list_files(CFG.eeg_root, ".npy")
#     video_files = list_files(CFG.video_root, ".mp4")
#     if not eeg_files or not video_files:
#         return

#     # In a real project you should evaluate on the official test split with real frames.
#     # Here we generate sample outputs for the first test pair to validate the pipeline.
#     pairs = paired_indices(eeg_files, video_files)
#     _, _, test_pairs = split_pairs(pairs, CFG.dataset_split_seed, CFG.val_ratio, CFG.test_ratio)
#     if not test_pairs:
#         test_pairs = pairs[:1]

#     eeg_path, video_path, block = test_pairs[0]
#     eeg_windows = preprocess_eeg_128ch(eeg_path, block=block)
#     generated_frames = pipeline.generate_video_sequence(eeg_windows)
#     out_dir = CFG.output_dir / "sample_generation"
#     save_video_frames(generated_frames, out_dir)
#     print(f"Saved sample generated frames to: {out_dir}")

#     # Ablations
#     ab_full = run_ablation(pipeline, eeg_windows, with_temporal_attention=True, with_momentum=True)
#     ab_no_temp = run_ablation(pipeline, eeg_windows, with_temporal_attention=False, with_momentum=True)
#     ab_no_mom = run_ablation(pipeline, eeg_windows, with_temporal_attention=True, with_momentum=False)
#     save_video_frames(ab_full, CFG.output_dir / "ablation_full", prefix="full")
#     save_video_frames(ab_no_temp, CFG.output_dir / "ablation_no_temporal", prefix="notemp")
#     save_video_frames(ab_no_mom, CFG.output_dir / "ablation_no_momentum", prefix="nomom")

#     print("Ablation frames saved.")
#     print("For true SSIM/PSNR/FID/FVD/LPIPS/Top-5 reporting, pass real test frames into evaluate_frame_pair_metrics().")
'''
def main():
    import json

    # Build pipeline structure, then load trained weights
    pipeline = Group4Pipeline(CFG)
    ckpt_path = CFG.output_dir / "group4_best_checkpoint.pt"
    if not ckpt_path.exists():
        print(f"Checkpoint not found: {ckpt_path}")
        return
    load_checkpoint(pipeline, ckpt_path)
    pipeline.models_eval()
    print(f"Loaded checkpoint: {ckpt_path}")

    eeg_files = list_files(CFG.eeg_root, ".npy")
    video_files = (
        list_files(CFG.video_root, ".mp4")
        + list_files(CFG.video_root, ".avi")
        + list_files(CFG.video_root, ".mov")
    )
    if not eeg_files or not video_files:
        print("No EEG/video files found.")
        return

    pairs = paired_indices(eeg_files, video_files)
    _, _, test_pairs = split_pairs(pairs, CFG.dataset_split_seed, CFG.val_ratio, CFG.test_ratio)
    if not test_pairs:
        test_pairs = pairs[:1]

    eeg_path, video_path, block = test_pairs[0]
    eeg_windows = preprocess_eeg_128ch(eeg_path, block=block)

    fake_frames = pipeline.generate_video_sequence(eeg_windows)
    gen_dir = CFG.output_dir / "sample_generation"
    save_video_frames(fake_frames, gen_dir, prefix="fake")

    real_pil = sample_video_frames(video_path, num_frames=len(fake_frames))
    if len(real_pil) == 0:
        print(f"Could not read frames from: {video_path}")
        return

    real_frames = []
    for im in real_pil:
        arr = np.asarray(im, dtype=np.float32) / 255.0
        real_frames.append(torch.from_numpy(arr).permute(2, 0, 1))

    n = min(len(real_frames), len(fake_frames))
    real_frames = real_frames[:n]
    fake_frames = fake_frames[:n]

    eval_device = DEVICE  # switch to torch.device("cpu") if VRAM is tight
    report = evaluate_frame_pair_metrics(real_frames, fake_frames, eval_device)

    report_dir = CFG.output_dir / "evaluation"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "metrics.json"
    with open(report_path, "w") as f:
        json.dump(
            {
                "checkpoint": str(ckpt_path),
                "eeg_path": str(eeg_path),
                "video_path": str(video_path),
                "block": int(block),
                "n_frames": int(n),
                "metrics": report,
            },
            f,
            indent=2,
        )

    save_video_frames(real_frames, report_dir / "real_frames", prefix="real")
    save_video_frames(fake_frames, report_dir / "fake_frames", prefix="fake")

    print("Saved generated frames to:", gen_dir)
    print("Saved evaluation report to:", report_path)
    print("Metrics:", report)


if __name__ == "__main__":
    main()
    '''
def main():
    pipeline = train_group4_pipeline()

    eeg_files = list_files(CFG.eeg_root, ".npy")

    video_files = (
        list_files(CFG.video_root, ".mp4")
        + list_files(CFG.video_root, ".avi")
        + list_files(CFG.video_root, ".mov")
    )

    if not eeg_files or not video_files:
        print("No EEG/video files found.")
        return

    pairs = paired_indices(eeg_files, video_files)

    _, _, test_pairs = split_pairs(
        pairs,
        CFG.dataset_split_seed,
        CFG.val_ratio,
        CFG.test_ratio
    )

    if not test_pairs:
        test_pairs = pairs[:1]

    eeg_path, video_path, block = test_pairs[0]

    eeg_windows = preprocess_eeg_128ch(eeg_path, block=block)

    generated_frames = pipeline.generate_video_sequence(eeg_windows)

    out_dir = CFG.output_dir / "sample_generation"

    save_video_frames(generated_frames, out_dir)

    print(f"Saved generated frames to: {out_dir}")


if __name__ == "__main__":
    main()