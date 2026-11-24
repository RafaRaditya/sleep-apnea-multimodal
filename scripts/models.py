"""Stage 4a: model definitions for the proposed architecture and three baselines.

Implements Sections 3.4.1 and 3.4.5-3.4.8 of paper RM (1).pdf:

  ECG   -> patch embedding + learnable positional embedding -> Transformer encoder
  Resp  -> patch embedding + learnable positional embedding -> Transformer encoder
        -> bidirectional cross-attention (ECG<-resp and resp<-ECG)
        -> mean pooling of each fused sequence -> concatenation
        -> dropout + hidden FC + nonlinearity + 2 logits

Full-length windows are used, so no attention mask is needed: tail windows were
truncated in Stage 2 rather than padded. Numeric hyperparameters come from
config/model.json and are provisional (TBD in the paper).

Baselines (3.4.8): ECG-only, respiration-only, simple fusion of the two pooled
encoder outputs without cross-attention.

Run directly to check shapes, parameter counts, gradient flow, and CPU step time.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
CFG = json.loads((ROOT / "config" / "model.json").read_text(encoding="utf-8"))
ARCH = CFG["architecture"]
ECG_FS = 100
RESP_FS = 25


class PatchEmbedding(nn.Module):
    """Non-overlapping patch projection: 1D convolution with kernel = stride = P."""

    def __init__(self, patch_samples: int, d_model: int, dropout: float):
        super().__init__()
        self.proj = nn.Conv1d(1, d_model, kernel_size=patch_samples, stride=patch_samples)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.proj(x).transpose(1, 2))


class BranchEncoder(nn.Module):
    """Patch + positional embedding + M pre-norm Transformer layers."""

    def __init__(self, patch_seconds: float, fs: int, duration_s: int, arch: dict):
        super().__init__()
        patch_samples = int(round(patch_seconds * fs))
        if patch_samples < 1:
            raise ValueError(f"patch_seconds={patch_seconds} too small for fs={fs}")
        n_samples = int(round(duration_s * fs))
        if n_samples % patch_samples:
            raise ValueError(
                f"{duration_s}s at {fs} Hz gives {n_samples} samples, not divisible by "
                f"patch of {patch_samples} samples; padding/truncation policy is TBD")
        self.patch_samples = patch_samples
        self.n_tokens = n_samples // patch_samples
        self.embed = PatchEmbedding(patch_samples, arch["d_model"], arch["dropout"])
        self.pos = nn.Parameter(torch.zeros(1, self.n_tokens, arch["d_model"]))
        nn.init.normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=arch["d_model"],
            nhead=arch["n_heads"],
            dim_feedforward=arch["ffn_dim"],
            dropout=arch["dropout"],
            activation="gelu",
            batch_first=True,
            norm_first=arch["norm_first"],
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=arch["n_layers"],
                                             enable_nested_tensor=False)
        self.norm = nn.LayerNorm(arch["d_model"])

    def _positional(self, n_tokens: int, ref: torch.Tensor) -> torch.Tensor:
        if n_tokens != self.n_tokens:
            raise ValueError(
                f"window produced {n_tokens} tokens, positional embedding expects "
                f"{self.n_tokens}; a branch must keep a fixed input length")
        return self.pos.to(ref.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.embed(x)
        h = h + self._positional(h.shape[1], h)
        return self.norm(self.encoder(h))


class CrossAttentionBlock(nn.Module):
    """One direction of cross-attention with residual, layer norm, and FFN."""

    def __init__(self, arch: dict):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            arch["d_model"], arch["n_heads"], dropout=arch["attn_dropout"],
            batch_first=True)
        self.norm1 = nn.LayerNorm(arch["d_model"])
        self.norm2 = nn.LayerNorm(arch["d_model"])
        d, f = arch["d_model"], arch["ffn_dim"]
        self.ffn = nn.Sequential(
            nn.Linear(d, f), nn.GELU(), nn.Dropout(arch["dropout"]), nn.Linear(f, d))
        self.drop = nn.Dropout(arch["dropout"])

    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        h = self.norm1(query)
        attn_out, _ = self.attn(h, context, context, need_weights=False)
        x = query + self.drop(attn_out)
        return x + self.drop(self.ffn(self.norm2(x)))


class FusionHead(nn.Module):
    """Dropout + hidden FC + nonlinearity + 2-class output."""

    def __init__(self, in_dim: int, arch: dict):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(arch["dropout"]),
            nn.Linear(in_dim, arch["head_hidden"]),
            nn.GELU(),
            nn.Dropout(arch["dropout"]),
            nn.Linear(arch["head_hidden"], arch["n_classes"]),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ProposedModel(nn.Module):
    def __init__(self, arch: dict = ARCH, duration_s: int = 10,
                 patch_seconds: float | None = None):
        super().__init__()
        patch_seconds = patch_seconds or arch["patch_seconds"]
        self.ecg = BranchEncoder(patch_seconds, ECG_FS, duration_s, arch)
        self.resp = BranchEncoder(patch_seconds, RESP_FS, duration_s, arch)
        self.ecg_from_resp = CrossAttentionBlock(arch)
        self.resp_from_ecg = CrossAttentionBlock(arch)
        self.head = FusionHead(2 * arch["d_model"], arch)

    def forward(self, ecg: torch.Tensor, resp: torch.Tensor) -> torch.Tensor:
        he = self.ecg(ecg)
        hr = self.resp(resp)
        fe = self.ecg_from_resp(he, hr)
        fr = self.resp_from_ecg(hr, he)
        z = torch.cat([fe.mean(dim=1), fr.mean(dim=1)], dim=1)
        return self.head(z)


class SingleModalModel(nn.Module):
    """ECG-only or respiration-only baseline."""

    def __init__(self, modality: str, arch: dict = ARCH, duration_s: int = 10,
                 patch_seconds: float | None = None):
        super().__init__()
        if modality not in ("ecg", "resp"):
            raise ValueError(modality)
        patch_seconds = patch_seconds or arch["patch_seconds"]
        fs = ECG_FS if modality == "ecg" else RESP_FS
        self.modality = modality
        self.branch = BranchEncoder(patch_seconds, fs, duration_s, arch)
        self.head = FusionHead(arch["d_model"], arch)

    def forward(self, ecg: torch.Tensor, resp: torch.Tensor) -> torch.Tensor:
        x = ecg if self.modality == "ecg" else resp
        return self.head(self.branch(x).mean(dim=1))


class SimpleFusionModel(nn.Module):
    """Two encoders and concatenated pooled features, no cross-attention."""

    def __init__(self, arch: dict = ARCH, duration_s: int = 10,
                 patch_seconds: float | None = None):
        super().__init__()
        patch_seconds = patch_seconds or arch["patch_seconds"]
        self.ecg = BranchEncoder(patch_seconds, ECG_FS, duration_s, arch)
        self.resp = BranchEncoder(patch_seconds, RESP_FS, duration_s, arch)
        self.head = FusionHead(2 * arch["d_model"], arch)

    def forward(self, ecg: torch.Tensor, resp: torch.Tensor) -> torch.Tensor:
        z = torch.cat([self.ecg(ecg).mean(dim=1), self.resp(resp).mean(dim=1)], dim=1)
        return self.head(z)


def build(name: str, duration_s: int = 10, arch: dict = ARCH) -> nn.Module:
    if name == "proposed":
        return ProposedModel(arch, duration_s)
    if name in ("ecg_only", "resp_only"):
        return SingleModalModel("ecg" if name == "ecg_only" else "resp", arch, duration_s)
    if name == "simple_fusion":
        return SimpleFusionModel(arch, duration_s)
    raise ValueError(f"unknown model: {name}")


MODEL_NAMES = ["proposed", "ecg_only", "resp_only", "simple_fusion"]


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _benchmark(name: str, duration: int, batch: int = 32, steps: int = 3) -> dict:
    torch.manual_seed(0)
    model = build(name)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    ecg = torch.randn(batch, 1, duration * ECG_FS)
    resp = torch.randn(batch, 1, duration * RESP_FS)
    target = torch.randint(0, 2, (batch,))

    t0 = time.perf_counter()
    for _ in range(steps):
        optimizer.zero_grad()
        loss = criterion(model(ecg, resp), target)
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        optimizer.step()
    dt = (time.perf_counter() - t0) / steps
    return {
        "model": name, "duration_s": duration, "params": n_params(model),
        "tokens_per_branch": duration // 2 if duration % 2 == 0 else None,
        "sec_per_step_batch32": round(dt, 4),
        "windows_per_sec": round(batch / dt, 1),
        "loss": round(float(loss.detach()), 4), "grad_norm": round(grad_norm, 2),
        "finite": bool(torch.isfinite(loss)),
    }


def main() -> None:
    print("architecture:", json.dumps(ARCH))
    for name in MODEL_NAMES:
        for dur in (10, 30):
            m = build(name, dur)
            ecg = torch.randn(2, 1, dur * ECG_FS)
            resp = torch.randn(2, 1, dur * RESP_FS)
            out = m(ecg, resp)
            assert out.shape == (2, 2), (name, dur, out.shape)
            assert torch.isfinite(out).all()
            print(f"{name:<14} {dur}s logits={tuple(out.shape)} params={n_params(m):,}")
    print()

    # 10 s -> 50 tokens/branch, 30 s -> 150 tokens/branch at 0.2 s patches
    for dur in (10, 30):
        tok = int(round(dur / ARCH["patch_seconds"]))
        print(f"{dur}s window -> {tok} tokens per branch "
              f"(ecg {dur * ECG_FS} samples, resp {dur * RESP_FS} samples, "
              f"patch {int(round(ARCH['patch_seconds'] * ECG_FS))}"
              f"/{int(round(ARCH['patch_seconds'] * RESP_FS))} samples)")
    print()

    rows = [_benchmark(n, 10) for n in MODEL_NAMES]
    for r in rows:
        print(f"{r['model']:<14} params={r['params']:>9,} "
              f"step={r['sec_per_step_batch32']:.3f}s "
              f"win/s={r['windows_per_sec']:>6.1f} loss={r['loss']:.4f} "
              f"grad_norm={r['grad_norm']:.2f} finite={r['finite']}")

    out_dir = ROOT / "artifacts" / "pilot"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "model_smoke.json").write_text(
        json.dumps({"architecture": ARCH, "benchmark": rows}, indent=2), encoding="utf-8")
    print("\nwrote artifacts/pilot/model_smoke.json")


if __name__ == "__main__":
    main()
