"""Controlled compositing-provenance clips with independent video/identity pools."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter


def clip(rng, composite, size=64, frames=8):
    yy, xx = np.mgrid[:size, :size]
    base = rng.normal(size=(size, size, 3))
    base = gaussian_filter(base, (0.7, 0.7, 0))
    base /= max(float(base.std()), 1e-6)
    base = 0.5 + rng.uniform(0.045, 0.11) * base
    box = np.array([size // 4, size // 4, 3 * size // 4, 3 * size // 4])
    face = (xx >= box[0]) & (xx < box[2]) & (yy >= box[1]) & (yy < box[3])
    natural_detail = rng.uniform(0.75, 1.3)
    base[face] = 0.5 + natural_detail * (base[face] - 0.5)
    values = []
    amplitude = rng.uniform(0.005, 0.09)
    for t in range(frames):
        frame = np.roll(base, int(np.sin(t / 2) * 2), axis=1).copy()
        frame += rng.normal(0, 0.008, frame.shape)
        if composite:
            # Independently rendered patch has resampling/phase mismatch; the label
            # is the compositing operation, never a bit appended to image features.
            patch = np.sin(2 * np.pi * (0.25 * xx + 0.31 * yy) + t * rng.uniform(0.1, 0.4))
            frame += amplitude * patch[..., None] * face[..., None]
        values.append(np.clip(frame, 0, 1))
    landmarks = np.array([[size * 0.4, size * 0.4], [size * 0.6, size * 0.4],
                          [size * 0.5, size * 0.55], [size * 0.45, size * 0.65]])
    return dict(clip=np.stack(values).astype(np.float32), box=box, landmarks=landmarks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--groups", type=int, default=80)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    if args.groups < 10:
        parser.error("Use at least ten independent identity/video groups")
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(str(output))
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for group in range(args.groups):
        split = "calibration" if group < args.groups // 2 else "test"
        rng = np.random.default_rng(np.random.SeedSequence([args.seed, group]))
        label = group % 2
        name = f"clip_{group:03d}.npz"
        np.savez_compressed(output / name, **clip(rng, bool(label)))
        records.append(dict(id=str(group), path=name, video=f"video-{group}",
                            identity=f"identity-{group}", split=split, label=label,
                            allowed_cues=["frequency", "boundary", "temporal"]))
    (output / "manifest.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    (output / "source.json").write_text(json.dumps(dict(
        source="procedural texture clips with compositing provenance",
        truth="whether an independent resampled patch was composited; no human identity claim",
        seed=args.seed, groups=args.groups,
    ), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
