"""Export an actual fixed spatial/temporal residual detector, without weights."""

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


class ResidualDetector(nn.Module):
    """Compare central-patch high-frequency and temporal energy to its surround."""

    def __init__(self, threshold=0.012, scale=80.0):
        super().__init__()
        self.threshold = threshold
        self.scale = scale

    def contrast(self, value):
        h, w = value.shape[-2:]
        center = value[..., h // 4:3 * h // 4, w // 4:3 * w // 4]
        inner = center.mean((1, 2, 3, 4))
        total = value.sum((1, 2, 3, 4))
        inner_sum = center.sum((1, 2, 3, 4))
        outside = (total - inner_sum) / (value[0].numel() - center[0].numel())
        return inner - outside

    def forward(self, video):
        residual = (video - F.avg_pool3d(video, (1, 5, 5), stride=1, padding=(0, 2, 2))).abs()
        spatial = self.contrast(residual)
        temporal = self.contrast((video[:, :, 1:] - video[:, :, :-1]).abs())
        return (self.scale * (spatial + 0.4 * temporal - self.threshold))[:, None]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--threshold", type=float, default=0.012)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(str(output))
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.jit.script(ResidualDetector(args.threshold).eval()).save(str(output))
    output.with_suffix(".json").write_text(json.dumps(dict(
        detector="fixed_spatial_temporal_residual_contrast", learned=False,
        input="RGB [B,3,T,H,W] in [0,1], T>=2", threshold=args.threshold,
    ), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
