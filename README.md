# SIASE

Implementation of paired shortcut-intervention audits for selective cross-generator deepfake forensics.

## Installation

Use Python `>=3.11`. Run commands from the repository root. Create the environment, activate it with `source .venv/bin/activate` (Linux/macOS) or `.venv\Scripts\Activate.ps1` (PowerShell), then install the package.

```bash
python -m venv .venv
```

```bash
python -m pip install -e .
```

## Data generation and preparation

The generator creates aligned geometric clips and a fixed boundary-energy detector. These are generated interventions and a fixed detector, not natural deepfake videos. `source.json` records the process and seed.

```bash
python -m siase.synthetic --output data/generated --groups 80 --seed 17
python -m siase.fixed_detector --output checkpoints/reproduce_detector.ts --threshold 0.012
siase prepare --config configs/reproduce.json --input data/generated/manifest.json --data data/reproduce
```

For external data: Acquire [FaceForensics++](https://github.com/ondyari/FaceForensics), [Celeb-DF](https://github.com/yuezunli/celeb-deepfakeforensics) or [DF40](https://github.com/YZY-stack/DF40) under their access terms. Export aligned RGB clips as NPZ `clip` [time,height,width,3] in [0,1], `box` [x0,y0,x1,y1] and visible `landmarks` [...,2]. A JSON manifest contains `id`, `path`, original `video`, `identity`, `split` (calibration/test), `label` (fake=1), and `allowed_cues` drawn from frequency/boundary/temporal. Allowed cues declare provenance-reviewed label-preserving interventions. Split identities and original videos before clip extraction; preparation rejects overlap.

## Training and inference

`configs/reproduce.json` fixes seed `17`. Use the same config and prepared inputs throughout; change `seed` there for another run and keep its output directories separate.

```bash
siase audit --config configs/reproduce.json --data data/reproduce --detector checkpoints/reproduce_detector.ts --output runs/reproduce_audit
siase train --config configs/reproduce.json --audits runs/reproduce_audit/audits.json --output checkpoints/reproduce
```

## Evaluation

```bash
siase evaluate --config configs/reproduce.json --audits runs/reproduce_audit/audits.json --checkpoint checkpoints/reproduce/risk.json --output runs/reproduce
```

## Comparisons and ablations

Fit each supported `--variant` into a separate directory, evaluate its own checkpoint, then pass the evaluated run directories to `compare --inputs`.

```bash
siase compare --inputs runs/reproduce --output outputs/comparison
siase plot --output outputs/comparison
```

## Outputs

The selected commands write their generated inputs under `data/generated/`, prepared data under `data/reproduce/`, and checkpoints/results in the explicit output directories. Corresponding steps produce `audits.json`, `pairs.json`, `risk.json`, `metrics.json`, `predictions.csv`, `source.json`. Comparisons write CSV/JSON and plots under their output directory.

## Verification

Source checks used Python 3.13.5. Installed dependency metadata: numpy 2.1.3, torch 2.7.1+cu118, scipy 1.15.3, scikit-learn 1.6.1, matplotlib 3.10.0. Python AST, configuration files, CLI parser definitions and source input/output paths were checked without invoking command handlers. Training, full evaluation and CI were not run for this review.
