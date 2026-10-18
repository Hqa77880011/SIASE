# SIASE

Implementation of paired shortcut-intervention audits for selective cross-generator deepfake forensics.

## Installation


Use Python 3.11 or newer. Create and activate a virtual environment, then install:

```bash
python -m venv .venv
```

Activate with `source .venv/bin/activate` on Linux/macOS or `.venv\Scripts\Activate.ps1` in PowerShell, then install:

```bash
python -m pip install -e .
``` For GPU execution, install matching PyTorch and torchvision builds using the [PyTorch selector](https://pytorch.org/get-started/locally/).


## Clips and frozen detectors

Acquire [FaceForensics++](https://github.com/ondyari/FaceForensics), [Celeb-DF](https://github.com/yuezunli/celeb-deepfakeforensics) or [DF40](https://github.com/YZY-stack/DF40) under their access terms. Export aligned RGB clips as NPZ `clip` [time,height,width,3] in [0,1], `box` [x0,y0,x1,y1] and visible `landmarks` [...,2]. A JSON manifest contains `id`, `path`, original `video`, `identity`, `split` (calibration/test), `label` (fake=1), and `allowed_cues` drawn from frequency/boundary/temporal. Allowed cues declare provenance-reviewed label-preserving interventions. Split identities and original videos before clip extraction; preparation rejects overlap.

The detector is a frozen TorchScript module taking [batch,3,time,height,width] in RGB [0,1] and returning one fake logit or two real/fake logits. Export your SBI, AltFreezing, FreqNet, TALL, RECCE or other detector with its native preprocessing wrapped into that module. No detector parameters are updated by the audit.

```bash
siase prepare --input data/manifest.json --data data/clips
siase audit --data data/clips --detector checkpoints/detector.ts --output runs/audit
siase train --audits runs/audit/audits.json --variant siase --output checkpoints
siase evaluate --audits runs/audit/audits.json --checkpoint checkpoints/risk.json --output runs/full
siase compare --inputs runs/full --output outputs/comparison
siase plot --output outputs/comparison
```

## Interventions and risk calibration

Nuisances are JPEG, resize/recovery, mild photometry and repeated-frame rate reduction. Signed support always uses the **original** predicted sign. Cue attenuation removes spatial high-frequency residuals, face-boundary residuals or motion-compensated temporal residuals. Optical flow warps the previous frame into the current frame. Shams apply the same residual family to equal-area off-cue support, scanning the configured strengths. Pairs must match PSNR, LPIPS, measured MJPEG-95 bitrate, blur and face-region Laplacian quality within frozen tolerances. Geometric landmark visibility and source provenance are checked before error calibration. `pairs.json` logs accepted and rejected comparisons; no accepted pair means the clip is outside the audit's usable scope.

Calibration fits a median/MAD-scaled logistic error predictor from margin, nuisance variance and paired sham-minus-target support. The detector is frozen. Train separate risk models with `margin`, `nuisance`, `target`, `unmatched` and `uncalibrated` for incremental controls, then evaluate each checkpoint to a separate directory. The uncalibrated control uses fixed standardized risk coefficients [-1,1,-1] for margin/variance/contrast without fitting detector errors. This implementation uses explicitly defined residual supports and quality proxies; application-specific forensic cue masks can replace the support function without changing the paired contrast.

## Settings and outputs

`configs/default.json` controls severity, sham strengths, perceptual matching tolerances, logistic regularization, report risk threshold, FPS and device. LPIPS downloads its published AlexNet weights on first use. The calibration threshold is an estimated per-sample error probability, not a finite-sample guarantee. Evaluation reports base AUROC/TPR at 1% FPR, error-ranking AUROC, AURC, empirical maximum coverage at 5% selective error, deployed threshold coverage/error and accepted-pair coverage. Base and audit metrics use the same accepted-pair cohort; the cohort fraction is always reported. The empirical coverage-at-error metric is a test diagnostic, never a threshold selected for deployment.

`audits.json`, `pairs.json`, `risk.json`, per-clip predictions and metric JSON retain the complete audit chain. Compare detector checkpoints with identical manifests, supports and tolerances. Comparison CSV/JSON and PNG/SVG plots are generated from saved measurements.

## Registered compact reproduction setting

One frozen detector and provenance-reviewed cue supports, fixed target/sham severity grid and the registered quality-matching tolerances. Logistic calibration is trained locally; test coverage-at-error remains a diagnostic.

Run from the repository root after installation. The following commands use the registered setting; they are provided for execution and have not been run as part of the source review.

```bash
siase prepare --config configs/reproduce.json --input data/manifest.json --data data/reproduce
siase audit --config configs/reproduce.json --data data/reproduce --detector checkpoints/detector.ts --output runs/reproduce_audit
siase train --config configs/reproduce.json --audits runs/reproduce_audit/audits.json --output checkpoints/reproduce
siase evaluate --config configs/reproduce.json --audits runs/reproduce_audit/audits.json --checkpoint checkpoints/reproduce/risk.json --output runs/reproduce
```

Required external inputs: identity/video-disjoint aligned clips with cue provenance, a frozen detector TorchScript and cached LPIPS weights. The formal input fields are specified above; preparation/training interfaces consume those fields directly. No particular paper dataset or pretrained task model is assumed beyond these declared inputs.

## Multiple seeds

Run matched configurations with different seeds in separate directories, then aggregate the generated metric files:

```bash
siase-report --inputs runs/seed0/metrics.json runs/seed1/metrics.json --output outputs/seeds
```

Use actual metric paths from your runs, including method subdirectories where applicable. Keep dataset, split and experimental condition fixed within a report. Summary CSV/JSON gives per-method arithmetic means and sample standard deviations; a single run has no standard-deviation estimate.
