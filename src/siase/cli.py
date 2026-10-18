import argparse
from pathlib import Path
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from scipy.special import expit
from .common import load, save, table, plot, auc, roc
from .interventions import transform, nuisance, quality, matched


def prepare(a, c):
    records = load(a.input)
    groups = {}
    result = []
    for record in records:
        for key in ["video", "identity"]:
            group = (key, str(record[key]))
            if group in groups and groups[group] != record["split"]:
                raise ValueError("Video or identity crosses data splits")
            groups[group] = record["split"]
        path = Path(record["path"])
        path = path if path.is_absolute() else Path(a.input).parent / path
        result.append(dict(record, path=str(path.resolve())))
    save(Path(a.data) / "manifest.json", result)


def audit(a, c):
    model = torch.jit.load(a.detector, map_location=c["device"]).eval()
    perceptual_key = c.get("perceptual_metric", "lpips")
    if perceptual_key == "lpips":
        import lpips
        perceptual = lpips.LPIPS(net="alex").to(c["device"]).eval()
    elif perceptual_key == "fixed_multiscale_l1":
        perceptual = None
    else:
        raise ValueError("perceptual_metric must be lpips or fixed_multiscale_l1")
    records = load(Path(a.data) / "manifest.json")
    rows = []
    pairs = []

    def logit(clip):
        x = (
            torch.from_numpy(clip.astype("float32"))
            .permute(3, 0, 1, 2)[None]
            .to(c["device"])
        )
        with torch.no_grad():
            value = model(x).detach().cpu().reshape(-1).numpy()
        if len(value) == 1:
            return float(value[0])
        if len(value) == 2:
            return float(value[1] - value[0])
        raise ValueError("Detector must output one logit or two class logits")

    for record in records:
        data = np.load(record["path"])
        clip = data["clip"].astype("float32")
        box = data["box"].tolist()
        if clip.min() < 0 or clip.max() > 1:
            raise ValueError("Clips must be RGB float in [0,1]")
        if "landmarks" not in data:
            raise ValueError(
                "Landmark coordinates are required for visibility screening"
            )
        landmarks = data["landmarks"]
        h, w = clip.shape[1:3]
        visible = bool(
            np.all(
                (landmarks[..., 0] >= 0)
                & (landmarks[..., 0] < w)
                & (landmarks[..., 1] >= 0)
                & (landmarks[..., 1] < h)
            )
        )
        original = logit(clip)
        sign = 1 if original >= 0 else -1
        variance = float(np.var([sign * logit(x) for x in nuisance(clip)]))
        contrasts = []
        attenuation = []
        unmatched = []
        for kind in ["frequency", "boundary", "temporal"]:
            if kind not in record["allowed_cues"] or not visible:
                continue
            target = transform(clip, box, kind, c["severity"], True)
            if target is None:
                continue
            tq = quality(clip, target, box, perceptual, c["device"], c["fps"])
            target_support = sign * logit(target)
            shams = []
            for strength in c["sham_strengths"]:
                sham = transform(clip, box, kind, strength, False)
                if sham is None:
                    continue
                sq = quality(clip, sham, box, perceptual, c["device"], c["fps"])
                support = sign * logit(sham)
                shams.append(
                    (
                        abs(tq["psnr"] - sq["psnr"])
                        + 20 * abs(tq[perceptual_key] - sq[perceptual_key]),
                        support,
                        sq,
                        strength,
                    )
                )
            if not shams:
                continue
            best = min(shams, key=lambda z: z[0])
            accepted = matched(tq, best[2], c)
            unmatched.append(shams[0][1] - target_support)
            pairs.append(
                dict(
                    id=record["id"],
                    cue=kind,
                    target_quality=tq,
                    sham_quality=best[2],
                    sham_strength=best[3],
                    accepted=accepted,
                    contrast=best[1] - target_support,
                )
            )
            if accepted:
                contrasts.append(best[1] - target_support)
                attenuation.append(sign * original - target_support)
        rows.append(
            dict(
                id=record["id"],
                video=record["video"],
                split=record["split"],
                label=record["label"],
                logit=original,
                margin=abs(original),
                nuisance_variance=variance,
                contrast=float(np.mean(contrasts)) if contrasts else None,
                target_attenuation=float(np.mean(attenuation)) if attenuation else None,
                unmatched=float(np.mean(unmatched)) if unmatched else None,
                accepted_pairs=len(contrasts),
            )
        )
    save(Path(a.output) / "audits.json", rows)
    save(Path(a.output) / "pairs.json", pairs)
    save(Path(a.output) / "audit_config.json", dict(c, detector=a.detector))


def features(rows, variant):
    keys = {
        "margin": ["margin"],
        "nuisance": ["margin", "nuisance_variance"],
        "target": ["margin", "target_attenuation"],
        "unmatched": ["margin", "nuisance_variance", "unmatched"],
        "uncalibrated": ["margin", "nuisance_variance", "contrast"],
        "siase": ["margin", "nuisance_variance", "contrast"],
    }[variant]
    return np.array([[r[k] for k in keys] for r in rows]), keys


def train(a, c):
    rows = [
        r
        for r in load(a.audits)
        if r["split"] == "calibration" and r["accepted_pairs"] > 0
    ]
    x, keys = features(rows, a.variant)
    y = np.array([(r["logit"] >= 0) != bool(r["label"]) for r in rows])
    if a.variant != "uncalibrated" and len(np.unique(y)) != 2:
        raise ValueError(
            "Risk calibration requires correct and incorrect detector decisions"
        )
    median = np.median(x, axis=0)
    scale = np.maximum(np.median(np.abs(x - median), axis=0), 1e-4)
    if a.variant == "uncalibrated":
        coefficient = np.array([-1.0, 1.0, -1.0])
        intercept = 0.0
    else:
        model = LogisticRegression(C=c["logistic_C"], random_state=c["seed"]).fit(
            (x - median) / scale, y
        )
        coefficient = model.coef_[0]
        intercept = float(model.intercept_[0])
    save(
        Path(a.output) / "risk.json",
        dict(
            variant=a.variant,
            keys=keys,
            median=median,
            scale=scale,
            coefficient=coefficient,
            intercept=intercept,
            threshold=c["error_threshold"],
            videos=sorted({r["video"] for r in rows}),
        ),
    )


def evaluate(a, c):
    state = load(a.checkpoint)
    all_rows = [r for r in load(a.audits) if r["split"] == "test"]
    rows = [r for r in all_rows if r["accepted_pairs"] > 0]
    if {r["video"] for r in rows} & set(state["videos"]):
        raise ValueError("Calibration/test videos overlap")
    x = np.array([[r[k] for k in state["keys"]] for r in rows])
    normalized = (x - np.array(state["median"])) / np.array(state["scale"])
    risk = expit(normalized @ np.array(state["coefficient"]) + state["intercept"])
    y = np.array([r["label"] for r in rows])
    score = np.array([r["logit"] for r in rows])
    error = (score >= 0) != y.astype(bool)
    order = np.argsort(risk, kind="stable")
    curve = np.cumsum(error[order]) / np.arange(1, len(rows) + 1)
    feasible = np.flatnonzero(curve <= c["error_threshold"])
    fpr, tpr = roc(y, score)
    summary = dict(
        method=state["variant"],
        base_auroc=auc(y, score),
        tpr_at_1pct_fpr=float(np.max(tpr[fpr <= 0.01])) if fpr is not None else None,
        error_auroc=auc(error, risk),
        aurc=float(curve.mean()),
        coverage_at_target_error=float((feasible[-1] + 1) / len(rows))
        if len(feasible)
        else 0.0,
        calibrated_coverage=float(np.mean(risk <= state["threshold"])),
        accepted_pair_coverage=len(rows) / max(1, len(all_rows)),
        selective_error=float(error[risk <= state["threshold"]].mean())
        if np.any(risk <= state["threshold"])
        else None,
    )
    save(Path(a.output) / "metrics.json", summary)
    table(
        Path(a.output) / "predictions.csv",
        [
            dict(
                id=r["id"],
                risk=float(q),
                error=bool(e),
                accepted=bool(q <= state["threshold"]),
            )
            for r, q, e in zip(rows, risk, error)
        ],
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "command", choices=["prepare", "audit", "train", "evaluate", "compare", "plot"]
    )
    p.add_argument("--config", default="configs/default.json")
    p.add_argument("--input")
    p.add_argument("--data", default="data/clips")
    p.add_argument("--output", default="runs/full")
    p.add_argument("--detector")
    p.add_argument("--audits", default="runs/audit/audits.json")
    p.add_argument("--checkpoint", default="checkpoints/risk.json")
    p.add_argument("--inputs", nargs="*")
    p.add_argument(
        "--variant",
        choices=["margin", "nuisance", "target", "unmatched", "uncalibrated", "siase"],
        default="siase",
    )
    a = p.parse_args()
    c = load(a.config)
    if a.command == "prepare":
        prepare(a, c)
    elif a.command == "audit":
        audit(a, c)
    elif a.command == "train":
        train(a, c)
    elif a.command == "evaluate":
        evaluate(a, c)
    elif a.command == "compare":
        rows = [load(Path(d) / "metrics.json") for d in a.inputs]
        save(Path(a.output) / "comparison.json", rows)
        table(Path(a.output) / "comparison.csv", rows)
    else:
        plot(Path(a.output) / "comparison.json", Path(a.output) / "figures")
