import cv2
import numpy as np
import torch


def support(shape, box, kind, target):
    t, h, w, channels = shape
    x0, y0, x1, y1 = map(int, box)
    face = np.zeros((h, w), float)
    face[max(0, y0) : min(h, y1), max(0, x0) : min(w, x1)] = 1
    if kind == "boundary":
        kernel = np.ones((max(3, int(min(x1 - x0, y1 - y0) * 0.12) | 1),) * 2, np.uint8)
        inner = cv2.erode(face.astype("uint8"), kernel)
        chosen = face - inner if target else inner
    else:
        chosen = face if target else 1 - face
    if not target:
        count = int(np.sum(support(shape, box, kind, True)))
        selected = np.flatnonzero(chosen)
        if len(selected) < count:
            return None
        result = np.zeros(h * w)
        result[selected[:count]] = 1
        chosen = result.reshape(h, w)
    return chosen[None, :, :, None]


def transform(clip, box, kind, strength, target):
    mask = support(clip.shape, box, kind, target)
    if mask is None or not mask.any():
        return None
    if kind in ("frequency", "boundary"):
        blurred = np.stack([cv2.GaussianBlur(frame, (5, 5), 1.2) for frame in clip])
        residual = clip - blurred
    elif kind == "temporal":
        residual = np.zeros_like(clip)
        h, w = clip.shape[1:3]
        yy, xx = np.mgrid[:h, :w].astype("float32")
        for i in range(1, len(clip)):
            previous = cv2.cvtColor(clip[i - 1], cv2.COLOR_RGB2GRAY)
            current = cv2.cvtColor(clip[i], cv2.COLOR_RGB2GRAY)
            flow = cv2.calcOpticalFlowFarneback(
                (current * 255).astype("uint8"),
                (previous * 255).astype("uint8"),
                None,
                0.5,
                3,
                15,
                3,
                5,
                1.2,
                0,
            )
            warped = cv2.remap(
                clip[i - 1],
                xx + flow[:, :, 0],
                yy + flow[:, :, 1],
                cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT,
            )
            residual[i] = clip[i] - warped
    else:
        raise ValueError(kind)
    return np.clip(clip - strength * mask * residual, 0, 1).astype("float32")


def nuisance(clip):
    result = []
    jpeg = []
    for frame in clip:
        ok, encoded = cv2.imencode(
            ".jpg",
            np.uint8(np.clip(frame, 0, 1) * 255)[:, :, ::-1],
            [cv2.IMWRITE_JPEG_QUALITY, 80],
        )
        if not ok:
            raise ValueError("JPEG encoding failed")
        jpeg.append(cv2.imdecode(encoded, cv2.IMREAD_COLOR)[:, :, ::-1] / 255.0)
    result.append(np.stack(jpeg).astype("float32"))
    h, w = clip.shape[1:3]
    result.append(
        np.stack(
            [
                cv2.resize(cv2.resize(f, (max(2, w // 2), max(2, h // 2))), (w, h))
                for f in clip
            ]
        )
    )
    result.append(np.clip(clip * 1.05 + 0.01, 0, 1))
    result.append(clip[np.minimum(np.arange(len(clip)) // 2 * 2, len(clip) - 1)])
    return result


def quality(original, altered, box, lpips_model, device, fps):
    mse = float(np.mean((original - altered) ** 2))
    psnr = float(-10 * np.log10(max(mse, 1e-12)))
    if lpips_model is None:
        differences = []
        for sigma in (0.0, 1.0, 2.0):
            left = original if sigma == 0 else np.stack([cv2.GaussianBlur(frame, (0, 0), sigma) for frame in original])
            right = altered if sigma == 0 else np.stack([cv2.GaussianBlur(frame, (0, 0), sigma) for frame in altered])
            differences.append(float(np.mean(np.abs(left - right))))
        perceptual = float(np.mean(differences))
        perceptual_key = "fixed_multiscale_l1"
    else:
        x = torch.from_numpy(original).permute(0, 3, 1, 2).to(device) * 2 - 1
        y = torch.from_numpy(altered).permute(0, 3, 1, 2).to(device) * 2 - 1
        with torch.no_grad():
            perceptual = float(lpips_model(x, y).mean())
        perceptual_key = "lpips"
    bits = 0
    blur = []
    face_quality = []
    x0, y0, x1, y1 = map(int, box)
    for frame in altered:
        image = np.uint8(np.clip(frame, 0, 1) * 255)
        ok, buffer = cv2.imencode(
            ".jpg", image[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 95]
        )
        if not ok:
            raise ValueError("JPEG encoding failed")
        bits += len(buffer) * 8
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        blur.append(lap.var())
        face_quality.append(lap[y0:y1, x0:x1].var())
    return dict(
        psnr=psnr,
        **{perceptual_key: perceptual},
        bitrate=bits * fps / len(altered),
        blur=np.mean(blur),
        face_quality=np.mean(face_quality),
    )


def matched(left, right, c):
    if abs(left["psnr"] - right["psnr"]) > c["psnr_tolerance"]:
        return False
    key = c.get("perceptual_metric", "lpips")
    tolerance = c["lpips_tolerance"] if key == "lpips" else c["perceptual_tolerance"]
    if abs(left[key] - right[key]) > tolerance:
        return False
    for key in ["bitrate", "blur", "face_quality"]:
        if (
            abs(left[key] - right[key]) / max(left[key], right[key], 1e-8)
            > c[key + "_tolerance"]
        ):
            return False
    return True
