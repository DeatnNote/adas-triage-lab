"""Synthetic dashcam frames rendered from each event's scene attributes, plus a classical image-feature baseline.

Real SSR footage is not public, so the vision step is evaluated on rendered frames whose contents are known
exactly. They are deliberately simple; a vision-language model that fails here would fail on real video too,
but passing here says little about real-world accuracy.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

SKY = {"day": (120, 170, 230), "low_sun": (235, 150, 90), "night": (15, 20, 40)}


def render(sc: dict, seed: int, w: int = 512, h: int = 288) -> Image.Image:
    rng = np.random.default_rng(seed)
    light, weather = sc["light"], sc["weather"]
    objs = set(sc["objects"].split("|")) if isinstance(sc["objects"], str) else set(sc["objects"])
    objs.discard("")
    img = Image.new("RGB", (w, h), SKY[light])
    d = ImageDraw.Draw(img, "RGBA")
    hz = int(h * 0.45)
    ground = (60, 60, 62) if light != "night" else (25, 25, 28)
    d.rectangle([0, hz, w, h], fill=(70, 110, 60) if light != "night" else (20, 30, 20))
    # road: straight or curving
    curve = 80 if "curve" in objs else 0
    top_l, top_r = w * 0.47 + curve, w * 0.53 + curve
    d.polygon([(w * 0.05, h), (w * 0.95, h), (top_r, hz), (top_l, hz)], fill=ground)
    # lane markings
    faded = sc["lane_markings"] == "faded"
    for frac in (0.36, 0.64):
        for k in range(10):
            if faded and rng.random() < 0.6:
                continue
            y0 = hz + (h - hz) * (k / 10) ** 1.6
            y1 = hz + (h - hz) * ((k + 0.5) / 10) ** 1.6
            t0, t1 = (y0 - hz) / (h - hz), (y1 - hz) / (h - hz)
            x0 = (top_l + (top_r - top_l) * frac) * (1 - t0) + (w * 0.05 + w * 0.9 * frac) * t0
            x1 = (top_l + (top_r - top_l) * frac) * (1 - t1) + (w * 0.05 + w * 0.9 * frac) * t1
            a = 90 if faded else 255
            d.line([(x0, y0), (x1, y1)], fill=(240, 240, 240, a), width=max(1, int(1 + 4 * t1)))
    if "sun" in objs:
        cx = w * rng.uniform(0.35, 0.65)
        for r, a in ((70, 60), (45, 110), (22, 255)):
            d.ellipse([cx - r, hz - 10 - r, cx + r, hz - 10 + r], fill=(255, 245, 200, a))
    if "overpass" in objs:
        d.rectangle([0, hz - 55, w, hz - 25], fill=(110, 110, 115))
        for x in (w * 0.15, w * 0.85):
            d.rectangle([x - 8, hz - 25, x + 8, hz + 20], fill=(100, 100, 105))
    if "tunnel_exit" in objs:
        d.rectangle([0, 0, w, h], fill=(10, 10, 12, 255))
        d.rounded_rectangle([w * 0.3, hz - 50, w * 0.7, hz + 40], radius=40, fill=(255, 255, 250))
        d.polygon([(w * 0.05, h), (w * 0.95, h), (w * 0.62, hz + 40), (w * 0.38, hz + 40)], fill=(30, 30, 32))
    if "lead_car" in objs:
        cx, cy, s = w * 0.5 + curve * 0.5, hz + 25, 26
        d.rectangle([cx - s, cy - s * 0.6, cx + s, cy + s * 0.4], fill=(40, 60, 140))
        d.rectangle([cx - s, cy - s * 0.1, cx - s + 8, cy + 2], fill=(220, 30, 30))
        d.rectangle([cx + s - 8, cy - s * 0.1, cx + s, cy + 2], fill=(220, 30, 30))
    if "car_adjacent_close" in objs:
        side = rng.choice([-1, 1])
        cx, cy, s = w * (0.5 + side * 0.27), h * 0.72, 80
        d.rectangle([cx - s, cy - s * 0.55, cx + s, cy + s * 0.45], fill=(150, 30, 30))
        d.rectangle([cx - s * 0.7, cy - s * 0.5, cx + s * 0.7, cy - s * 0.15], fill=(60, 80, 100))
        for x in (cx - s * 0.7, cx + s * 0.7):
            d.ellipse([x - 16, cy + s * 0.3, x + 16, cy + s * 0.6], fill=(15, 15, 15))
    if "cones" in objs:
        for k in range(5):
            t = 0.2 + 0.16 * k
            x = (top_l + (top_r - top_l) * 0.3) * (1 - t) + (w * 0.2) * t
            y = hz + (h - hz) * t
            s = 6 + 18 * t
            d.polygon([(x, y - s * 1.6), (x - s * 0.6, y), (x + s * 0.6, y)], fill=(255, 120, 0))
            d.line([(x - s * 0.35, y - s * 0.7), (x + s * 0.35, y - s * 0.7)], fill=(255, 255, 255), width=2)
    if "pedestrian" in objs:
        x, y, s = w * rng.uniform(0.42, 0.58), hz + 70, 34
        col = (240, 220, 60)
        d.ellipse([x - 6, y - s - 12, x + 6, y - s], fill=col)
        d.line([(x, y - s), (x, y - 12)], fill=col, width=5)
        d.line([(x, y - 12), (x - 10, y + 6)], fill=col, width=5)
        d.line([(x, y - 12), (x + 10, y + 6)], fill=col, width=5)
        d.line([(x - 12, y - s + 10), (x + 12, y - s + 6)], fill=col, width=4)
    if light == "night":
        d.polygon([(w * 0.2, h), (w * 0.8, h), (w * 0.56, hz + 20), (w * 0.44, hz + 20)], fill=(255, 250, 200, 45))
    if weather == "rain":
        for _ in range(260):
            x, y = rng.uniform(0, w), rng.uniform(0, h)
            d.line([(x, y), (x - 4, y + 16)], fill=(200, 210, 230, 140), width=1)
    elif weather == "snow":
        for _ in range(400):
            x, y, r = rng.uniform(0, w), rng.uniform(0, h), rng.uniform(1, 3)
            d.ellipse([x - r, y - r, x + r, y + r], fill=(255, 255, 255, 220))
        d.rectangle([0, h - 18, w, h], fill=(235, 235, 240, 200))
    elif weather == "fog":
        img = Image.alpha_composite(img.convert("RGBA"), Image.new("RGBA", (w, h), (215, 215, 215, 150))).convert("RGB")
        img = img.filter(ImageFilter.GaussianBlur(2))
    return img


def image_features(img: Image.Image) -> np.ndarray:
    """Hand-crafted features: colour statistics by region, brightness, edge density. No learned vision."""
    a = np.asarray(img.convert("RGB").resize((128, 72)), dtype="float32") / 255
    g = a.mean(2)
    feats = []
    for rows in (slice(0, 24), slice(24, 48), slice(48, 72)):
        for cols in (slice(0, 43), slice(43, 85), slice(85, 128)):
            blk = a[rows, cols]
            feats += list(blk.mean((0, 1))) + [blk.std()]
    gx, gy = np.abs(np.diff(g, axis=1)).mean(), np.abs(np.diff(g, axis=0)).mean()
    hsv_orange = ((a[..., 0] > 0.8) & (a[..., 1] > 0.3) & (a[..., 1] < 0.6) & (a[..., 2] < 0.2)).mean()
    yellow = ((a[..., 0] > 0.8) & (a[..., 1] > 0.75) & (a[..., 2] < 0.4)).mean()
    white = (g > 0.9).mean()
    feats += [g.mean(), g.std(), gx, gy, hsv_orange, yellow, white]
    return np.array(feats, dtype="float32")
