import math
import random
import tempfile
import time
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import requests


DEFAULT_CAPTIONS = [
    "Our first hello",
    "Coffee and laughter",
    "Late-night conversations",
    "Tiny adventures",
    "You + Me",
    "Sunset promises",
    "Favorite memory",
    "Still falling for you",
    "Forever vibe",
    "Love in every frame",
]

LOVE_LETTER = [
    "My love,",
    "",
    "Every little moment with you feels like a star I get to keep.",
    "Thank you for filling ordinary days with wonder.",
    "I choose you in every timeline, every universe, every heartbeat.",
    "",
    "Happy Valentine's Day.",
    "Always yours.",
]

HELP_TEXT = "Move hand = rotate | Move closer = zoom | Pinch = memory burst | Q/Esc to quit | R to reset"
PINCH_NORM_THRESH = 0.04


def clamp(value, vmin, vmax):
    return max(vmin, min(vmax, value))


def lerp(a, b, t):
    return a + (b - a) * t


def ensure_bgr(img):
    if img is None:
        return None
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


def make_gradient_placeholder(width, height, text, idx):
    y = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    x = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :]

    c1 = np.array(
        [
            30 + (idx * 23) % 90,
            35 + (idx * 31) % 110,
            90 + (idx * 43) % 120,
        ],
        dtype=np.float32,
    )
    c2 = np.array(
        [
            110 + (idx * 17) % 120,
            50 + (idx * 19) % 100,
            160 + (idx * 29) % 90,
        ],
        dtype=np.float32,
    )

    mix = 0.62 * x + 0.38 * y
    img = (
        c1[None, None, :] * (1.0 - mix[..., None]) + c2[None, None, :] * mix[..., None]
    )

    circles = np.zeros_like(img)
    for k in range(6):
        cx = int((k + 1) * width / 7)
        cy = int((0.2 + 0.6 * ((k + idx) % 5) / 4.0) * height)
        rr = int(min(width, height) * (0.05 + 0.03 * ((k + idx) % 3)))
        cv2.circle(circles, (cx, cy), rr, (255, 255, 255), -1, cv2.LINE_AA)
    circles = cv2.GaussianBlur(circles, (0, 0), 16)
    img = np.clip(img + circles * 0.18, 0, 255)

    out = img.astype(np.uint8)
    cv2.putText(
        out,
        text,
        (int(width * 0.07), int(height * 0.55)),
        cv2.FONT_HERSHEY_DUPLEX,
        1.2,
        (245, 245, 250),
        2,
        cv2.LINE_AA,
    )
    return out


def download_placeholder_images(count=8):
    ids = [10, 20, 28, 30, 37, 42, 55, 63, 70, 81, 92, 101, 119, 129]
    random.shuffle(ids)
    temp_dir = Path(tempfile.mkdtemp(prefix="memories_"))
    images = []

    for i in range(count):
        image = None
        pid = ids[i % len(ids)]
        url = f"https://picsum.photos/id/{pid}/960/720"
        try:
            resp = requests.get(url, timeout=6)
            if resp.status_code == 200 and resp.content:
                buf = np.frombuffer(resp.content, dtype=np.uint8)
                image = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                if image is not None:
                    save_path = temp_dir / f"memory_{i + 1}.jpg"
                    cv2.imwrite(str(save_path), image)
        except Exception:
            image = None

        if image is None:
            image = make_gradient_placeholder(960, 720, f"Memory #{i + 1}", i)

        images.append(image)

    return images


def load_images(memories_dir, min_count=6, max_count=10):
    path = Path(memories_dir)
    images = []
    exts = {".jpg", ".jpeg", ".png", ".webp"}

    if path.exists() and path.is_dir():
        files = sorted([p for p in path.iterdir() if p.suffix.lower() in exts])
        for p in files:
            img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
            img = ensure_bgr(img)
            if img is not None and img.size > 0:
                images.append(img)

    if len(images) == 0:
        count = random.randint(min_count, max_count)
        images = download_placeholder_images(count=count)

    if len(images) == 0:
        count = 8
        images = [
            make_gradient_placeholder(960, 720, f"Memory #{i + 1}", i)
            for i in range(count)
        ]

    return images


def make_rounded_card_mask(width, height, radius):
    radius = int(clamp(radius, 2, min(width, height) // 2))
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.rectangle(mask, (radius, 0), (width - radius, height), 255, -1)
    cv2.rectangle(mask, (0, radius), (width, height - radius), 255, -1)
    cv2.circle(mask, (radius, radius), radius, 255, -1, cv2.LINE_AA)
    cv2.circle(mask, (width - radius, radius), radius, 255, -1, cv2.LINE_AA)
    cv2.circle(mask, (radius, height - radius), radius, 255, -1, cv2.LINE_AA)
    cv2.circle(mask, (width - radius, height - radius), radius, 255, -1, cv2.LINE_AA)
    return mask


def overlay_image_with_mask(frame, overlay, x, y, mask):
    h, w = overlay.shape[:2]
    fh, fw = frame.shape[:2]

    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(fw, x + w)
    y2 = min(fh, y + h)

    if x1 >= x2 or y1 >= y2:
        return

    ox1 = x1 - x
    oy1 = y1 - y
    ox2 = ox1 + (x2 - x1)
    oy2 = oy1 + (y2 - y1)

    roi = frame[y1:y2, x1:x2]
    src = overlay[oy1:oy2, ox1:ox2]
    m = mask[oy1:oy2, ox1:ox2].astype(np.float32) / 255.0

    roi[:] = (src * m[..., None] + roi * (1.0 - m[..., None])).astype(np.uint8)


def draw_glow_points(glow_layer, points, colors, radii):
    for (px, py), color, radius in zip(points, colors, radii):
        cv2.circle(glow_layer, (int(px), int(py)), int(radius), color, -1, cv2.LINE_AA)


def get_vignette(height, width):
    y = np.linspace(-1.0, 1.0, height, dtype=np.float32)[:, None]
    x = np.linspace(-1.0, 1.0, width, dtype=np.float32)[None, :]
    r = np.sqrt(x * x + y * y)
    v = 1.0 - np.clip((r - 0.05) / 1.25, 0.0, 1.0)
    return v**1.6


def create_card_assets(images):
    cards = []
    for i, img in enumerate(images):
        h, w = img.shape[:2]
        target_w = 320
        scale = target_w / max(1, w)
        target_h = int(h * scale)
        base = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)

        cards.append(
            {
                "base": base,
                "caption": DEFAULT_CAPTIONS[i % len(DEFAULT_CAPTIONS)],
                "base_angle": (2.0 * math.pi * i) / max(1, len(images)),
                "radius": 240 + 24 * (i % 3),
                "lift": -60 + 20 * ((i % 5) - 2),
                "blast_offset": np.zeros(2, dtype=np.float32),
                "blast_vel": np.zeros(2, dtype=np.float32),
            }
        )
    return cards


# MediaPipe hand skeleton connections (pairs of landmark indices)
HAND_CONNECTIONS = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),  # thumb
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),  # index
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),  # middle
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),  # ring
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),  # pinky
    (5, 9),
    (9, 13),
    (13, 17),  # palm
]


def draw_hand_landmarks(frame, landmarks, w, h):
    """Draw hand skeleton and landmark dots on the frame."""
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

    # Draw connections
    for i, j in HAND_CONNECTIONS:
        cv2.line(frame, pts[i], pts[j], (180, 140, 255), 2, cv2.LINE_AA)

    # Draw landmark dots
    for px, py in pts:
        cv2.circle(frame, (px, py), 5, (100, 255, 200), -1, cv2.LINE_AA)
        cv2.circle(frame, (px, py), 5, (255, 255, 255), 1, cv2.LINE_AA)


HAND_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)


def is_open_palm(landmarks):
    """Check if the hand is an open palm.  `landmarks` is a list of NormalizedLandmark."""
    tip_ids = [8, 12, 16, 20]
    pip_ids = [6, 10, 14, 18]
    extended = 0
    for t, p in zip(tip_ids, pip_ids):
        if landmarks[t].y < landmarks[p].y:
            extended += 1
    return extended >= 4


def _get_model_path():
    """Download the hand_landmarker.task model if needed and return its path."""
    cache_dir = Path(tempfile.gettempdir()) / "mediapipe_models"
    cache_dir.mkdir(exist_ok=True)
    model_path = cache_dir / "hand_landmarker.task"
    if not model_path.exists():
        print("Downloading hand_landmarker model …")
        urllib.request.urlretrieve(HAND_LANDMARKER_URL, str(model_path))
    return str(model_path)


def create_hand_landmarker():
    """Create a HandLandmarker using the new MediaPipe Tasks API."""
    from mediapipe.tasks.python import vision as mp_vision
    from mediapipe.tasks.python import BaseOptions

    model_path = _get_model_path()
    options = mp_vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6,
    )
    return mp_vision.HandLandmarker.create_from_options(options)


def main():
    script_dir = Path(__file__).resolve().parent
    cwd_mem = Path("./memories")
    script_mem = script_dir / "memories"

    if cwd_mem.exists():
        memories_dir = cwd_mem
    else:
        memories_dir = script_mem

    raw_images = load_images(memories_dir)
    cards = create_card_assets(raw_images[:10])

    rng = np.random.default_rng(7)
    p_count = 420
    p_radius = rng.uniform(90.0, 520.0, size=p_count).astype(np.float32)
    p_angle = rng.uniform(0.0, 2.0 * math.pi, size=p_count).astype(np.float32)
    p_speed = rng.uniform(-0.12, 0.12, size=p_count).astype(np.float32)
    p_size = rng.uniform(1.0, 2.8, size=p_count).astype(np.float32)
    p_brightness = rng.uniform(0.45, 1.0, size=p_count).astype(np.float32)
    p_blast_offset = np.zeros((p_count, 2), dtype=np.float32)
    p_blast_vel = np.zeros((p_count, 2), dtype=np.float32)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Failed to open webcam")
        return

    window_name = "Valentine Memory Galaxy"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    try:
        cv2.setWindowProperty(
            window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
        )
    except Exception:
        pass

    landmarker = create_hand_landmarker()
    start_time_ms = int(time.time() * 1000)

    rotation_angle = 0.0
    rotation_vel = 0.0
    zoom = 1.0
    zoom_target = 1.0
    prev_wrist = None

    pinch_active = False
    blast_factor = 0.0

    letter_char_progress = 0.0

    last_time = time.time()

    vignette_cache = None
    vignette_size = None

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        center = np.array([w * 0.5, h * 0.5], dtype=np.float32)

        if vignette_cache is None or vignette_size != (h, w):
            vignette_cache = get_vignette(h, w)
            vignette_size = (h, w)

        now = time.time()
        dt = max(1e-3, now - last_time)
        last_time = now
        t = now

        # --- Run hand detection on the actual webcam image ---
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int(time.time() * 1000) - start_time_ms
        results = landmarker.detect_for_video(mp_image, timestamp_ms)

        # --- Build dark galaxy background for display ---
        background = np.zeros_like(frame)
        bg_base = np.array([14, 8, 18], dtype=np.float32)
        bg_tint = np.array([8, 10, 20], dtype=np.float32)
        wave = 0.5 + 0.5 * math.sin(t * 0.35)
        base = bg_base + bg_tint * wave
        background[:] = base.astype(np.uint8)

        vignette_rgb = np.dstack([vignette_cache, vignette_cache, vignette_cache])
        frame = np.clip(
            background.astype(np.float32) * vignette_rgb + 2.0, 0, 255
        ).astype(np.uint8)

        open_palm = False
        hand_found = False

        if results.hand_landmarks:
            hand_found = True
            lm = results.hand_landmarks[0]

            wrist = np.array([lm[0].x, lm[0].y], dtype=np.float32)
            mid_mcp = np.array([lm[9].x, lm[9].y], dtype=np.float32)
            thumb_tip = np.array([lm[4].x, lm[4].y], dtype=np.float32)
            index_tip = np.array([lm[8].x, lm[8].y], dtype=np.float32)

            if prev_wrist is not None:
                delta = wrist - prev_wrist
                rotation_vel += float(delta[0] * 3.2 + delta[1] * 0.9)
                rotation_vel = clamp(rotation_vel, -0.22, 0.22)
            prev_wrist = wrist

            d = float(np.linalg.norm(wrist - mid_mcp))
            zoom_target = clamp(1.7 - d * 7.5, 0.75, 1.45)

            pinch_dist = float(np.linalg.norm(thumb_tip - index_tip))
            pinch_active = pinch_dist < PINCH_NORM_THRESH

            open_palm = is_open_palm(lm)
        else:
            prev_wrist = None
            pinch_active = False

        if not hand_found:
            rotation_vel *= 0.94

        rotation_vel *= 0.93
        rotation_angle += rotation_vel

        zoom = lerp(zoom, zoom_target, 0.12)

        blast_target = 1.0 if pinch_active else 0.0
        blast_factor = lerp(blast_factor, blast_target, 0.2 if pinch_active else 0.08)

        glow = np.zeros_like(frame)

        p_angle_curr = p_angle + (t * p_speed) + rotation_angle * 0.4
        px = center[0] + np.cos(p_angle_curr) * p_radius * zoom
        py = center[1] + np.sin(p_angle_curr * 1.25) * p_radius * 0.52 * zoom
        base_positions = np.stack([px, py], axis=1)

        if pinch_active:
            dirs = base_positions - center[None, :]
            norms = np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-5
            dirs /= norms
            impulse = (0.65 + 2.4 * blast_factor) * (0.9 + 0.25 * p_brightness[:, None])
            p_blast_vel += dirs * impulse

        p_blast_vel *= 0.9
        p_blast_offset += p_blast_vel

        if not pinch_active:
            p_blast_offset *= 0.9

        p_positions = base_positions + p_blast_offset

        p_colors = []
        p_radii = []
        p_points = []
        for i in range(p_count):
            if 0 <= p_positions[i, 0] < w and 0 <= p_positions[i, 1] < h:
                b = p_brightness[i]
                if pinch_active:
                    color = (
                        int(40 + 120 * b),
                        int(80 + 130 * b),
                        int(210 + 45 * b),
                    )
                else:
                    color = (
                        int(40 + 70 * b),
                        int(50 + 80 * b),
                        int(120 + 125 * b),
                    )
                p_points.append((p_positions[i, 0], p_positions[i, 1]))
                p_colors.append(color)
                p_radii.append(p_size[i] + 2.2 * blast_factor)

        draw_glow_points(glow, p_points, p_colors, p_radii)

        breathe = 1.0 + 0.14 * math.sin(t * 2.3)
        heart_scale = (65.0 + 25.0 * blast_factor) * breathe * zoom
        core_count = 140
        core_points = []
        core_colors = []
        core_radii = []

        jitter = np.zeros(2, dtype=np.float32)
        if pinch_active:
            jitter = rng.uniform(-6.0, 6.0, size=2).astype(np.float32) * blast_factor

        core_center = center + jitter

        for i in range(core_count):
            u = (i / core_count) * 2.0 * math.pi
            hx = 16.0 * (math.sin(u) ** 3)
            hy = (
                13.0 * math.cos(u)
                - 5.0 * math.cos(2 * u)
                - 2.0 * math.cos(3 * u)
                - math.cos(4 * u)
            )
            px_i = core_center[0] + hx * heart_scale * 0.06
            py_i = core_center[1] - hy * heart_scale * 0.06
            px_i += rng.uniform(-2.0, 2.0)
            py_i += rng.uniform(-2.0, 2.0)

            if pinch_active:
                c = (40, 60, 255)
            else:
                c = (115, 85, 255)

            core_points.append((px_i, py_i))
            core_colors.append(c)
            core_radii.append(2.0 + 2.8 * breathe + 2.5 * blast_factor)

        draw_glow_points(glow, core_points, core_colors, core_radii)

        ring_color = (40, 170, 255) if pinch_active else (100, 90, 210)
        ring_r = int((90 + 45 * breathe) * zoom + 55 * blast_factor)
        cv2.circle(
            glow,
            (int(core_center[0]), int(core_center[1])),
            ring_r,
            ring_color,
            2,
            cv2.LINE_AA,
        )

        glow_blur = cv2.GaussianBlur(glow, (0, 0), sigmaX=8.0, sigmaY=8.0)
        frame = cv2.addWeighted(frame, 1.0, glow_blur, 0.95, 0.0)

        card_draw_list = []
        for i, card in enumerate(cards):
            ang = (
                card["base_angle"]
                + rotation_angle
                + (0.06 * t if i % 2 == 0 else -0.05 * t)
            )
            depth = (math.sin(ang) + 1.0) * 0.5
            scale = (0.62 + 0.52 * depth) * zoom
            radius = card["radius"] * (0.95 + 0.25 * (1.0 - depth)) * zoom

            bx = center[0] + math.cos(ang) * radius
            by = center[1] + card["lift"] + math.sin(ang * 1.1) * 70.0
            base_pos = np.array([bx, by], dtype=np.float32)

            if pinch_active:
                direction = base_pos - center
                nrm = np.linalg.norm(direction) + 1e-5
                direction = direction / nrm
                card["blast_vel"] += direction * (1.0 + 3.2 * blast_factor)

            card["blast_vel"] *= 0.87
            card["blast_offset"] += card["blast_vel"]

            if not pinch_active:
                card["blast_offset"] = lerp(
                    card["blast_offset"], np.zeros(2, dtype=np.float32), 0.12
                )

            pos = base_pos + card["blast_offset"]
            brightness = 0.8 + 0.35 * depth
            card_draw_list.append((depth, pos, scale, brightness, card))

        card_draw_list.sort(key=lambda x: x[0])

        for depth, pos, scale, brightness, card in card_draw_list:
            base = card["base"]
            bw, bh = base.shape[1], base.shape[0]
            tw = int(clamp(bw * scale, 110, w * 0.45))
            th = int(clamp(bh * scale, 80, h * 0.45))

            if tw < 8 or th < 8:
                continue

            card_img = cv2.resize(base, (tw, th), interpolation=cv2.INTER_LINEAR)
            card_img = np.clip(card_img.astype(np.float32) * brightness, 0, 255).astype(
                np.uint8
            )

            mask = make_rounded_card_mask(tw, th, radius=int(min(tw, th) * 0.08))

            x = int(pos[0] - tw // 2)
            y = int(pos[1] - th // 2)

            glow_box = np.zeros_like(frame)
            cv2.rectangle(
                glow_box,
                (x - 6, y - 6),
                (x + tw + 6, y + th + 6),
                (100, 70, 255),
                -1,
                cv2.LINE_AA,
            )
            glow_box = cv2.GaussianBlur(glow_box, (0, 0), 10)
            frame = cv2.addWeighted(frame, 1.0, glow_box, 0.23 + 0.2 * depth, 0)

            overlay_image_with_mask(frame, card_img, x, y, mask)

            border = np.zeros((th, tw, 3), dtype=np.uint8)
            cv2.rectangle(
                border, (0, 0), (tw - 1, th - 1), (235, 235, 245), 2, cv2.LINE_AA
            )
            border_mask = make_rounded_card_mask(tw, th, radius=int(min(tw, th) * 0.08))
            overlay_image_with_mask(frame, border, x, y, border_mask)

            cap_text = card["caption"]
            tx = int(pos[0] - tw // 2)
            ty = int(y + th + 24)
            if ty < h - 5:
                cv2.putText(
                    frame,
                    cap_text,
                    (tx + 1, ty + 1),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.56,
                    (15, 15, 20),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    frame,
                    cap_text,
                    (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.56,
                    (240, 236, 245),
                    1,
                    cv2.LINE_AA,
                )

        if open_palm and blast_factor < 0.18:
            chars_per_sec = 34.0
            total_chars = sum(len(line) for line in LOVE_LETTER) + len(LOVE_LETTER) - 1
            letter_char_progress = clamp(
                letter_char_progress + dt * chars_per_sec, 0.0, float(total_chars)
            )

            panel_w = int(w * 0.52)
            panel_h = int(h * 0.33)
            px = int(w * 0.05)
            py = int(h * 0.58)

            panel = frame.copy()
            cv2.rectangle(
                panel,
                (px, py),
                (px + panel_w, py + panel_h),
                (26, 20, 42),
                -1,
                cv2.LINE_AA,
            )
            frame = cv2.addWeighted(frame, 0.7, panel, 0.3, 0.0)
            cv2.rectangle(
                frame,
                (px, py),
                (px + panel_w, py + panel_h),
                (180, 150, 240),
                1,
                cv2.LINE_AA,
            )

            remaining = int(letter_char_progress)
            y_cursor = py + 38
            for line in LOVE_LETTER:
                reveal = min(len(line), remaining)
                show = line[:reveal]
                remaining -= len(line) + 1
                cv2.putText(
                    frame,
                    show,
                    (px + 20, y_cursor),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.62,
                    (236, 228, 248),
                    1,
                    cv2.LINE_AA,
                )
                y_cursor += 30
        else:
            letter_char_progress = max(0.0, letter_char_progress - dt * 22.0)

        # --- Draw hand tracking overlay ---
        if results.hand_landmarks:
            draw_hand_landmarks(frame, results.hand_landmarks[0], w, h)

        # --- Hand status indicator ---
        if hand_found:
            status = (
                "PINCH" if pinch_active else ("OPEN PALM" if open_palm else "TRACKING")
            )
            status_color = (
                (0, 200, 255)
                if pinch_active
                else ((0, 255, 180) if open_palm else (200, 200, 200))
            )
            cv2.putText(
                frame,
                status,
                (w - 200, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 0),
                3,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                status,
                (w - 200, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                status_color,
                2,
                cv2.LINE_AA,
            )

        cv2.putText(
            frame,
            HELP_TEXT,
            (16, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (10, 10, 10),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            HELP_TEXT,
            (16, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (240, 235, 245),
            1,
            cv2.LINE_AA,
        )

        cv2.imshow(window_name, frame)
        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), 27):
            break
        if key == ord("r"):
            rotation_angle = 0.0
            rotation_vel = 0.0
            zoom = 1.0
            zoom_target = 1.0
            blast_factor = 0.0
            pinch_active = False
            prev_wrist = None
            for c in cards:
                c["blast_offset"][:] = 0.0
                c["blast_vel"][:] = 0.0
            p_blast_offset[:] = 0.0
            p_blast_vel[:] = 0.0

    landmarker.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
