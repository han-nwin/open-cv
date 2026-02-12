import math
import os
import time
import random

import cv2
import numpy as np

from hand_detector import (
    HandDetector,
    draw_hand,
    is_pinching,
    hand_centroid,
    hand_apparent_size,
)

CANVAS_W, CANVAS_H = 1280, 720
PIP_W, PIP_H = 256, 144

# Colors (BGR)
PINK = (180, 105, 255)
GOLD = (0, 215, 255)
SOFT_WHITE = (200, 200, 220)
ARROW_COLOR = (200, 220, 255)
BOW_COLOR = (140, 120, 255)
HEART_RED = (80, 80, 255)
STRING_COLOR = (220, 210, 240)

# 3D perspective
FOV_DEG = 70
FOCAL = CANVAS_W / (2 * math.tan(math.radians(FOV_DEG / 2)))
CAM_Y = -2.5  # camera height (negative = above ground y=0)

# Physics (world units)
GRAVITY_3D = 2.5
ARROW_MAX_SPEED = 40.0
PULL_SENSITIVITY = 15.0
AIM_SENSITIVITY = 2.5

YEARS = [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026]
MEMORY_DURATION = 5.0

MEMORY_TEXTS = {
    2018: "where it all began",
    2019: "falling for you, one day at a time",
    2020: "together through everything",
    2021: "growing closer, growing stronger",
    2022: "every moment with you is magic",
    2023: "welcoming an angel into our life",
    2024: "building a family together",
    2025: "you are always my biggest blessing",
    2026: "and the best is yet to come.\n I LOVE YOU, MY VALENTINE",
}

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_memory_img_cache: dict[int, np.ndarray | None] = {}


def _load_memory_image(year):
    """Load and cache a memory photo for the given year."""
    if year in _memory_img_cache:
        return _memory_img_cache[year]
    path = os.path.join(_SCRIPT_DIR, "memories", f"{year}.jpg")
    img = cv2.imread(path)
    if img is None:
        _memory_img_cache[year] = None
        return None
    max_w, max_h = 420, 300
    h, w = img.shape[:2]
    scale = min(max_w / w, max_h / h)
    img = cv2.resize(
        img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA
    )
    _memory_img_cache[year] = img
    return img


# Sky/ground gradient endpoints (BGR, float for lerp)
SKY_TOP = np.array([15, 8, 12], dtype=np.float64)
SKY_BOT = np.array([50, 25, 45], dtype=np.float64)
GROUND_TOP = np.array([35, 18, 30], dtype=np.float64)
GROUND_BOT = np.array([25, 12, 22], dtype=np.float64)

# Draw strength thresholds (hand_apparent_size values)
REST_SIZE = 0.15  # Hand extended (arm straight)
DRAW_SIZE = 0.40  # Hand close to face (fully drawn)


class Stabilizer:
    """Exponential smoothing to reduce mediapipe jitter."""

    def __init__(self, alpha=0.5):
        self.alpha = alpha
        self.prev_val = None

    def update(self, val):
        if self.prev_val is None:
            self.prev_val = val
        self.prev_val = self.prev_val * (1 - self.alpha) + val * self.alpha
        return self.prev_val

    def reset(self):
        self.prev_val = None


def get_draw_strength(landmarks):
    """Returns 0.0 (relaxed) to 1.0 (fully drawn) based on hand size."""
    size = hand_apparent_size(landmarks)
    strength = (size - REST_SIZE) / (DRAW_SIZE - REST_SIZE)
    return max(0.0, min(1.0, strength))


# ---------------------------------------------------------------------------
# 3D helpers
# ---------------------------------------------------------------------------


def project(wx, wy, wz):
    if wz < 0.5:
        return None
    sx = int(FOCAL * wx / wz + CANVAS_W / 2)
    sy = int(FOCAL * (wy - CAM_Y) / wz + CANVAS_H / 2)
    scale = FOCAL / wz
    return sx, sy, scale


def fog_color(color, wz, max_z=55):
    f = max(0.3, 1.0 - wz / max_z)
    return tuple(int(c * f) for c in color)


def draw_small_heart(canvas, cx, cy, size, color):
    pts = []
    for deg in range(0, 360, 8):
        t = math.radians(deg)
        x = 16 * math.sin(t) ** 3
        y = -(
            13 * math.cos(t)
            - 5 * math.cos(2 * t)
            - 2 * math.cos(3 * t)
            - math.cos(4 * t)
        )
        pts.append((int(cx + x * size / 17), int(cy + y * size / 17)))
    if len(pts) > 2:
        cv2.fillPoly(canvas, [np.array(pts)], color, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Arrow flying through 3D space
# ---------------------------------------------------------------------------


class Arrow3D:
    def __init__(self):
        self.active = False
        self.x = self.y = self.z = 0.0
        self.vx = self.vy = self.vz = 0.0
        self.trail: list[tuple[float, float, float]] = []

    def launch(self, dir_x, dir_y, speed):
        self.active = True
        self.z = 2.0
        self.x = dir_x * self.z
        self.y = CAM_Y + dir_y * self.z
        mag = math.sqrt(dir_x**2 + dir_y**2 + 1.0)
        self.vx = dir_x / mag * speed
        self.vy = dir_y / mag * speed
        self.vz = 1.0 / mag * speed
        self.trail.clear()

    def update(self, dt):
        if not self.active:
            return
        self.trail.append((self.x, self.y, self.z))
        if len(self.trail) > 30:
            self.trail.pop(0)
        self.x += self.vx * dt
        self.vy += GRAVITY_3D * dt
        self.y += self.vy * dt
        self.z += self.vz * dt
        if self.z > 60 or self.y > 1 or self.z < 0:
            self.active = False

    def draw(self, canvas):
        if not self.active:
            return
        # Sparkle trail
        n = len(self.trail)
        for i, (tx, ty, tz) in enumerate(self.trail):
            p = project(tx, ty, tz)
            if not p:
                continue
            sx, sy, sc = p
            if not (0 <= sx < CANVAS_W and 0 <= sy < CANVAS_H):
                continue
            a = (i + 1) / n
            f = max(0.3, 1.0 - tz / 55)
            r = max(1, int(3 * a * min(1.0, sc / 60)))
            c = (int(180 * a * f), int((80 * a + 140 * (1 - a)) * f), int(255 * a * f))
            cv2.circle(canvas, (sx, sy), r, c, -1, cv2.LINE_AA)

        # Arrow body from projected 3D points along velocity direction
        speed = math.sqrt(self.vx * self.vx + self.vy * self.vy + self.vz * self.vz)
        if speed < 1e-5:
            return

        nx, ny, nz = self.vx / speed, self.vy / speed, self.vz / speed
        head_p = project(self.x, self.y, self.z)
        tail_len = 2.2
        tail_p = project(
            self.x - nx * tail_len, self.y - ny * tail_len, self.z - nz * tail_len
        )
        if not head_p or not tail_p:
            return

        hx, hy, hsc = head_p
        tx, ty, tsc = tail_p
        if not (-80 <= hx < CANVAS_W + 80 and -80 <= hy < CANVAS_H + 80):
            return

        # Shaft
        avg_z = (self.z + (self.z - nz * tail_len)) * 0.5
        shaft_th = max(1, int(((hsc + tsc) * 0.5) / 48))
        cv2.line(
            canvas,
            (tx, ty),
            (hx, hy),
            fog_color(ARROW_COLOR, avg_z),
            shaft_th,
            cv2.LINE_AA,
        )

        # Tip
        tip_size = max(3, int(hsc / 15))
        draw_small_heart(canvas, hx, hy, tip_size, fog_color(HEART_RED, self.z))

        # Fletching near tail, built from screen-space normal of shaft direction
        dx, dy = hx - tx, hy - ty
        dlen = math.hypot(dx, dy)
        if dlen > 1.0:
            px, py = -dy / dlen, dx / dlen
            bl = max(5, int(tsc / 16))
            bw = max(3, int(tsc / 22))
            c = fog_color(PINK, self.z - nz * tail_len)
            base = (tx, ty)
            feather1 = (
                int(base[0] + px * bw - dx / dlen * bl),
                int(base[1] + py * bw - dy / dlen * bl),
            )
            feather2 = (
                int(base[0] - px * bw - dx / dlen * bl),
                int(base[1] - py * bw - dy / dlen * bl),
            )
            cv2.line(canvas, base, feather1, c, max(1, shaft_th), cv2.LINE_AA)
            cv2.line(canvas, base, feather2, c, max(1, shaft_th), cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Year target in 3D
# ---------------------------------------------------------------------------


class YearTarget3D:
    def __init__(self, year, x, y, z):
        self.year = year
        self.wx, self.wy, self.wz = float(x), float(y), float(z)
        self.radius_w = 1.2
        self.hit = False
        self.glow_t = 0.0
        self.bob_phase = random.uniform(0, math.pi * 2)
        self._cur_y = float(y)

    def update(self, dt, elapsed):
        self._cur_y = self.wy + math.sin(elapsed * 1.2 + self.bob_phase) * 0.3
        if self.hit:
            self.glow_t += dt

    def check_hit(self, ax, ay, az):
        if self.hit:
            return False
        dist = math.sqrt(
            (ax - self.wx) ** 2 + (ay - self._cur_y) ** 2 + (az - self.wz) ** 2
        )
        if dist < self.radius_w + 1.0:
            self.hit = True
            self.glow_t = 0.0
            return True
        return False

    def draw(self, canvas):
        p = project(self.wx, self._cur_y, self.wz)
        if not p:
            return
        sx, sy, sc = p
        r = max(6, int(self.radius_w * sc))
        if sx < -r or sx > CANVAS_W + r or sy < -r or sy > CANVAS_H + r:
            return

        # Heart outline points
        heart_pts = []
        for deg in range(0, 360, 6):
            t = math.radians(deg)
            hx = 16 * math.sin(t) ** 3
            hy = -(
                13 * math.cos(t)
                - 5 * math.cos(2 * t)
                - 2 * math.cos(3 * t)
                - math.cos(4 * t)
            )
            heart_pts.append((int(sx + hx * r / 17), int(sy + hy * r / 17)))
        heart_np = np.array(heart_pts, dtype=np.int32)

        if self.hit:
            # --- Fill up like pouring water ---
            fill_frac = min(1.0, self.glow_t / 1.5)
            ys = [p[1] for p in heart_pts]
            h_min, h_max = min(ys), max(ys)
            wave = math.sin(self.glow_t * 8) * 2 * (1.0 - fill_frac)
            cutoff_y = int(h_max - fill_frac * (h_max - h_min) + wave)

            fill_color = fog_color(HEART_RED, self.wz)
            fill_layer = canvas.copy()
            cv2.fillPoly(fill_layer, [heart_np], fill_color, cv2.LINE_AA)
            # Only keep the filled portion below the cutoff line
            fill_layer[: max(0, cutoff_y)] = canvas[: max(0, cutoff_y)]
            np.copyto(canvas, fill_layer)

            # Soft glow behind once mostly filled
            if fill_frac > 0.5:
                glow_a = 0.2 * min(1.0, (fill_frac - 0.5) * 2)
                glow_r = int(r * 1.3)
                glow_layer = canvas.copy()
                draw_small_heart(glow_layer, sx, sy, glow_r, GOLD)
                cv2.addWeighted(glow_layer, glow_a, canvas, 1 - glow_a, 0, canvas)

            # Outline
            cv2.polylines(
                canvas,
                [heart_np],
                True,
                fog_color(PINK, self.wz),
                max(1, int(r / 10)),
                cv2.LINE_AA,
            )

            # --- Sparkle hearts bursting out ---
            if self.glow_t < 2.5:
                for i in range(8):
                    ang = i * math.pi * 2 / 8 + math.sin(self.glow_t * 3 + i) * 0.3
                    spd = 0.5 + (i % 3) * 0.25
                    dist = self.glow_t * r * spd
                    sp_x = int(sx + math.cos(ang) * dist)
                    sp_y = int(sy + math.sin(ang) * dist - self.glow_t * r * 0.25)
                    life = max(0.0, 1.0 - self.glow_t / 2.5)
                    sp_sz = max(3, int(r * 0.22 * life))
                    sp_c = tuple(int(v * life) for v in fog_color(PINK, self.wz))
                    draw_small_heart(canvas, sp_x, sp_y, sp_sz, sp_c)

            tc = (30, 30, 30)
        else:
            # Transparent dark fill + pink outline
            fill_overlay = canvas.copy()
            cv2.fillPoly(
                fill_overlay, [heart_np], fog_color((40, 20, 50), self.wz), cv2.LINE_AA
            )
            cv2.addWeighted(fill_overlay, 0.3, canvas, 0.7, 0, canvas)
            cv2.polylines(
                canvas,
                [heart_np],
                True,
                fog_color(PINK, self.wz),
                max(1, int(r / 10)),
                cv2.LINE_AA,
            )
            tc = fog_color(SOFT_WHITE, self.wz)

        text = str(self.year)
        font = cv2.FONT_HERSHEY_SIMPLEX
        fs = max(0.3, min(1.2, sc / 100))
        thick = max(1, int(fs * 2))
        (tw, th), _ = cv2.getTextSize(text, font, fs, thick)
        cv2.putText(
            canvas, text, (sx - tw // 2, sy + th // 2), font, fs, tc, thick, cv2.LINE_AA
        )


# ---------------------------------------------------------------------------
# Scene background
# ---------------------------------------------------------------------------

_bg_cache: np.ndarray | None = None


def _build_bg():
    global _bg_cache
    mid = CANVAS_H // 2
    canvas = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)

    # Sky gradient
    t = np.linspace(0, 1, mid).reshape(-1, 1, 1)
    sky = (SKY_TOP * (1 - t) + SKY_BOT * t).astype(np.uint8)
    canvas[:mid] = np.broadcast_to(sky, (mid, CANVAS_W, 3))

    # Ground gradient
    gh = CANVAS_H - mid
    t = np.linspace(0, 1, gh).reshape(-1, 1, 1)
    ground = (GROUND_TOP * (1 - t) + GROUND_BOT * t).astype(np.uint8)
    canvas[mid:] = np.broadcast_to(ground, (gh, CANVAS_W, 3))

    _bg_cache = canvas


_stars_cache: list | None = None


def draw_scene(canvas, elapsed):
    """Draw sky + ground + stars + grid."""
    if _bg_cache is None:
        _build_bg()
    np.copyto(canvas, _bg_cache)

    # Stars
    global _stars_cache
    if _stars_cache is None:
        _stars_cache = [
            (
                random.randint(0, CANVAS_W),
                random.randint(0, CANVAS_H // 2 - 10),
                random.uniform(0, math.pi * 2),
            )
            for _ in range(120)
        ]
    for sx, sy, phase in _stars_cache:
        br = int(30 + 22 * math.sin(elapsed * 1.5 + phase * 3))
        cv2.circle(canvas, (sx, sy), 1, (br, br, br + 12), -1)

    # Ground grid — horizontal depth lines
    horizon_y = CANVAS_H // 2
    for z in range(4, 55, 3):
        p = project(0, 0, z)
        if p and horizon_y <= p[1] < CANVAS_H:
            fog = max(25, int(65 * (1.0 - z / 55)))
            cv2.line(
                canvas,
                (0, p[1]),
                (CANVAS_W, p[1]),
                (fog, fog // 2, fog + 8),
                1,
                cv2.LINE_AA,
            )

    # Radial lines from vanishing point
    for i in range(-8, 9):
        p = project(i * 6, 0, 50)
        if p:
            cv2.line(
                canvas,
                (CANVAS_W // 2, horizon_y),
                (p[0], CANVAS_H),
                (40, 28, 48),
                1,
                cv2.LINE_AA,
            )


# ---------------------------------------------------------------------------
# FPS bow at bottom of screen
# ---------------------------------------------------------------------------


def _norm2(dx, dy):
    mag = math.hypot(dx, dy)
    if mag < 1e-5:
        return 1.0, 0.0, 1.0
    return dx / mag, dy / mag, mag


def _normalize2(dx, dy):
    nx, ny, _ = _norm2(dx, dy)
    return nx, ny


def _choose_perp_sign(forward, grip):
    px, py = _normalize2(forward[1], -forward[0])
    cx, cy = CANVAS_W * 0.5 - grip[0], CANVAS_H * 0.5 - grip[1]
    if px * cx + py * cy < 0:
        px, py = -px, -py
    return (px, py)


def _fps_local_to_world(grip, perp, forward, lx, ly):
    return (
        int(grip[0] + perp[0] * lx + forward[0] * ly),
        int(grip[1] + perp[1] * lx + forward[1] * ly),
    )


def _draw_tapered_arrow(canvas, nock_xy, tip_xy, perp, near_w, far_w):
    nock_x, nock_y = nock_xy
    tip_x, tip_y = tip_xy
    shaft = np.array(
        [
            (int(nock_x + perp[0] * near_w), int(nock_y + perp[1] * near_w)),
            (int(nock_x - perp[0] * near_w), int(nock_y - perp[1] * near_w)),
            (int(tip_x - perp[0] * far_w), int(tip_y - perp[1] * far_w)),
            (int(tip_x + perp[0] * far_w), int(tip_y + perp[1] * far_w)),
        ],
        dtype=np.int32,
    )
    cv2.fillConvexPoly(canvas, shaft, (188, 206, 244), cv2.LINE_AA)
    cv2.polylines(canvas, [shaft], True, (116, 132, 176), 1, cv2.LINE_AA)
    cv2.line(
        canvas,
        (int(nock_x + perp[0] * near_w * 0.35), int(nock_y + perp[1] * near_w * 0.35)),
        (int(tip_x + perp[0] * 0.6), int(tip_y + perp[1] * 0.6)),
        (246, 248, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.line(
        canvas,
        (int(nock_x - perp[0] * near_w * 0.55), int(nock_y - perp[1] * near_w * 0.55)),
        (int(tip_x - perp[0] * 0.9), int(tip_y - perp[1] * 0.9)),
        (90, 98, 136),
        2,
        cv2.LINE_AA,
    )


def draw_fps_bow(canvas, power, nocked, aim_x, aim_y, bow_anchor_px=None):
    elapsed = time.time()

    par_x = int((aim_x - 0.5) * 16)
    par_y = int((aim_y - 0.5) * 12)
    grip = (int(CANVAS_W * 0.82 + par_x), int(CANVAS_H * 0.66 + par_y))
    aim_pt = (aim_x * CANVAS_W, aim_y * CANVAS_H)
    forward = _normalize2(aim_pt[0] - grip[0], aim_pt[1] - grip[1])
    perp = _choose_perp_sign(forward, grip)

    # Top-down FPS bow: limbs span along perp (left-right),
    # curve bulges along forward (toward target). Rotates with aim.
    half_w = 200
    steps = 46
    limb_pts = []
    for i in range(steps):
        t = (i / (steps - 1)) * 2.0 - 1.0
        at = abs(t)
        local_y = t * half_w  # limb span along perp (left-right)
        local_x = 84 * (1.0 - t * t) + 26 * max(0.0, (at - 0.72) / 0.28) ** 1.6
        limb_pts.append(_fps_local_to_world(grip, forward, perp, local_x, local_y))

    left_tip = limb_pts[0]
    right_tip = limb_pts[-1]
    gp = limb_pts[steps // 2]

    # Everything anchored to gp (bow rest) along the forward/aim axis.
    rest_x, rest_y = float(gp[0]), float(gp[1])
    pull = (power * 140.0) if nocked else 0.0
    vib = (math.sin(elapsed * 42.0) * (0.8 + 1.6 * power)) if nocked else 0.0

    # Nock sits behind the rest along -forward; pull stretches it further back.
    base_nock_dist = 40.0
    nock_x = rest_x - forward[0] * (base_nock_dist + pull) + forward[0] * vib
    nock_y = rest_y - forward[1] * (base_nock_dist + pull) + forward[1] * vib
    nock_ix, nock_iy = int(nock_x), int(nock_y)

    # Arrow tip extends past the bow along +forward so the heart stays visible.
    tip_x = rest_x + forward[0] * 28
    tip_y = rest_y + forward[1] * 28

    # String from limb tips through nock (curved via control points).
    c1 = (
        int((left_tip[0] + nock_ix) * 0.5 + forward[0] * vib),
        int((left_tip[1] + nock_iy) * 0.5 + forward[1] * vib),
    )
    c2 = (
        int((right_tip[0] + nock_ix) * 0.5 - forward[0] * vib),
        int((right_tip[1] + nock_iy) * 0.5 - forward[1] * vib),
    )
    cv2.polylines(
        canvas,
        [np.array([left_tip, c1, (nock_ix, nock_iy)], dtype=np.int32)],
        False,
        STRING_COLOR,
        2,
        cv2.LINE_AA,
    )
    cv2.polylines(
        canvas,
        [np.array([(nock_ix, nock_iy), c2, right_tip], dtype=np.int32)],
        False,
        STRING_COLOR,
        2,
        cv2.LINE_AA,
    )

    # Arrow aligned with forward (parallel to line of sight).
    near_w = 4 + int(2 * power)
    _draw_tapered_arrow(canvas, (nock_x, nock_y), (tip_x, tip_y), perp, near_w, 1)
    draw_small_heart(canvas, int(tip_x), int(tip_y), 5, HEART_RED)

    # Bow limbs above string/arrow.
    for i in range(steps - 1):
        frac = abs(i - steps // 2) / (steps // 2)
        th = max(2, int(11 - 6 * frac))
        c = tuple(int(v * (1.0 - 0.24 * frac)) for v in BOW_COLOR)
        cv2.line(canvas, limb_pts[i], limb_pts[i + 1], c, th, cv2.LINE_AA)

    # Grip: narrow rectangular wrap, not a fat oval.
    grip_hw, grip_hh = 6, 20  # slim along forward, longer along perp
    grip_corners = []
    for sx_sign, sy_sign in [(-1, -1), (1, -1), (1, 1), (-1, 1)]:
        grip_corners.append(
            _fps_local_to_world(gp, forward, perp, sx_sign * grip_hw, sy_sign * grip_hh)
        )
    grip_poly = np.array(grip_corners, dtype=np.int32)
    cv2.fillConvexPoly(canvas, grip_poly, (76, 56, 118), cv2.LINE_AA)
    cv2.polylines(canvas, [grip_poly], True, (124, 98, 172), 2, cv2.LINE_AA)

    # Fingers wrapping around the grip (3 thin bands across the handle).
    hand_overlay = canvas.copy()
    for j in range(3):
        off = -10 + j * 10  # spread along perp
        fa = _fps_local_to_world(gp, forward, perp, -8, off)
        fb = _fps_local_to_world(gp, forward, perp, 8, off)
        cv2.line(hand_overlay, fa, fb, (160, 132, 190), 4, cv2.LINE_AA)
        cv2.line(hand_overlay, fa, fb, (140, 112, 170), 2, cv2.LINE_AA)
    cv2.addWeighted(hand_overlay, 0.5, canvas, 0.5, 0, canvas)

    cv2.circle(canvas, (nock_ix, nock_iy), 2, (236, 228, 248), -1, cv2.LINE_AA)
    if power > 0.55 and nocked:
        glow = np.zeros_like(canvas)
        cv2.circle(
            glow,
            (nock_ix, nock_iy),
            int(14 + power * 30),
            (95, 72, 225),
            -1,
            cv2.LINE_AA,
        )
        glow = cv2.GaussianBlur(glow, (0, 0), 5 + power * 4)
        cv2.addWeighted(glow, 0.14 + 0.20 * power, canvas, 1.0, 0, canvas)


# ---------------------------------------------------------------------------
# HUD elements
# ---------------------------------------------------------------------------


def draw_crosshair(canvas, sx, sy):
    size, gap = 18, 5
    color = (180, 160, 220)
    cv2.line(canvas, (sx - size, sy), (sx - gap, sy), color, 1, cv2.LINE_AA)
    cv2.line(canvas, (sx + gap, sy), (sx + size, sy), color, 1, cv2.LINE_AA)
    cv2.line(canvas, (sx, sy - size), (sx, sy - gap), color, 1, cv2.LINE_AA)
    cv2.line(canvas, (sx, sy + gap), (sx, sy + size), color, 1, cv2.LINE_AA)
    cv2.circle(canvas, (sx, sy), 2, color, -1, cv2.LINE_AA)


def draw_bow_hud(frame, landmarks, w, h, strength):
    """Draw bow string overlay on the webcam PiP instead of hand skeleton."""
    cx, cy = hand_centroid(landmarks)
    aim_px, aim_py = int(cx * w), int(cy * h)

    # Bow handle anchored to bottom-left corner
    bow_anchor = (0, h)

    # Dynamic color: green (relaxed) -> red (tension)
    color = (0, int(255 * (1 - strength)), int(255 * strength))

    if is_pinching(landmarks):
        # String from anchor to hand
        cv2.line(frame, bow_anchor, (aim_px, aim_py), color, 3, cv2.LINE_AA)

        # Nock point circle grows with tension
        radius = int(5 + strength * 10)
        cv2.circle(frame, (aim_px, aim_py), radius, color, 2, cv2.LINE_AA)

        # Crosshair when fully drawn
        if strength > 0.9:
            mx, my = w // 2, h // 2
            cv2.line(frame, (mx - 10, my), (mx + 10, my), (0, 255, 255), 2)
            cv2.line(frame, (mx, my - 10), (mx, my + 10), (0, 255, 255), 2)
    else:
        # Hand open — show aim cursor only
        cv2.circle(frame, (aim_px, aim_py), 5, (200, 200, 200), -1)


def draw_power_meter(canvas, power):
    x, y, w, h = 40, CANVAS_H - 100, 18, 140
    cv2.rectangle(canvas, (x - 1, y - h - 1), (x + w + 1, y + 1), (40, 40, 40), -1)
    cv2.rectangle(canvas, (x, y - h), (x + w, y), (80, 80, 80), 1)
    fh = int(h * power)
    if fh > 0:
        c = (int(180 - 120 * power), int(100 - 40 * power), 255)
        cv2.rectangle(canvas, (x, y - fh), (x + w, y), c, -1)
    cv2.putText(
        canvas,
        "PWR",
        (x - 4, y + 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.35,
        SOFT_WHITE,
        1,
        cv2.LINE_AA,
    )


def draw_memory_overlay(canvas, year, timer):
    # Fade envelope: in 0-0.8s, hold, out last 0.8s
    if timer < 0.8:
        fade = timer / 0.8
    elif timer > MEMORY_DURATION - 0.8:
        fade = max(0.0, (MEMORY_DURATION - timer) / 0.8)
    else:
        fade = 1.0

    # Dreamy tint
    tint = np.zeros_like(canvas)
    for row in range(CANVAS_H):
        r = row / CANVAS_H
        tint[row, :] = (int(30 + 25 * r), int(10 + 15 * r), int(50 + 45 * r))
    cv2.addWeighted(
        tint, min(0.8, fade * 0.8), canvas, 1 - min(0.8, fade * 0.8), 0, canvas
    )

    if fade < 0.05:
        return

    font = cv2.FONT_HERSHEY_SIMPLEX
    script = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX
    cx = CANVAS_W // 2
    img = _load_memory_image(year)

    if img is not None:
        ih, iw = img.shape[:2]
        pad_s, pad_t, pad_b = 14, 14, 62

        # Polaroid frame centered, nudged up
        fw, fh = iw + pad_s * 2, ih + pad_t + pad_b
        fx = cx - fw // 2
        fy = CANVAS_H // 2 - fh // 2 - 25

        # Soft shadow
        cv2.rectangle(
            canvas, (fx + 5, fy + 5), (fx + fw + 5, fy + fh + 5), (12, 8, 18), -1
        )
        # White frame
        cv2.rectangle(canvas, (fx, fy), (fx + fw, fy + fh), (248, 244, 252), -1)
        # Thin pink border
        cv2.rectangle(canvas, (fx, fy), (fx + fw, fy + fh), (200, 170, 230), 2)

        # Photo blended with fade
        px, py = fx + pad_s, fy + pad_t
        roi = canvas[py : py + ih, px : px + iw]
        canvas[py : py + ih, px : px + iw] = cv2.addWeighted(
            img, fade, roi, 1 - fade, 0
        )

        # Small heart accent in corner of polaroid
        draw_small_heart(canvas, fx + fw - 22, fy + 20, 8, HEART_RED)

        # Year inside polaroid bottom strip
        yr_text = str(year)
        (yw, yh), _ = cv2.getTextSize(yr_text, script, 1.4, 2)
        yr_x = fx + fw // 2 - yw // 2
        yr_y = fy + pad_t + ih + (pad_b + yh) // 2
        cv2.putText(
            canvas,
            yr_text,
            (yr_x + 1, yr_y + 1),
            script,
            1.4,
            (180, 150, 210),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas, yr_text, (yr_x, yr_y), script, 1.4, (60, 40, 100), 2, cv2.LINE_AA
        )

        # Caption below polaroid (supports \n for multi-line)
        caption = MEMORY_TEXTS.get(year, f"memories of {year}")
        lines = caption.split("\n")
        line_y = fy + fh + 40
        for li, line_text in enumerate(lines):
            cap = f"~ {line_text.strip()} ~" if li == 0 else line_text.strip()
            (cw, ch), _ = cv2.getTextSize(cap, font, 0.65, 2)
            cap_x = cx - cw // 2
            cv2.putText(canvas, cap, (cap_x + 1, line_y + 1),
                        font, 0.65, (50, 25, 60), 3, cv2.LINE_AA)
            cv2.putText(canvas, cap, (cap_x, line_y),
                        font, 0.65, PINK, 2, cv2.LINE_AA)
            if li == 0:
                draw_small_heart(canvas, cap_x - 20, line_y - 6, 7, HEART_RED)
                draw_small_heart(canvas, cap_x + cw + 20, line_y - 6, 7, HEART_RED)
            line_y += ch + 16
    else:
        # No image — large year + caption text
        cy = CANVAS_H // 2 - 30
        yr_text = str(year)
        (tw, th), _ = cv2.getTextSize(yr_text, font, 3.0, 4)
        for rr in range(3, 0, -1):
            gc = tuple(int(v * rr / 3) for v in (100, 80, 200))
            cv2.putText(
                canvas,
                yr_text,
                (cx - tw // 2 + rr, cy + th // 2 + rr),
                font,
                3.0,
                gc,
                4 + rr * 2,
                cv2.LINE_AA,
            )
        cv2.putText(
            canvas,
            yr_text,
            (cx - tw // 2, cy + th // 2),
            font,
            3.0,
            GOLD,
            4,
            cv2.LINE_AA,
        )

        caption = MEMORY_TEXTS.get(year, f"memories of {year}")
        lines = caption.split("\n")
        line_y = cy + th // 2 + 65
        for li, line_text in enumerate(lines):
            cap = f"~ {line_text.strip()} ~" if li == 0 else line_text.strip()
            (sw, sh), _ = cv2.getTextSize(cap, font, 0.8, 2)
            cv2.putText(canvas, cap, (cx - sw // 2, line_y),
                        font, 0.8, PINK, 2, cv2.LINE_AA)
            line_y += sh + 18

    # Floating hearts around the edges
    for i in range(10):
        hx = int(CANVAS_W * 0.05 + (i * 137) % int(CANVAS_W * 0.90))
        bob = math.sin(timer * 2.0 + i * 0.7) * 30
        hy = int(CANVAS_H * 0.10 + bob) if i < 5 else int(CANVAS_H * 0.85 + bob)
        sz = 9 + int(math.sin(timer * 1.5 + i) * 4)
        ha = fade * (0.5 + 0.5 * math.sin(timer * 2 + i))
        hc = (int(70 * ha), int(50 * ha), int(220 * ha))
        draw_small_heart(canvas, hx, hy, sz, hc)


# ---------------------------------------------------------------------------
# Hit particles in 3D
# ---------------------------------------------------------------------------


class HitParticles3D:
    def __init__(self):
        self.particles: list[list] = []

    def spawn(self, wx, wy, wz, count=15):
        for _ in range(count):
            ang = random.uniform(0, math.pi * 2)
            spd = random.uniform(2, 5)
            self.particles.append(
                [
                    wx,
                    wy,
                    wz,
                    spd * math.cos(ang),
                    random.uniform(-3, 0),
                    spd * math.sin(ang),
                    1.0,
                ]
            )

    def update(self, dt):
        alive = []
        for p in self.particles:
            p[0] += p[3] * dt
            p[4] += 4.0 * dt
            p[1] += p[4] * dt
            p[2] += p[5] * dt
            p[6] -= dt * 0.5
            if p[6] > 0:
                alive.append(p)
        self.particles = alive

    def draw(self, canvas):
        for p in self.particles:
            pr = project(p[0], p[1], p[2])
            if not pr:
                continue
            sx, sy, sc = pr
            if not (0 <= sx < CANVAS_W and 0 <= sy < CANVAS_H):
                continue
            life = p[6]
            a = max(0.0, min(1.0, life))
            size = max(3, int(5 * a * min(1.0, sc / 40)))
            c = (int(80 * a), int(80 * a), int(255 * a))
            draw_small_heart(canvas, sx, sy, size, c)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    detector = HandDetector()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Failed to open webcam")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    start_ms = int(time.time() * 1000)
    prev_time = time.time()

    arrows: list[Arrow3D] = []
    particles = HitParticles3D()
    pinch_prev = False
    nocked = False
    power = 0.0
    aim_x, aim_y = 0.5, 0.5
    _init_hand_size = 0.0
    _init_centroid = (0.0, 0.0)
    _aim_at_start = (0.5, 0.5)
    bow_anchor_uv = (0.70, 0.62)
    bow_anchor_px = (int(bow_anchor_uv[0] * CANVAS_W), int(bow_anchor_uv[1] * CANVAS_H))
    stab_aim_x = Stabilizer(alpha=0.5)
    stab_aim_y = Stabilizer(alpha=0.5)
    stab_power = Stabilizer(alpha=0.4)

    # Targets spread at different depths
    placements = [
        (-5.5, -4.2, 15),
        (1.5, -1.1, 18),
        (-3.5, -4.9, 20),
        (3.5, -2.6, 22),
        (-7.0, -0.8, 26),
        (-0.5, -5.2, 30),
        (-3.0, -3.6, 34),
        (1.0, -2.2, 39),
        (-5.0, -1.6, 44),
    ]
    targets = [YearTarget3D(yr, *pos) for yr, pos in zip(YEARS, placements)]

    mem_year: int | None = None
    mem_timer = 0.0

    while True:
        now = time.time()
        dt = min(now - prev_time, 0.05)
        prev_time = now
        elapsed = now - start_ms / 1000.0

        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        fh, fw = frame.shape[:2]

        timestamp_ms = int(time.time() * 1000) - start_ms
        _, landmarks = detector.detect(frame, timestamp_ms)

        pinch_now = False
        if landmarks:
            draw_hand(frame, landmarks, fw, fh)
            pinch_now = is_pinching(landmarks)

        # ---- State machine ----
        if mem_year is None:
            if pinch_now and not pinch_prev:
                nocked = True
                power = 0.0
                _init_hand_size = hand_apparent_size(landmarks)
                _init_centroid = hand_centroid(landmarks)
                _aim_at_start = (aim_x, aim_y)
                bow_anchor_uv = _init_centroid
                bow_anchor_px = (
                    int(bow_anchor_uv[0] * CANVAS_W),
                    int(bow_anchor_uv[1] * CANVAS_H),
                )
                stab_aim_x.reset()
                stab_aim_y.reset()
                stab_power.reset()

            if pinch_now and nocked:
                # Power from depth: hand moving away → smaller apparent size
                cur_size = hand_apparent_size(landmarks)
                delta = _init_hand_size - cur_size
                raw_power = max(0.0, min(1.0, delta * PULL_SENSITIVITY))
                power = stab_power.update(raw_power)

                # Aim from centroid movement (like rotation in CubeController)
                cx, cy = hand_centroid(landmarks)
                dx = cx - _init_centroid[0]
                dy = cy - _init_centroid[1]
                raw_aim_x = max(
                    0.05, min(0.95, _aim_at_start[0] + dx * AIM_SENSITIVITY)
                )
                raw_aim_y = max(
                    0.05, min(0.95, _aim_at_start[1] + dy * AIM_SENSITIVITY)
                )
                aim_x = stab_aim_x.update(raw_aim_x)
                aim_y = stab_aim_y.update(raw_aim_y)
                bow_anchor_uv = (cx, cy)
                bow_anchor_px = (int(cx * CANVAS_W), int(cy * CANVAS_H))

            if not pinch_now and pinch_prev and nocked:
                if power > 0.06:
                    aim_sx = aim_x * CANVAS_W
                    aim_sy = aim_y * CANVAS_H
                    dir_x = (aim_sx - CANVAS_W / 2) / FOCAL
                    dir_y = (aim_sy - CANVAS_H / 2) / FOCAL
                    speed = power * ARROW_MAX_SPEED
                    new_arrow = Arrow3D()
                    new_arrow.launch(dir_x, dir_y, speed)
                    arrows.append(new_arrow)
                nocked = False
                power = 0.0

            if not pinch_now and not pinch_prev:
                nocked = False

        pinch_prev = pinch_now

        # ---- Update ----
        for a in arrows:
            a.update(dt)
            if a.active:
                for tgt in targets:
                    if tgt.check_hit(a.x, a.y, a.z):
                        a.active = False
                        particles.spawn(tgt.wx, tgt._cur_y, tgt.wz)
                        mem_year = tgt.year
                        mem_timer = 0.0
        arrows = [a for a in arrows if a.active]

        particles.update(dt)

        if mem_year is not None:
            mem_timer += dt
            if mem_timer > MEMORY_DURATION:
                mem_year = None

        # ---- Render 3D scene ----
        canvas = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)
        draw_scene(canvas, elapsed)

        # Targets (far first for painter's algorithm)
        for tgt in sorted(targets, key=lambda t: -t.wz):
            tgt.update(dt, elapsed)
            tgt.draw(canvas)

        mem_delay = 2.0  # let heart fill animation play before showing overlay
        if mem_year is not None and mem_timer > mem_delay:
            draw_memory_overlay(canvas, mem_year, mem_timer - mem_delay)
        else:
            # Arrows (far first)
            for a in sorted(arrows, key=lambda ar: -ar.z):
                a.draw(canvas)
            particles.draw(canvas)

            # FPS bow/arrow overlay
            draw_fps_bow(canvas, power, nocked, aim_x, aim_y, bow_anchor_px)

            # HUD
            draw_crosshair(canvas, int(aim_x * CANVAS_W), int(aim_y * CANVAS_H))

            if nocked:
                draw_power_meter(canvas, power)
                cv2.putText(
                    canvas,
                    "DRAWING...",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    PINK,
                    2,
                    cv2.LINE_AA,
                )
            else:
                cv2.putText(
                    canvas,
                    "Pinch to draw bow",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (180, 180, 200),
                    1,
                    cv2.LINE_AA,
                )

        # Hit counter
        hits = sum(1 for t in targets if t.hit)
        if hits > 0:
            cv2.putText(
                canvas,
                f"Memories: {hits}/{len(targets)}",
                (20, CANVAS_H - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                GOLD,
                2,
                cv2.LINE_AA,
            )

        cv2.putText(
            canvas,
            "Our Love Story",
            (CANVAS_W // 2 - 110, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            PINK,
            2,
            cv2.LINE_AA,
        )

        # PiP webcam (top-left, small)
        pip = cv2.resize(frame, (PIP_W, PIP_H))
        pip_x = CANVAS_W - PIP_W - 10
        pip_y = CANVAS_H - PIP_H - 10
        canvas[pip_y : pip_y + PIP_H, pip_x : pip_x + PIP_W] = pip
        cv2.rectangle(canvas, (pip_x - 1, pip_y - 1), (pip_x + PIP_W + 1, pip_y + PIP_H + 1), (100, 70, 100), 1)

        cv2.imshow("Our Love Story", canvas)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("r"):
            for tgt in targets:
                tgt.hit = False
                tgt.glow_t = 0.0
            arrows.clear()
            mem_year = None

    detector.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
