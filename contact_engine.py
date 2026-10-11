"""
Contact-detection engine for the NFL player-contact project.

Everything the Streamlit front end needs lives here so it can be tested
without a browser:

  * model definitions (identical layer names to the training notebook, so the
    saved .pth files load directly)
  * helmet-tracking / label CSV readers (chunked, because the Kaggle files are huge)
  * frame sampling that mirrors the training pipeline
  * CNN per-frame and LSTM per-window inference, plus Grad-CAM
  * player-pair analysis from helmet boxes
  * rule-based guidance text
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------
# Constants (match the training notebook)
# --------------------------------------------------------------------------
IMG_SIZE = 224
SEQ_LEN = 8          # frames per LSTM window
SEQ_STRIDE = 4       # window stride used when building sequences.pkl
FRAME_STRIDE = 5     # every 5th video frame was sampled for training
REF_W, REF_H = 1280, 720   # coordinate system of the Kaggle helmet CSV

HIGH_BAND = 0.70     # legend: red  = contact   (>70 %)
LOW_BAND = 0.40      # legend: blue = no contact (<40 %), yellow in between

REPORTED_METRICS = {
    "cnn": {"name": "CNN Model 2", "accuracy": 86.46, "f1": 0.8303,
            "type": "Per-frame", "detail": "3 conv + BatchNorm, LeakyReLU, Adam, dropout 0.3"},
    "lstm": {"name": "LSTM Model 1", "accuracy": 71.35, "f1": 0.7779,
             "type": "8-frame sequence", "detail": "3 conv + BatchNorm -> 1-layer LSTM(64)"},
}


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------
class CNNModel2(nn.Module):
    """CNN_Model2 from the notebook, split into features() / head() for Grad-CAM."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.leaky_relu = nn.LeakyReLU(0.1)
        self.avgpool = nn.AvgPool2d(2, 2)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(28 * 28 * 128, 256)
        self.fc2 = nn.Linear(256, 1)

    def features(self, x):
        x = self.avgpool(self.leaky_relu(self.bn1(self.conv1(x))))
        x = self.avgpool(self.leaky_relu(self.bn2(self.conv2(x))))
        return self.leaky_relu(self.bn3(self.conv3(x)))          # (B,128,56,56)

    def head(self, f):
        x = self.avgpool(f).flatten(1)                           # (B,100352)
        return self.fc2(self.dropout(self.leaky_relu(self.fc1(x))))

    def forward(self, x):
        return self.head(self.features(x))


class LSTMModel1(nn.Module):
    """LSTM_Model1 from the notebook: small CNN per frame -> LSTM over 8 frames."""

    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(4, 4),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(4, 4),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Sequential(nn.Linear(64, 64), nn.ReLU())
        self.lstm = nn.LSTM(64, 64, num_layers=1, batch_first=True)
        self.classifier = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):                                        # x: (B,T,3,H,W)
        b, t = x.shape[:2]
        f = self.cnn(x.reshape(b * t, *x.shape[2:])).reshape(b * t, 64)
        f = self.proj(f).reshape(b, t, 64)
        out, _ = self.lstm(f)
        return self.classifier(out[:, -1, :])


def pick_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_state(model: nn.Module, path: str | Path, device: torch.device, label: str):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{label} weights not found at '{path}'.")
    state = torch.load(path, map_location=device, weights_only=True)
    if isinstance(state, dict):
        for key in ("state_dict", "model_state_dict"):
            if key in state and isinstance(state[key], dict):
                state = state[key]
                break
    try:
        model.load_state_dict(state)
    except RuntimeError as exc:
        raise RuntimeError(
            f"{label} weights at '{path}' do not match the expected architecture. "
            f"Use the best_weights.pth from the notebook run that produced this model.\n{exc}"
        ) from exc
    return model.to(device).eval()


def load_models(cnn_path, lstm_path, device=None):
    device = device or pick_device()
    cnn = _load_state(CNNModel2(), cnn_path, device, "CNN Model 2")
    lstm = _load_state(LSTMModel1(), lstm_path, device, "LSTM Model 1")
    return cnn, lstm


# --------------------------------------------------------------------------
# CSV helpers
# --------------------------------------------------------------------------
def parse_video_name(name: str):
    """'58168_003392_Sideline.mp4' -> ('58168_003392', 'Sideline'); (None, None) if no match."""
    m = re.match(r"^(\d+_\d+)_(endzone|sideline)", Path(name).stem, re.I)
    if not m:
        return None, None
    return m.group(1), m.group(2).capitalize()


def _chunks(source, usecols, chunksize=500_000):
    if hasattr(source, "seek"):
        source.seek(0)
    return pd.read_csv(source, usecols=lambda c: c in usecols, chunksize=chunksize)


def load_helmets(source, game_play: str | None, view: str | None) -> pd.DataFrame:
    """Read helmet boxes for one play/view from the Kaggle CSV (path or file object)."""
    cols = {"game_play", "view", "frame", "nfl_player_id", "player_label",
            "left", "top", "width", "height"}
    parts = []
    for ch in _chunks(source, cols):
        if game_play and "game_play" in ch:
            ch = ch[ch["game_play"] == game_play]
        if view and "view" in ch:
            ch = ch[ch["view"] == view]
        if len(ch):
            parts.append(ch)
    if not parts:
        return pd.DataFrame()
    df = pd.concat(parts, ignore_index=True)
    missing = {"frame", "left", "top", "width", "height"} - set(df.columns)
    if missing:
        raise ValueError(f"Helmet CSV is missing columns: {sorted(missing)}")
    if "nfl_player_id" not in df:
        df["nfl_player_id"] = df.groupby("frame").cumcount()
    if "player_label" not in df:
        df["player_label"] = df["nfl_player_id"].astype(str)
    df["frame"] = df["frame"].astype(int)
    return df


def load_contact_labels(source, game_play: str) -> pd.DataFrame:
    """Per-step ground truth (any pair in contact) for one play."""
    parts = []
    for ch in _chunks(source, {"game_play", "step", "contact"}):
        ch = ch[ch["game_play"] == game_play]
        if len(ch):
            parts.append(ch[["step", "contact"]])
    if not parts:
        return pd.DataFrame(columns=["step", "contact"])
    return pd.concat(parts).groupby("step", as_index=False)["contact"].max()


# --------------------------------------------------------------------------
# Frame sampling
# --------------------------------------------------------------------------
@dataclass
class Clip:
    frames_224: np.ndarray            # (N,224,224,3) uint8 RGB, what the models see
    frames_disp: list                 # RGB frames for display
    frame_nums: list                  # original video frame numbers
    fps: float
    width: int
    height: int
    snap_offset: int
    has_helmets: bool


def sample_frames(video_path, helmets: pd.DataFrame | None = None, max_frames=30,
                  stride=FRAME_STRIDE, display_width=720) -> Clip:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError("Could not open the video. Try an .mp4 encoded with H.264.")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    has_helmets = helmets is not None and len(helmets) > 0
    if has_helmets:                       # same rule as the training extraction
        start = int(helmets["frame"].min())
        end = int(helmets["frame"].max())
        valid = set(helmets["frame"].astype(int).unique())
    else:
        start, end, valid = 0, max(total - 1, 0), None

    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    disp_h = int(round(height * display_width / width)) if width > display_width else height
    disp_w = display_width if width > display_width else width

    f224, fdisp, nums = [], [], []
    cur = start
    while len(nums) < max_frames and cur <= end:
        ok, frame = cap.read()
        if not ok:
            break
        if (cur - start) % stride == 0 and (valid is None or cur in valid):
            f224.append(cv2.cvtColor(cv2.resize(frame, (IMG_SIZE, IMG_SIZE)), cv2.COLOR_BGR2RGB))
            fdisp.append(cv2.cvtColor(cv2.resize(frame, (disp_w, disp_h)), cv2.COLOR_BGR2RGB))
            nums.append(cur)
        cur += 1
    cap.release()
    if not nums:
        raise ValueError("No frames could be read. If you loaded a helmet CSV, check that the "
                         "game-play / view match this video.")
    return Clip(np.stack(f224), fdisp, nums, float(fps), width, height, start, has_helmets)


# --------------------------------------------------------------------------
# Inference
# --------------------------------------------------------------------------
def _to_tensor(frames_224: np.ndarray, device) -> torch.Tensor:
    # training used RGB scaled to 0-1, no mean/std normalisation
    return torch.from_numpy(frames_224).permute(0, 3, 1, 2).float().div(255).to(device)


@torch.no_grad()
def cnn_probs(model, frames_224, device, batch=16) -> np.ndarray:
    out = []
    for i in range(0, len(frames_224), batch):
        out.append(torch.sigmoid(model(_to_tensor(frames_224[i:i + batch], device))).squeeze(1).cpu())
    return torch.cat(out).numpy()


def window_starts(n: int, seq_len=SEQ_LEN, stride=SEQ_STRIDE) -> list:
    if n < seq_len:
        return []
    starts = list(range(0, n - seq_len + 1, stride))
    if starts[-1] != n - seq_len:         # cover the tail as well
        starts.append(n - seq_len)
    return starts


@torch.no_grad()
def lstm_probs(model, frames_224, device, seq_len=SEQ_LEN, stride=SEQ_STRIDE):
    """Returns [(start_idx, end_idx, prob), ...] over sliding windows."""
    wins = []
    for s in window_starts(len(frames_224), seq_len, stride):
        x = _to_tensor(frames_224[s:s + seq_len], device).unsqueeze(0)
        wins.append((s, s + seq_len - 1, float(torch.sigmoid(model(x)).item())))
    return wins


def gradcam(model: CNNModel2, frame_224: np.ndarray, device) -> np.ndarray:
    """Grad-CAM of the contact logit on the last conv block -> (224,224) in [0,1]."""
    model.eval()
    x = _to_tensor(frame_224[None], device)
    with torch.enable_grad():
        feats = model.features(x)
        feats.retain_grad()
        logit = model.head(feats)[0, 0]
        model.zero_grad(set_to_none=True)
        logit.backward()
    w = feats.grad.mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((w * feats).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)[0, 0]
    cam = cam / (cam.max() + 1e-8)
    return cam.detach().cpu().numpy()


# --------------------------------------------------------------------------
# Whole-clip analysis
# --------------------------------------------------------------------------
@dataclass
class Analysis:
    clip: Clip
    cnn: np.ndarray
    windows: list
    lstm_at: np.ndarray          # LSTM prob covering each frame (nan if none)
    fused: np.ndarray
    events: list
    key_idx: int
    thr: float
    min_run: int = 2
    helmets: pd.DataFrame | None = None
    labels: pd.DataFrame | None = None
    game_play: str | None = None
    view: str | None = None
    cams: dict = field(default_factory=dict)


def fuse_scores(cnn, windows, n):
    acc, cnt = np.zeros(n), np.zeros(n)
    for s, e, p in windows:
        acc[s:e + 1] += p
        cnt[s:e + 1] += 1
    lstm_at = np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan)
    fused = np.where(np.isnan(lstm_at), cnn, (cnn + lstm_at) / 2)
    return lstm_at, fused


def find_events(fused, thr, min_run):
    """Runs of >= min_run consecutive sampled frames at/above the threshold."""
    events, i, n = [], 0, len(fused)
    while i < n:
        if fused[i] >= thr:
            j = i
            while j + 1 < n and fused[j + 1] >= thr:
                j += 1
            if j - i + 1 >= min_run:
                peak = i + int(np.argmax(fused[i:j + 1]))
                events.append({"start": i, "end": j, "peak": peak, "peak_prob": float(fused[peak])})
            i = j + 1
        else:
            i += 1
    return events


def analyze(video_path, cnn_model, lstm_model, device, helmets=None, labels=None,
            game_play=None, view=None, thr=0.5, min_run=2, max_frames=30,
            stride=FRAME_STRIDE) -> Analysis:
    clip = sample_frames(video_path, helmets, max_frames, stride)
    cnn = cnn_probs(cnn_model, clip.frames_224, device)
    wins = lstm_probs(lstm_model, clip.frames_224, device)
    lstm_at, fused = fuse_scores(cnn, wins, len(cnn))
    events = find_events(fused, thr, min_run)
    key_idx = max(events, key=lambda e: e["peak_prob"])["peak"] if events else int(np.argmax(fused))
    return Analysis(clip, cnn, wins, lstm_at, fused, events, key_idx, thr, min_run,
                    helmets if clip.has_helmets else None, labels, game_play, view)


def reevaluate(a: Analysis, thr, min_run) -> Analysis:
    """Re-apply thresholds without re-running the networks."""
    a.thr, a.min_run = thr, min_run
    a.events = find_events(a.fused, thr, min_run)
    if a.events:
        a.key_idx = max(a.events, key=lambda e: e["peak_prob"])["peak"]
    return a


def true_label(a: Analysis, idx: int):
    """Ground truth at a sampled frame, using the same frame->step rule as the notebook."""
    if a.labels is None or a.labels.empty:
        return None
    step = a.clip.frame_nums[idx] - a.clip.snap_offset
    row = a.labels[a.labels["step"] == step]
    return None if row.empty else int(row["contact"].iloc[0])


def time_s(a: Analysis, idx: int) -> float:
    return a.clip.frame_nums[idx] / a.clip.fps


def band(p: float) -> str:
    return "high" if p >= HIGH_BAND else ("mid" if p >= LOW_BAND else "low")


# --------------------------------------------------------------------------
# Players and pairs
# --------------------------------------------------------------------------
def helmets_at(a: Analysis, idx: int) -> pd.DataFrame:
    """Helmet boxes for the nearest tracked frame, scaled to display coordinates."""
    if a.helmets is None or a.helmets.empty:
        return pd.DataFrame()
    fr = a.clip.frame_nums[idx]
    frames = a.helmets["frame"].unique()
    near = frames[np.argmin(np.abs(frames - fr))]
    b = a.helmets[a.helmets["frame"] == near].copy().reset_index(drop=True)
    h, w = a.clip.frames_disp[idx].shape[:2]
    sx, sy = w / REF_W, h / REF_H
    b["cx"] = (b["left"] + b["width"] / 2) * sx
    b["cy"] = (b["top"] + b["height"] / 2) * sy
    b["w"] = b["width"] * sx
    b["h"] = b["height"] * sy
    return b


def analyze_pairs(boxes: pd.DataFrame, cam: np.ndarray | None, disp_shape, prob: float,
                  thr: float, dist_thr: float) -> pd.DataFrame:
    """Pairs of helmets that are close together, ranked by distance.

    Distance is measured in helmet widths, so it is the same for near and far
    players. 'Probable contact' needs both: the models say contact AND the pair is close.
    """
    if boxes.empty or len(boxes) < 2:
        return pd.DataFrame()
    H, W = disp_shape[:2]
    cx, cy, w, h = (boxes[c].to_numpy() for c in ("cx", "cy", "w", "h"))
    ids, labels = boxes["nfl_player_id"].to_numpy(), boxes["player_label"].astype(str).to_numpy()
    rows = []
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            d = float(np.hypot(cx[i] - cx[j], cy[i] - cy[j]))
            dn = d / max((w[i] + w[j]) / 2, 1e-6)
            if dn > dist_thr:
                continue
            mx, my = (cx[i] + cx[j]) / 2, (cy[i] + cy[j]) / 2
            att = np.nan
            if cam is not None:
                gx, gy = int(np.clip(mx / W * IMG_SIZE, 0, IMG_SIZE - 1)), int(np.clip(my / H * IMG_SIZE, 0, IMG_SIZE - 1))
                att = float(cam[max(gy - 6, 0):gy + 7, max(gx - 6, 0):gx + 7].mean())
            rows.append({
                "Player A": labels[i], "Player B": labels[j],
                "Distance (helmet widths)": round(dn, 2),
                "Model attention": None if np.isnan(att) else round(att, 2),
                "Status": "Probable contact" if prob >= thr else "Close, not flagged",
                "_ids": (int(ids[i]) if str(ids[i]).isdigit() else ids[i], int(ids[j]) if str(ids[j]).isdigit() else ids[j]),
                "_i": i, "_j": j, "_mx": mx, "_my": my,
            })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("Distance (helmet widths)").reset_index(drop=True)


# --------------------------------------------------------------------------
# Drawing
# --------------------------------------------------------------------------
def overlay_heatmap(img: np.ndarray, cam: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    c = cv2.resize(cam, (w, h), interpolation=cv2.INTER_CUBIC).clip(0, 1)
    color = cv2.cvtColor(cv2.applyColorMap((c * 255).astype(np.uint8), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    alpha = (0.65 * c ** 0.8)[..., None]
    return (img * (1 - alpha) + color * alpha).astype(np.uint8)


def draw_players(img: np.ndarray, boxes: pd.DataFrame, pairs: pd.DataFrame, max_pairs=6) -> np.ndarray:
    out = img.copy()
    in_pair = set()
    if pairs is not None and not pairs.empty:
        for _, r in pairs.head(max_pairs).iterrows():
            in_pair.update((r["_i"], r["_j"]))
    for i, b in boxes.iterrows():
        x0, y0 = int(b["cx"] - b["w"] / 2), int(b["cy"] - b["h"] / 2)
        color = (255, 255, 255) if i in in_pair else (60, 200, 110)
        cv2.rectangle(out, (x0, y0), (int(x0 + b["w"]), int(y0 + b["h"])), color, 1)
    if pairs is not None and not pairs.empty:
        for k, r in pairs.head(max_pairs).iterrows():
            probable = r["Status"] == "Probable contact"
            color = (230, 50, 40) if probable else (245, 170, 30)
            a, b = boxes.loc[r["_i"]], boxes.loc[r["_j"]]
            cv2.line(out, (int(a["cx"]), int(a["cy"])), (int(b["cx"]), int(b["cy"])), color, 2)
            cv2.circle(out, (int(r["_mx"]), int(r["_my"])), max(4, int(a["w"] * 0.8)), color, 2)
            label = f"{r['Player A']}-{r['Player B']}"
            cv2.putText(out, label, (int(r["_mx"]) + 8, int(r["_my"]) - 8), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(out, label, (int(r["_mx"]) + 8, int(r["_my"]) - 8), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, color, 1, cv2.LINE_AA)
    return out


# --------------------------------------------------------------------------
# Guidance
# --------------------------------------------------------------------------
def _fmt_t(t):
    return f"{t:.2f} s"


def build_guidance(a: Analysis, pairs_key: pd.DataFrame | None, n_players: int,
                   dist_thr: float) -> dict:
    """Plain-language review guidance from the analysis. Pure rules, no model calls."""
    k = a.key_idx
    cnn_p, lstm_p, fused = float(a.cnn[k]), a.lstm_at[k], float(a.fused[k])
    lstm_known = not np.isnan(lstm_p)
    cnn_yes = cnn_p >= a.thr
    lstm_yes = lstm_known and lstm_p >= a.thr
    contact = bool(a.events)
    other_view = {"Sideline": "Endzone", "Endzone": "Sideline"}.get(a.view or "", "other")
    per_step = a.clip.fps / max(1, (a.clip.frame_nums[1] - a.clip.frame_nums[0])) if len(a.clip.frame_nums) > 1 else a.clip.fps

    g = {"contact": contact, "findings": [], "actions": [], "cautions": []}

    if contact:
        ev = max(a.events, key=lambda e: e["peak_prob"])
        t0, t1 = time_s(a, ev["start"]), time_s(a, ev["end"])
        n_fr = ev["end"] - ev["start"] + 1
        dur = n_fr / per_step
        if cnn_yes and lstm_yes and fused >= 0.8:
            g["tier"], g["tier_text"] = "high", "High confidence: both models agree"
        elif cnn_yes and lstm_yes:
            g["tier"], g["tier_text"] = "mid", "Moderate confidence: both models agree"
        else:
            g["tier"], g["tier_text"] = "mid", "Low confidence: only one model flags contact"
            if not lstm_known:
                g["tier_text"] = "Moderate confidence: single model (clip too short for the LSTM)"
        g["headline"] = (f"Contact detected at {_fmt_t(t0)}" if n_fr == 1 else
                         f"Contact detected from {_fmt_t(t0)} to {_fmt_t(t1)}")
        extra = f" {len(a.events) - 1} further event(s) were also flagged." if len(a.events) > 1 else ""
        g["findings"].append(
            f"The strongest event spans {n_fr} sampled frame(s), about {dur:.2f} s, peaking at "
            f"{_fmt_t(time_s(a, ev['peak']))} (combined score {ev['peak_prob']:.0%}).{extra}")
        g["findings"].append(
            f"At the peak frame the CNN gives {cnn_p:.0%} and the LSTM "
            f"{'%.0f%%' % (lstm_p * 100) if lstm_known else 'n/a'}.")
        if dur >= 0.5:
            g["findings"].append("Contact is sustained, which usually means a continuous engagement "
                                 "such as a block or a tackle rather than a brief touch.")
        else:
            g["findings"].append("Contact is brief. It may be a glancing touch, so confirm it on the other angle.")

        acts = [f"Open the frame viewer and step through {_fmt_t(t0)} to {_fmt_t(t1)}. Start at the peak frame, {_fmt_t(time_s(a, ev['peak']))}."]
        if pairs_key is not None and not pairs_key.empty:
            top = pairs_key.iloc[0]
            more = f" {len(pairs_key) - 1} other pair(s) are also close." if len(pairs_key) > 1 else ""
            acts.append(f"Check players {top['Player A']} and {top['Player B']} first. They are the closest pair at the "
                        f"peak frame ({top['Distance (helmet widths)']} helmet widths apart).{more}")
        elif a.clip.has_helmets:
            acts.append("No two helmets are within the close-pair distance at this frame. The models flagged contact, "
                        "so widen the distance in the sidebar or check the neighbouring frames.")
        else:
            acts.append("Add the helmet tracking CSV (sidebar) to see which players are involved.")
        if g["tier"] == "high":
            acts.append(f"Mark this play for review and confirm it on the {other_view} view.")
        else:
            acts.append(f"Treat this result as a lead, not a finding. Check the {other_view} view and the frames either side of the peak.")
        if n_players >= 18:
            acts.append(f"{n_players} players are in this frame. Overlapping bodies cause occlusion, so verify by eye.")
        g["actions"] = acts
    else:
        g["tier"], g["tier_text"] = "low", "No sustained contact flagged"
        g["headline"] = "No sustained contact detected"
        g["findings"].append(f"No run of {a.min_run} or more sampled frames reached the {a.thr:.0%} threshold. "
                             f"The highest combined score was {a.fused.max():.0%}.")
        acts = ["No review is needed based on this clip."]
        if pairs_key is not None and not pairs_key.empty:
            acts.append(f"{len(pairs_key)} pair(s) are close at the key frame, led by {pairs_key.iloc[0]['Player A']} and "
                        f"{pairs_key.iloc[0]['Player B']}. Check them by eye if the play looks physical.")
        acts.append(f"If you expect contact, lower the threshold in the sidebar or analyse the {other_view} view.")
        g["actions"] = acts

    g["cautions"] = [
        "This is review support, not an officiating or medical decision.",
        "The models score whole frames. They were trained on 14,400 frames from 240 plays, with a random frame split, "
        "so the reported accuracy may be optimistic on new games.",
    ]
    if a.clip.has_helmets:
        g["cautions"].append("Player names come from helmet-box distance, not from the models. Nearby players are not always touching.")
    else:
        g["cautions"].append("No helmet tracking was loaded, so the result covers the whole frame and cannot name players.")
    return g
