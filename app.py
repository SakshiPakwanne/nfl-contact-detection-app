"""
Contact review: Streamlit front end for the NFL player-contact detection project.

Run with:   streamlit run app.py
"""
from __future__ import annotations

import html
import io
import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

import contact_engine as ce

st.set_page_config(page_title="Contact review", page_icon=":football:", layout="wide")

# ---------------------------------------------------------------------------
# Look and feel
# ---------------------------------------------------------------------------
BAND = {"high": ("#c8322b", "Contact"), "mid": ("#d99a00", "Uncertain"), "low": ("#2c6bd8", "No contact")}

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@500;600;700&display=swap');
.block-container { max-width: 1180px; padding-top: 2.2rem; }
h1, h2, h3 { font-family: 'Barlow Condensed', 'Arial Narrow', sans-serif !important; letter-spacing: .2px; }
h1 { font-size: 2.9rem !important; font-weight: 700 !important; margin-bottom: .1rem; }
h2, h3 { font-weight: 600 !important; }
.lede { color: #5b6672; font-size: 1.05rem; max-width: 62ch; margin: 0 0 1.2rem 0; }

.board { background: #101820; color: #fff; border-radius: 6px; border-top: 8px solid var(--c);
         padding: 22px 28px; display: flex; flex-wrap: wrap; gap: 28px 44px; align-items: center; margin: .6rem 0 1rem; }
.board-main { flex: 1 1 280px; min-width: 240px; }
.board-where { color: #9aa7b4; font-size: .92rem; }
.board-verdict { font-family: 'Barlow Condensed', 'Arial Narrow', sans-serif; font-weight: 700;
                 font-size: 4.2rem; line-height: 1; text-transform: uppercase; color: var(--c); filter: brightness(1.35); }
.board-sub { font-size: 1.05rem; margin-top: 6px; }
.cells { display: flex; gap: 34px; flex-wrap: wrap; }
.cell .num { font-family: 'Barlow Condensed', 'Arial Narrow', sans-serif; font-weight: 600; font-size: 2.5rem; line-height: 1.05; }
.cell .lbl { color: #9aa7b4; font-size: .85rem; max-width: 12ch; }

.guide { background: #fff; border: 1px solid #d9dee3; border-left: 8px solid var(--c); border-radius: 6px; padding: 20px 26px; margin: 0 0 1.2rem; }
.guide h3 { margin: 0 0 .15rem 0; font-size: 1.7rem; }
.guide .tier { color: var(--c); font-weight: 600; margin-bottom: .6rem; }
.guide h4 { font-size: 1rem; margin: 1rem 0 .25rem; }
.guide ul, .guide ol { margin: 0; padding-left: 1.25rem; }
.guide li { margin: .22rem 0; max-width: 78ch; }
.guide .keep { color: #5b6672; font-size: .9rem; }

.legend { display: flex; flex-wrap: wrap; gap: 8px 22px; font-size: .88rem; color: #3b4652; margin: .2rem 0 .8rem; }
.legend i { display: inline-block; width: 14px; height: 4px; border-radius: 2px; margin-right: 7px; vertical-align: middle; }
.legend i.box { height: 12px; width: 12px; background: transparent; border: 2px solid; border-radius: 1px; }
.chip { display: inline-block; padding: 2px 10px; border-radius: 12px; color: #fff; font-size: .85rem; font-weight: 600; }
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_models(cnn_path: str, lstm_path: str, mtimes: tuple):
    device = ce.pick_device()
    cnn, lstm = ce.load_models(cnn_path, lstm_path, device)
    return cnn, lstm, device


@st.cache_data(show_spinner=False)
def cached_helmets(path: str, mtime: float, game_play, view):
    return ce.load_helmets(path, game_play, view)


@st.cache_data(show_spinner=False)
def cached_labels(path: str, mtime: float, game_play):
    return ce.load_contact_labels(path, game_play)


def mtime(p: str) -> float:
    return os.path.getmtime(p) if p and os.path.exists(p) else 0.0


def png_bytes(rgb: np.ndarray) -> bytes:
    return cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))[1].tobytes()


# ---------------------------------------------------------------------------
# Sidebar: setup and detection settings
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Setup")
    st.subheader("Model weights")
    cnn_path = st.text_input("CNN Model 2", "weights/cnn_model2/best_weights.pth")
    lstm_path = st.text_input("LSTM Model 1", "weights/lstm_model1/best_weights.pth")
    weights_ok = Path(cnn_path).exists() and Path(lstm_path).exists()
    if weights_ok:
        st.caption(f"Both weight files found. Running on {ce.pick_device().type.upper()}.")
    else:
        st.caption("A weight file was not found. Copy best_weights.pth from your project into the paths above.")

    st.subheader("Player tracking (optional)")
    helmets_path = st.text_input("Helmet CSV path", "data/train_baseline_helmets.csv",
                                 help="Kaggle train_baseline_helmets.csv, or a smaller file already cut down to one play.")
    helmets_upload = st.file_uploader("Or upload a helmet CSV", type=["csv"])
    labels_path = st.text_input("Labels CSV path (for ground truth)", "data/train_labels.csv")
    has_helmet_src = bool(helmets_upload) or Path(helmets_path).is_file()
    st.caption("With helmet data the app names the players involved. Without it, you get a whole-frame result only.")

    st.subheader("Detection settings")
    thr = st.slider("Contact threshold", 0.30, 0.90, 0.50, 0.05,
                    help="A frame counts as contact when the combined CNN and LSTM score reaches this value.")
    min_run = st.slider("Frames in a row", 1, 5, 2,
                        help="Contact must hold for this many sampled frames in a row. Higher values ignore one-frame flickers.")
    dist_thr = st.slider("Close-pair distance (helmet widths)", 0.5, 4.0, 2.0, 0.25,
                         help="Two players count as close when their helmet centres are this near, measured in helmet widths.")
    max_frames = st.slider("Frames to analyse", 8, 120, 30, 2,
                           help="30 matches how the models were trained. The LSTM needs at least 8.")
    stride = st.slider("Video frames between samples", 1, 15, ce.FRAME_STRIDE,
                       help="5 matches training. Use a smaller number for slower-moving footage.")


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown("# Contact review")
st.markdown('<p class="lede">Upload a play from the sideline or endzone camera. The app scores each frame for '
            "player contact, marks the players involved, and tells you where to look and what to check next.</p>",
            unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Upload and analyse
# ---------------------------------------------------------------------------
left, right = st.columns([3, 2], gap="large")
with left:
    video = st.file_uploader("Play video", type=["mp4", "mov", "m4v", "avi"],
                             help="Name it like 58168_003392_Sideline.mp4 so the app can match it to the helmet data.")
with right:
    gp0, view0 = ce.parse_video_name(video.name) if video else (None, None)
    game_play = st.text_input("Game and play", gp0 or "", placeholder="58168_003392",
                              disabled=not has_helmet_src, key=f"gp_{video.name if video else ''}")
    view = st.selectbox("Camera view", ["Sideline", "Endzone"],
                        index=0 if (view0 or "Sideline") == "Sideline" else 1, disabled=not has_helmet_src,
                        key=f"view_{video.name if video else ''}")
    go = st.button("Analyse contact", type="primary", disabled=not (video and weights_ok), width="stretch")
    if video and not weights_ok:
        st.caption("Set the model weight paths in the sidebar first.")

if go:
    tmp = None
    try:
        with st.spinner("Loading models"):
            cnn_m, lstm_m, device = get_models(cnn_path, lstm_path, (mtime(cnn_path), mtime(lstm_path)))
        helmets = labels = None
        if has_helmet_src and game_play:
            with st.spinner("Reading player tracking data (the first read of a large CSV can take a minute)"):
                if helmets_upload is not None:
                    helmets = ce.load_helmets(io.BytesIO(helmets_upload.getvalue()), game_play, view)
                else:
                    helmets = cached_helmets(helmets_path, mtime(helmets_path), game_play, view)
                if Path(labels_path).is_file():
                    labels = cached_labels(labels_path, mtime(labels_path), game_play)
            if helmets is None or helmets.empty:
                st.warning(f"No helmet rows found for play {game_play}, {view} view. Continuing with a whole-frame analysis.")
                helmets = None
        with st.spinner("Scoring frames"):
            tmp = tempfile.NamedTemporaryFile(suffix=Path(video.name).suffix, delete=False)
            tmp.write(video.getvalue())
            tmp.close()
            a = ce.analyze(tmp.name, cnn_m, lstm_m, device, helmets=helmets, labels=labels,
                           game_play=game_play or None, view=view if helmets is not None else None,
                           thr=thr, min_run=min_run, max_frames=max_frames, stride=stride)
        st.session_state.analysis = a
        st.session_state.models = (cnn_m, lstm_m, device)
        st.session_state.run_params = (max_frames, stride, video.name)
        st.session_state.frame_idx = a.key_idx
        st.session_state.video_name = video.name
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        st.error(str(exc))
    finally:
        if tmp is not None and os.path.exists(tmp.name):
            os.remove(tmp.name)

# ---------------------------------------------------------------------------
# Empty state
# ---------------------------------------------------------------------------
a: ce.Analysis | None = st.session_state.get("analysis")
if a is None:
    st.markdown("#### How it works")
    st.markdown(
        "1. Copy your trained weights into `weights/` (paths are in the sidebar).\n"
        "2. Upload a play video. For player names, also point the sidebar at the Kaggle helmet CSV.\n"
        "3. Press **Analyse contact**. You get a verdict, a timeline, a frame-by-frame viewer and written guidance."
    )
    st.stop()

# ---------------------------------------------------------------------------
# Re-apply settings that do not need the networks
# ---------------------------------------------------------------------------
ce.reevaluate(a, thr, min_run)
if st.session_state.get("run_params", (None,))[:2] != (max_frames, stride):
    st.info("Frame settings changed since this analysis. Press Analyse contact to apply them.")
n = len(a.cnn)
cnn_m, lstm_m, device = st.session_state.models


def cam_for(i: int) -> np.ndarray:
    if i not in a.cams:
        a.cams[i] = ce.gradcam(cnn_m, a.clip.frames_224[i], device)
    return a.cams[i]


def pairs_for(i: int):
    boxes = ce.helmets_at(a, i)
    pairs = ce.analyze_pairs(boxes, cam_for(i), a.clip.frames_disp[i].shape, float(a.fused[i]), a.thr, dist_thr)
    return boxes, pairs


# ---------------------------------------------------------------------------
# Scoreboard
# ---------------------------------------------------------------------------
k = a.key_idx
boxes_k, pairs_k = pairs_for(k)
guide = ce.build_guidance(a, pairs_k, len(boxes_k), dist_thr)
peak = max((e["peak_prob"] for e in a.events), default=float(a.fused[k]))
colour = BAND[ce.band(peak)][0] if a.events else BAND["low"][0]
lstm_k = a.lstm_at[k]
where = (f"{a.view} view, play {a.game_play}" if a.game_play and a.view else st.session_state.get("video_name", ""))
flagged = int((a.fused >= a.thr).sum())

st.markdown(
    f"""
<div class="board" style="--c:{colour}">
  <div class="board-main">
    <div class="board-where">{html.escape(where)}</div>
    <div class="board-verdict">{'Contact' if a.events else 'No contact'}</div>
    <div class="board-sub">{html.escape(guide['headline'])}</div>
  </div>
  <div class="cells">
    <div class="cell"><div class="num">{a.cnn[k]:.0%}</div><div class="lbl">CNN score at key frame</div></div>
    <div class="cell"><div class="num">{'n/a' if np.isnan(lstm_k) else format(lstm_k, '.0%')}</div><div class="lbl">LSTM score at key frame</div></div>
    <div class="cell"><div class="num">{len(boxes_k) if a.clip.has_helmets else 'n/a'}</div><div class="lbl">Players tracked</div></div>
    <div class="cell"><div class="num">{flagged}/{n}</div><div class="lbl">Frames flagged</div></div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Guidance
# ---------------------------------------------------------------------------
gcol = {"high": BAND["high"][0], "mid": BAND["mid"][0], "low": BAND["low"][0]}[guide["tier"]]
li = lambda items, tag="ul": f"<{tag}>" + "".join(f"<li>{html.escape(x)}</li>" for x in items) + f"</{tag}>"
st.markdown(
    f"""
<div class="guide" style="--c:{gcol}">
  <h3>What to do next</h3>
  <div class="tier">{html.escape(guide['tier_text'])}</div>
  <h4>What the models found</h4>{li(guide['findings'])}
  <h4>Steps to take</h4>{li(guide['actions'], 'ol')}
  <h4 class="keep">Keep in mind</h4><div class="keep">{li(guide['cautions'])}</div>
</div>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Timeline
# ---------------------------------------------------------------------------
st.subheader("Contact score over time")
times = np.round([ce.time_s(a, i) for i in range(n)], 2)
tl = pd.DataFrame({"CNN": a.cnn, "LSTM": a.lstm_at, "Combined": a.fused, "Threshold": a.thr}, index=times)
st.line_chart(tl, color=["#2c6bd8", "#1f5c3a", "#101820", "#c8322b"], x_label="Time in video (s)",
              y_label="Contact score", height=240)

# ---------------------------------------------------------------------------
# Frame viewer
# ---------------------------------------------------------------------------
st.subheader("Frame viewer")
if st.session_state.get("frame_idx", 0) >= n:
    st.session_state.frame_idx = 0
if n > 1:
    c1, c2 = st.columns([5, 1], vertical_alignment="bottom")
    with c1:
        idx = st.slider("Sampled frame", 0, n - 1, key="frame_idx",
                        help="Each step is one sampled frame. The peak frame is where the combined score is highest.")
    with c2:
        st.button("Jump to peak", width="stretch",
                  on_click=lambda: st.session_state.update(frame_idx=st.session_state.analysis.key_idx))
else:
    idx = 0

img = a.clip.frames_disp[idx]
cam = cam_for(idx)
boxes, pairs = pairs_for(idx)
p_cnn, p_lstm, p_fused = float(a.cnn[idx]), a.lstm_at[idx], float(a.fused[idx])
bcol, btxt = BAND[ce.band(p_fused)]
st.markdown(
    f'<span class="chip" style="background:{bcol}">{btxt}</span> &nbsp; Frame {a.clip.frame_nums[idx]} '
    f"at {ce.time_s(a, idx):.2f} s. CNN {p_cnn:.0%}, LSTM "
    f"{'n/a' if np.isnan(p_lstm) else format(p_lstm, '.0%')}, combined {p_fused:.0%}.",
    unsafe_allow_html=True,
)

players_img = ce.draw_players(img, boxes, pairs) if a.clip.has_helmets else img
cam_img = ce.overlay_heatmap(img, cam)
cols = st.columns(3 if a.clip.has_helmets else 2)
with cols[0]:
    st.image(img, caption="Original frame", width="stretch")
if a.clip.has_helmets:
    with cols[1]:
        st.image(players_img, caption=f"Players and close pairs ({len(pairs)} close)", width="stretch")
with cols[-1]:
    st.image(cam_img, caption="Where the CNN looked (Grad-CAM)", width="stretch")

st.markdown(
    '<div class="legend">'
    '<span><i style="background:#e6322a"></i>Probable contact pair</span>'
    '<span><i style="background:#f5aa1e"></i>Close pair, not flagged</span>'
    '<span><i class="box" style="border-color:#3cc86e"></i>Tracked helmet</span>'
    '<span><i style="background:linear-gradient(90deg,#2c6bd8,#e6322a)"></i>Heatmap: warm areas pushed the score toward contact</span>'
    "</div>",
    unsafe_allow_html=True,
)

truth = ce.true_label(a, idx)
if truth is not None:
    ok = lambda p: "correct" if (p >= a.thr) == bool(truth) else "wrong"
    st.info(f"Ground truth at this frame: {'contact' if truth else 'no contact'}. CNN {ok(p_cnn)}"
            + ("" if np.isnan(p_lstm) else f", LSTM {ok(p_lstm)}") + ". (Uses the notebook's frame-to-step rule.)")

if a.clip.has_helmets:
    if pairs.empty:
        st.caption("No two helmets are within the close-pair distance at this frame.")
    else:
        show = pairs[["Player A", "Player B", "Distance (helmet widths)", "Model attention", "Status"]].head(10)
        st.dataframe(show, hide_index=True, width="stretch")
        st.caption("Model attention is the Grad-CAM value at the midpoint between the two players (0 to 1). "
                   "It shows whether the CNN was looking there.")
else:
    st.caption("No helmet data loaded, so players are not marked. Add the helmet CSV in the sidebar to see who is involved.")

# ---------------------------------------------------------------------------
# Downloads and model details
# ---------------------------------------------------------------------------
report = pd.DataFrame({
    "sample": range(n), "video_frame": a.clip.frame_nums, "time_s": times, "cnn_score": a.cnn.round(4),
    "lstm_score": np.round(a.lstm_at, 4), "combined_score": a.fused.round(4), "flagged": a.fused >= a.thr,
})
d1, d2, _ = st.columns([1, 1, 2])
d1.download_button("Download this frame (PNG)", png_bytes(np.concatenate([img, players_img, cam_img], 1)),
                   file_name=f"frame_{a.clip.frame_nums[idx]}.png", mime="image/png", width="stretch")
d2.download_button("Download scores (CSV)", report.to_csv(index=False), file_name="contact_scores.csv",
                   mime="text/csv", width="stretch")

with st.expander("About the models and the method"):
    m = ce.REPORTED_METRICS
    st.table(pd.DataFrame([
        {"Model": m[x]["name"], "Scores": m[x]["type"], "Validation accuracy": f"{m[x]['accuracy']:.2f}%",
         "F1": m[x]["f1"], "Design": m[x]["detail"]} for x in ("cnn", "lstm")
    ]))
    st.markdown(
        "- The CNN scores each sampled frame. The LSTM scores each window of 8 sampled frames. "
        "The combined score is their average wherever both exist.\n"
        "- A contact event is a run of sampled frames at or above the threshold, as long as the run is as long as 'Frames in a row'.\n"
        "- Players come from the helmet CSV. A pair is 'close' when the helmet centres are within the chosen number of helmet widths.\n"
        "- The accuracies above come from a random 80/20 split of frames. Frames from the same play can sit in both sets, "
        "so expect lower accuracy on new plays."
    )
