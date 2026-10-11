# Contact review: NFL player contact detection app

A Streamlit front end for the CNN + LSTM models in this project. Upload a play video and the app:

- scores every sampled frame for player contact (CNN per frame, LSTM per 8-frame window),
- marks which players are close together and likely involved (needs the Kaggle helmet CSV),
- shows where the CNN looked (Grad-CAM heatmap),
- writes plain-language guidance on what to check next.

## Folder layout

```
app.py               Streamlit interface
contact_engine.py    models, inference, Grad-CAM, pair analysis, guidance text
requirements.txt
.streamlit/config.toml
weights/cnn_model2/best_weights.pth     <- you copy these in
weights/lstm_model1/best_weights.pth
data/                                   <- optional Kaggle files
```

## 1. Install

Python 3.10 or newer.

```bash
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # macOS / Linux
pip install -r requirements.txt
```

## 2. Add the trained weights

Copy these two files from your project folder (`DL project/`):

| From | To |
|---|---|
| `cnn_model2/best_weights.pth` | `weights/cnn_model2/best_weights.pth` |
| `lstm_model1/best_weights.pth` | `weights/lstm_model1/best_weights.pth` |

Use `lstm_model1`, not `lstm_model2`. The saved `lstm_model2` file (about 3 MB) does not match the notebook's
`LSTM_Improved` class (about 0.5 MB), so it will not load into this app.

## 3. Get a play to analyse (optional, for player names)

The app works on any video and gives a whole-frame result. To also see which players are involved, download from the
[Kaggle competition](https://www.kaggle.com/competitions/nfl-player-contact-detection/data) (accept the rules first):

- a video such as `train/58168_003392_Sideline.mp4` (keep the `gameplay_view` file name),
- `train_baseline_helmets.csv` into `data/`,
- `train_labels.csv` into `data/` if you want ground truth shown next to the predictions.

```bash
kaggle competitions download -c nfl-player-contact-detection -f train/58168_003392_Sideline.mp4
kaggle competitions download -c nfl-player-contact-detection -f train_baseline_helmets.csv
```

The helmet CSV is large, so the first read takes a while. To make it instant, cut it down to one play and upload that
in the sidebar instead:

```python
import pandas as pd
df = pd.concat(c[c.game_play == "58168_003392"] for c in pd.read_csv("data/train_baseline_helmets.csv", chunksize=500_000))
df.to_csv("data/one_play_helmets.csv", index=False)
```

## 4. Run

```bash
streamlit run app.py
```

Open http://localhost:8501, upload the video, press **Analyse contact**.

## Reading the result

- **Scoreboard**: verdict, CNN and LSTM scores at the key (peak) frame, players tracked, and how many frames were flagged.
- **What to do next**: tier of confidence, what was found, numbered steps, and cautions.
- **Timeline**: CNN, LSTM and combined score against time, with the threshold line.
- **Frame viewer**: step through sampled frames. Red links are probable contact pairs, amber links are close pairs the
  models did not flag, and the right-hand image is the Grad-CAM heatmap.

A contact event is a run of sampled frames whose combined score is at or above the threshold. Threshold, run length and
close-pair distance are in the sidebar and update instantly. Frames to analyse and frame spacing need a new analysis.

## What the app can and cannot say

- The models classify whole frames. Player pairs are inferred from helmet-box distance, so nearby players are not
  always touching.
- The reported accuracies (CNN 86.46%, LSTM 71.35%) come from a random 80/20 frame split. Frames from the same play can be in
  both sets, so expect lower accuracy on new plays.
- Output is review support, not an officiating decision.

## Put it on GitHub

The CNN weights are about 103 MB, above GitHub's 100 MB file limit, so they go through Git LFS (already set in `.gitattributes`).

```bash
git init
git lfs install
git add .
git commit -m "Add Streamlit contact review app"
git branch -M main
git remote add origin https://github.com/<your-username>/<repo>.git
git push -u origin main
```

If you would rather not use LFS, delete `.gitattributes`, leave out `weights/cnn_model2/` and attach that file to a GitHub Release.
