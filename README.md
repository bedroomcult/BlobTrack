BlobTrack is a simple Python-based implementation of blob tracking for videos, made without TouchDesigner and runs smoothly without requiring a GPU.

The script supports:
- 🧭 **Auto orientation** for portrait and landscape videos  
- 🎵 **Audio or video-based onset detection**  
- ✨ **Customizable text, color, and size**  
- 🔗 **Adjustable connection line modes**  
- 🔲 **Configurable box size, life span, and behavior**  

---

## ⚙️ Features

| Feature | Description |
|----------|-------------|
| 🎧 Audio onset detection | Reacts to beats in the soundtrack using Librosa |
| 🎥 Video threshold onset | Detects motion or brightness changes for silent videos |
| 🔄 Auto orientation | Automatically fixes sideways portrait videos |
| 🔗 Line modes | Choose how boxes connect (`original`, `none`, `nearby`) |
| 🧮 Box customization | Control maximum size, lifespan, and spawn rate |
| 🧾 Text overlays | Displays random 8-digit hexadecimal codes beside boxes |
| 🌓 Adaptive colors | `--text-color negative` makes text contrast auto-fit each box region |
| ⚡ Lightweight CLI | Fast, no GUI required — works directly in terminal |

---

## 🧰 Installation

### 1. Clone or copy the project
```bash
git clone https://github.com/yourusername/onset-visualizer.git
cd onset-visualizer
````

### 2. Install dependencies

Make sure you have Python 3.9+ installed.

```bash
pip install -r requirements.txt
```

### 3. (Optional) Install FFmpeg

MoviePy requires FFmpeg.
On Linux:

```bash
sudo apt install ffmpeg
```

On Windows:
Download and add FFmpeg to your PATH from [ffmpeg.org](https://ffmpeg.org/download.html).

---

## 🚀 Usage

Run the script directly from your terminal:

```bash
python3 main.py -i input.mp4 -o output.mp4
```

---

## 🧩 Command-Line Arguments

| Short  | Long                | Default       | Description                                                    |
| ------ | ------------------- | ------------- | -------------------------------------------------------------- |
| `-i`   | `--input`           | *(required)*  | Input video file path                                          |
| `-o`   | `--output`          | `output.mp4`  | Output video path                                              |
| `-t`   | `--text-size`       | `*(flag)*      | Remove text entirely                        |
| `-r`   | `--remove-text`       | `0.4`         | Size of the random hexadecimal text                            |
| `-c`   | `--text-color`      | `255 255 255` | Text color in B G R format, or `"negative"` for adaptive color |
| `-n`   | `--no-fill`         | *(flag)*      | Disable box inversion (transparent boxes)                      |
| `-a`   | `--ignore-audio`    | *(flag)*      | Ignore audio; use video intensity changes instead              |
| `-vth` | `--video-threshold` | `1.0`         | Sensitivity for motion detection; higher = less sensitive      |
| `-l`   | `--line-mode`       | `all`    | Connection line style: `all`, `none`, `near`            |
| `-m`   | `--max-box-size`    | `None`        | Maximum allowed box size                                       |

---

## 🪄 Examples

### 🔊 React to Music Beats

```bash
python3 main.py -i concert.mp4 -o beats.mp4
```

### 🎥 Silent Video Mode (Motion-Based)

```bash
python3 main.py -i timelapse.mp4 -o visual.mp4 -a --video-threshold 2.5
```

### 🌈 Adaptive Color + Nearby Lines Only

```bash
python3 main.py -i dance.mp4 -o styled.mp4 -c negative -l nearby
```

### 🧱 Larger Boxes, No Fill

```bash
python3 main.py -i city.mp4 -o abstract.mp4 -m 80 -n
```

---

## 🧩 Tips & Notes

* Lower `--video-threshold` → more sensitive to small movements
* `--text-color negative` automatically adapts brightness for better contrast
* Portrait videos will output correctly (no more sideways video)
* The script maintains your input resolution exactly (no stretching)
* `--line-mode nearby` is great for dense videos (prevents messy linking)

---

## 🧾 Requirements

Included in [`requirements.txt`](./requirements.txt):

```text
moviepy>=1.0.3
opencv-python>=4.8.0
librosa>=0.10.1
numpy>=1.25.0
ffmpeg-python>=0.2.0
```

Install with:

```bash
pip install -r requirements.txt
```