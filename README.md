BlobTrack is a simple Python-based implementation of blob tracking for videos, made without TouchDesigner and runs smoothly without requiring a GPU.

The script supports:
- 🧭 **Auto orientation** for portrait and landscape videos
- 🎵 **Audio or video-based onset detection**
- ✨ **Customizable text, color, size, and change rate**
- 🔗 **Adjustable connection line modes**
- 🔲 **Configurable box size, life span, and behavior**
- 🔍 **Multiple detection methods** (ORB, Contour, Hough, Background Subtraction, Optical Flow, Color, Edge Density)

---

## ⚙️ Features

| Feature | Description |
|----------|-------------|
| 🎧 Audio onset detection | Reacts to beats in the soundtrack using Librosa |
| 🎥 Video threshold onset | Detects motion or brightness changes for silent videos |
| 🔄 Auto orientation | Automatically fixes sideways portrait videos |
| 🔍 Multiple detection methods | Choose from 8 different detection algorithms (ORB, Contour, Hough Lines/Circles, Background Subtraction, Dense Optical Flow, Color Range, Edge Density) |
| 🧮 Box customization | Control maximum size, lifespan, and spawn rate |
| 🧾 Text overlays | Displays random 8-digit hexadecimal codes beside boxes |
| 🌓 Adaptive colors | `--text-color negative` makes text contrast auto-fit each box region |
| ⏱️ Configurable text rate | `--text-rate` controls how frequently the hex text changes, in milliseconds |
| ⚡ Lightweight CLI | Fast, no GUI required — works directly in terminal |
| ✨ Curved lines | Enable curved connections with configurable maximum curvature |
| 🔗 Distance-based connections | Control connections based on distance with automatic scaling |

---

## 🧰 Installation

### 1. Clone or copy the project
```bash
git clone https://github.com/yourusername/BlobTrack.git
cd BlobTrack
```

### 2. Install dependencies

Make sure you have Python 3.9+ installed.

```bash
pip install -r requirements.txt
```

### 3. Install FFmpeg

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
| `-t`   | `--text-size`       | `0.4`         | Size of the random hexadecimal text                            |
| `-r`   | `--remove-text`     | *(flag)*      | Remove text entirely                                           |
| `-c`   | `--text-color`      | `255 255 255` | Text color in B G R format, or `"negative"` for adaptive color |
| `-tr`  | `--text-rate`       | `0.0`         | Text change rate in milliseconds (0 = change every frame)      |
| `-n`   | `--no-fill`         | *(flag)*      | Disable box inversion (transparent boxes)                      |
| `-a`   | `--ignore-audio`    | *(flag)*      | Ignore audio; use video intensity changes instead              |
| `-vth` | `--video-threshold` | `1.0`         | Sensitivity for motion detection; higher = less sensitive      |
| `-l`   | `--line-distance`   | `80`          | Distance threshold for connecting points (0 for none, higher values for larger distances) |
| `-m`   | `--max-box-size`    | `None`        | Maximum allowed box size                                       |
|        | `--detection-method`| `orb`         | Detection method: `orb`, `contour`, `hl` (Hough lines), `hc` (Hough circles), `bgs` (background subtraction), `dof` (dense optical flow), `cr` (color range), `ed` (edge density) |
|        | `--curved-lines`    | *(flag)*      | Enable curved lines with distance-based curvature              |
|        | `--max-curvature`   | `0.5`         | Maximum curvature for curved lines (0.0 to 1.0)                |
|        | `--contour-min-area`| `50`          | Minimum contour area for contour detection                     |
|        | `--contour-max-area`| `5000`        | Maximum contour area for contour detection                     |
|        | `--hough-lines-threshold` | `100`   | Threshold parameter for Hough lines detection                  |
|        | `--hough-lines-min-length` | `30`   | Minimum line length for Hough lines detection                  |
|        | `--hough-lines-max-gap` | `10`     | Maximum gap between line segments for Hough lines detection    |
|        | `--hough-circles-min-radius` | `10` | Minimum circle radius for Hough circles detection              |
|        | `--hough-circles-max-radius` | `50` | Maximum circle radius for Hough circles detection              |
|        | `--hough-circles-param1` | `50`    | Parameter 1 for Hough circles detection                        |
|        | `--hough-circles-param2` | `30`    | Parameter 2 for Hough circles detection                        |
|        | `--bg-sub-learning-rate` | `0.0`   | Learning rate for background subtraction (0.0 = auto)          |
|        | `--bg-sub-detect-shadows` | *(flag)*| Enable shadow detection for background subtraction              |
|        | `--optical-flow-threshold` | `15.0`| Motion threshold for dense optical flow detection              |
|        | `--color-lower-hsv` | `[0, 50, 50]` | Lower HSV bounds for color detection [H, S, V]                 |
|        | `--color-upper-hsv` | `[10, 255, 255]`| Upper HSV bounds for color detection [H, S, V]                |
|        | `--edge-density-threshold` | `0.1` | Edge density threshold factor                                  |
|        | `--edge-density-grid-size` | `32`  | Grid cell size for edge density analysis                       |

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

### 🌈 Adaptive Color + Distance-based Lines

```bash
python3 main.py -i dance.mp4 -o styled.mp4 -c negative -l 50
```

### 🧱 Larger Boxes, No Fill

```bash
python3 main.py -i city.mp4 -o abstract.mp4 -m 80 -n
```

### 🔍 Contour Detection with Custom Sensitivity

```bash
python3 main.py --detection-method contour --contour-min-area 30 --contour-max-area 1000 -i input.mp4 -o output.mp4
```

### 🎨 Hough Lines Detection with Curved Lines

```bash
python3 main.py --detection-method hl --curved-lines --max-curvature 0.7 -i input.mp4 -o output.mp4
```

### 🌈 Color Range Detection (Red Objects)

```bash
python3 main.py --detection-method cr --color-lower-hsv 0 50 50 --color-upper-hsv 15 255 255 -i input.mp4 -o output.mp4
```

### 📊 Background Subtraction Detection

```bash
python3 main.py --detection-method bgs --bg-sub-detect-shadows -i input.mp4 -o output.mp4
```

### ⏱️ Control Text Change Rate

```bash
python3 main.py --text-rate 500 -i input.mp4 -o output.mp4  # Change text every 500ms
```

---

## 🧩 Tips & Notes

* Lower `--video-threshold` → more sensitive to small movements
* `--text-color negative` automatically adapts brightness for better contrast
* Portrait videos will output correctly (no more sideways video)
* The script maintains your input resolution exactly (no stretching)
* `--line-distance` scales automatically with video frame size
* Use `--curved-lines` with `--max-curvature` to create flowing visual connections
* Different detection methods work best for different content types:
  - `orb`: General purpose feature detection
  - `contour`: Good for object shapes with clear boundaries
  - `hl`: For linear structures and edges
  - `hc`: For circular/round objects
  - `bgs`: For moving objects against static backgrounds
  - `dof`: For motion patterns and flows
  - `cr`: For specific colored objects
  - `ed`: For textured areas and edge-rich regions

---

## 🧾 Requirements

Included in [`requirements.txt`](./requirements.txt):

```text
moviepy>=2.0.0
opencv-python>=4.8.0
librosa>=0.10.1
numpy>=1.25.0
ffmpeg-python>=0.2.0
```

Install with:

```bash
pip install -r requirements.txt
```