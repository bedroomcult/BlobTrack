BlobTrack is a ~~simple~~ Python-based implementation of blob tracking for videos, made without TouchDesigner and runs smoothly without requiring a GPU.

The script supports:
- 🧭 **Auto orientation** for portrait and landscape videos
- 🎵 **Audio or video-based onset detection**
- ✨ **Customizable text, color, size, and change rate**
- 🔗 **Adjustable connection line modes**
- 🔲 **Configurable box size, life span, and behavior**
- 🔍 **Multiple detection methods** (ORB, Contour, Hough, Background Subtraction, Optical Flow, Color, Edge Density, Temporal Difference)

---

## ⚙️ Features

| Feature | Description |
|----------|-------------|
| 🎧 Audio onset detection | Reacts to beats in the soundtrack using Librosa |
| 🎥 Video threshold onset | Detects motion or brightness changes for silent videos |
| 🔄 Auto orientation | Automatically fixes sideways portrait videos |
| 🔍 Multiple detection methods | Choose from 9 different detection algorithms (ORB, Contour, Hough Lines/Circles, Background Subtraction, Dense Optical Flow, Color Range, Edge Density, Temporal Difference) |
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
| `-t`   | `--text-size`       | `0.4`         | Size of the random hexadecimal text (0.1-2.0)                  |
| `-r`   | `--remove-text`     | *(flag)*      | Remove text entirely                                           |
| `-c`   | `--text-color`      | `255 255 255` | Text color in B G R format, or `"negative"` for adaptive color |
| `-tr`  | `--text-rate`       | `0.0`         | Text change rate in milliseconds (0 = change every frame)      |
| `-n`   | `--no-fill`         | *(flag)*      | Disable box inversion (transparent boxes)                      |
| `-a`   | `--ignore-audio`    | *(flag)*      | Ignore audio; use video intensity changes instead              |
| `-vth` | `--video-threshold` | `1.0`         | Video intensity change threshold (0.1-5.0): lower = more sensitive to small changes |
| `-ld`  | `--life-duration`   | `333.0`       | Duration in milliseconds that tracked points remain active (333ms ≈ 10 frames at 30fps) |
| `-l`   | `--line-distance`   | `80`          | Distance threshold for connecting points (0-99 as percentage of frame size, 100+ connects all points) |
|        | `--line-stability`  | *(flag)*      | Enable stable line connections that persist even when points move (reduces blinking) |
| `-m`   | `--max-box-size`    | `None`        | Maximum allowed box size (10-200 pixels)                       |
|        | `--detection-method`| `orb`         | Detection method: `orb`, `contour`, `hl` (Hough lines), `hc` (Hough circles), `bgs` (background subtraction), `dof` (dense optical flow), `cr` (color range), `ed` (edge density), `td` (temporal difference) |
|        | `--curved-lines`    | *(flag)*      | Enable curved lines with distance-based curvature              |
|        | `--max-curvature`   | `0.5`         | Maximum curvature for curved lines (0.0-1.0): higher = more curved |
|        | `--ct-min-area`     | `50`          | Minimum contour area (10-1000): lower = detect smaller objects |
|        | `--ct-max-area`     | `5000`        | Maximum contour area (100-10000): higher = allow larger objects |
|        | `--hl-threshold`    | `100`         | Hough lines threshold (50-200): lower = detect more lines      |
|        | `--hl-min-length`   | `30`          | Minimum line length (10-100): lower = detect shorter lines     |
|        | `--hl-max-gap`      | `10`          | Maximum gap between line segments (5-50): higher = connect more distant segments |
|        | `--hc-min-radius`   | `10`          | Minimum circle radius (5-50): lower = detect smaller circles   |
|        | `--hc-max-radius`   | `50`          | Maximum circle radius (20-200): higher = detect larger circles |
|        | `--hc-param1`       | `50`          | Hough circles param1 (30-100): lower = more sensitive detection |
|        | `--hc-param2`       | `30`          | Hough circles param2 (20-80): lower = detect more circles      |
|        | `--bgs-learning-rate`| `0.0`        | Background subtraction learning rate (0.0 = auto, 0.001-0.1)   |
|        | `--bgs-detect-shadows`| *(flag)*     | Enable shadow detection for background subtraction              |
|        | `--dof-threshold`   | `15.0`        | Optical flow motion threshold (5.0-50.0): lower = detect smaller movements |
|        | `--cr-lower-hsv`    | `[0, 50, 50]` | Lower HSV bounds for color detection [H, S, V]                 |
|        | `--cr-upper-hsv`    | `[10, 255, 255]`| Upper HSV bounds for color detection [H, S, V]                |
|        | `--ed-threshold`    | `0.1`         | Edge density threshold (0.05-0.5): lower = detect more textured areas |
|        | `--ed-grid-size`    | `32`          | Edge density grid cell size (16-128): smaller = finer analysis |
|        | `--td-delay`        | `1`           | Temporal difference frame delay (1-10): higher = detect slower movements |
|        | `--td-threshold`    | `30`          | Temporal difference threshold (10-100): lower = detect smaller movements |
| `-va`  | `--verbose-all`     | *(flag)*      | Analyze all detection methods with maximum sensitivity (no video output) |
|        | `--sample-frames`    | `20`          | Number of frames to sample for --verbose-all analysis (1-100)            |
|        | `--max`              | *(flag)*      | Force maximum sensitivity for all detection methods (overrides other settings). This args is not recommended when using Temporal difference (td) detection method |

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
python3 main.py --detection-method contour --ct-min-area 30 --ct-max-area 1000 -i input.mp4 -o output.mp4
```

### 🎨 Hough Lines Detection with Curved Lines

```bash
python3 main.py --detection-method hl --hl-threshold 80 --curved-lines --max-curvature 0.7 -i input.mp4 -o output.mp4
```

### 🌈 Color Range Detection (Red Objects)

```bash
python3 main.py --detection-method cr --cr-lower-hsv 0 50 50 --cr-upper-hsv 15 255 255 -i input.mp4 -o output.mp4
```

### 📊 Background Subtraction Detection

```bash
python3 main.py --detection-method bgs --bgs-detect-shadows -i input.mp4 -o output.mp4
```

### ⚡ Temporal Difference Motion Detection

```bash
python3 main.py --detection-method td --td-delay 2 --td-threshold 25 -i input.mp4 -o output.mp4
```

### 🔪 Edge Density with Fine Analysis

```bash
python3 main.py --detection-method ed --ed-threshold 0.08 --ed-grid-size 24 -i input.mp4 -o output.mp4
```

### ⏱️ Control Text Change Rate

```bash
python3 main.py --text-rate 500 -i input.mp4 -o output.mp4  # Change text every 500ms
```

### 🔍 Analyze All Detection Methods

```bash
python3 main.py --verbose-all -i input.mp4  # Compare all methods with max sensitivity
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
  - `td`: For motion detection using temporal differencing

---

## 📚 Parameter Guide

### 🎥 Video Threshold (`--video-threshold`)
Controls sensitivity for video-based onset detection when audio is ignored.
- **Lower values (0.1-0.5)**: More sensitive, detects subtle brightness/luminance changes. Good for quiet scenes or gradual transitions.
- **Higher values (2.0-5.0)**: Less sensitive, only detects significant changes. Good for noisy videos or when you want fewer onsets.
- **Default (1.0)**: Balanced sensitivity for most videos.

### 🔗 Line Distance (`--line-distance`)
Controls how points connect to each other with lines.
- **0**: No connections between points
- **1-99**: Percentage of frame size as connection threshold (e.g., 50 = connect points within 50% of frame diagonal)
- **100+**: Connect all points to each other (creates dense networks)

### 📏 Contour Detection Parameters
- **`--ct-min-area`**: Filter out small noise/contours. Lower = detect smaller objects, Higher = only large objects
- **`--ct-max-area`**: Prevent detection of very large areas. Lower = avoid huge detections, Higher = allow massive objects

### 📐 Hough Lines Parameters
- **`--hl-threshold`**: Minimum votes for line detection. Lower = more lines detected (may include noise), Higher = fewer, cleaner lines
- **`--hl-min-length`**: Shortest line to detect. Lower = detect short segments, Higher = only long lines
- **`--hl-max-gap`**: Maximum gap between line segments. Higher = connect distant segments into longer lines

### ⭕ Hough Circles Parameters
- **`--hc-min-radius` / --hc-max-radius`**: Circle size range. Smaller range = more precise detection
- **`--hc-param1`**: Edge detection sensitivity. Lower = more sensitive (may detect false circles)
- **`--hc-param2`**: Circle detection strictness. Lower = detect more circles (including imperfect ones)

### 🎨 Background Subtraction Parameters
- **`--bgs-learning-rate`**: How fast background model adapts. 0.0 = auto-adapt, 0.001-0.01 = slow adapt (stable backgrounds), 0.1 = fast adapt (changing backgrounds)
- **`--bgs-detect-shadows`**: Enable shadow detection to avoid false positives from moving shadows

### 🌊 Optical Flow Parameters
- **`--dof-threshold`**: Minimum motion magnitude to detect. Lower = detect subtle movements, Higher = only significant motion

### 🎨 Color Range Parameters
- **`--cr-lower-hsv` / --cr-upper-hsv`**: HSV color bounds [Hue, Saturation, Value]
  - Hue: 0-179 (Red=0/179, Green=60, Blue=120)
  - Saturation: 0-255 (0=grayscale, 255=full color)
  - Value: 0-255 (0=black, 255=white)

### 🔪 Edge Density Parameters
- **`--ed-threshold`**: Minimum edge density in grid cells. Lower = detect more textured areas, Higher = only high-contrast regions
- **`--ed-grid-size`**: Analysis grid resolution. Smaller = finer detail but slower, Larger = coarser but faster

### ⏰ Temporal Difference Parameters
- **`--td-delay`**: Frames to compare against. 1 = fast movements, Higher = slower movements
- **`--td-threshold`**: Pixel difference threshold. Lower = detect smaller movements, Higher = only large motions

### 🎛️ Text and Visual Parameters
- **`--text-size`**: Font scale multiplier. 0.1 = tiny text, 2.0 = large text
- **`--text-rate`**: Milliseconds between text changes. 0 = change every frame, 1000 = change every second
- **`--max-box-size`**: Largest allowed tracking box. None = no limit, 50-100 = reasonable bounds
- **`--max-curvature`**: Line curvature randomness. 0.0 = straight lines, 1.0 = highly curved

### 💡 Usage Tips
- **For busy videos**: Increase thresholds to reduce noise (`--video-threshold 2.0`, `--hl-threshold 150`)
- **For subtle content**: Decrease thresholds for more sensitivity (`--video-threshold 0.3`, `--ct-min-area 20`)
- **For performance**: Use larger grid sizes and higher thresholds
- **For precision**: Use smaller ranges and lower thresholds
- **For artistic effects**: Experiment with extreme values (very low/high thresholds)

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
