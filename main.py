#!/usr/bin/env python3
from __future__ import annotations
import argparse, logging, os, random, tempfile, warnings
from pathlib import Path
import cv2, librosa, moviepy.editor as mpy, numpy as np, ffmpeg

warnings.filterwarnings("ignore", message="Warning: in file .* bytes wanted but 0 bytes read.*",
                        category=UserWarning, module="moviepy.video.io.ffmpeg_reader")


def _extract_audio(video_path: Path, sr: int = 22050) -> Path:
    tmp_dir = Path(tempfile.mkdtemp())
    wav_path = tmp_dir / "temp_audio.wav"
    with mpy.VideoFileClip(str(video_path)) as clip:
        if clip.audio is None:
            raise ValueError("No audio track found in the video.")
        clip.audio.write_audiofile(str(wav_path), fps=sr, logger=None, verbose=False)
    return wav_path


def _detect_onsets_audio(wav_path: Path, sr: int = 22050) -> np.ndarray:
    y, _ = librosa.load(str(wav_path), sr=sr)
    return librosa.onset.onset_detect(y=y, sr=sr, units="time")


def _detect_onsets_video(clip: mpy.VideoFileClip, threshold: float, fps_sample: int = 10) -> np.ndarray:
    step = int(clip.fps // fps_sample) if clip.fps > fps_sample else 1
    prev_lum, onsets = None, []
    for t in np.arange(0, clip.duration, 1 / clip.fps * step):
        gray = cv2.cvtColor(clip.get_frame(t), cv2.COLOR_RGB2GRAY)
        lum = np.mean(gray)
        if prev_lum is not None and abs(lum - prev_lum) > threshold:
            onsets.append(t)
        prev_lum = lum
    return np.array(onsets, dtype=np.float32)


def get_rotation(path: Path) -> int:
    """Read rotation metadata in degrees."""
    try:
        meta = ffmpeg.probe(str(path))
        for stream in meta["streams"]:
            if stream["codec_type"] == "video" and "tags" in stream and "rotate" in stream["tags"]:
                return int(stream["tags"]["rotate"])
    except Exception:
        pass
    return 0


class TrackedPoint:
    def __init__(self, pos: tuple[float, float], life: int, size: int):
        self.pos = np.array(pos, dtype=np.float32)
        self.life = life
        self.size = size


def _sample_size_bell(min_s: int, max_s: int, width_div: float = 6.0) -> int:
    mean = (min_s + max_s) / 2.0
    sigma = (max_s - min_s) / width_div
    for _ in range(10):
        val = np.random.normal(mean, sigma)
        if min_s <= val <= max_s:
            return int(val)
    return int(np.clip(val, min_s, max_s))


def _adaptive_contrast_color(roi: np.ndarray) -> tuple[int, int, int]:
    """Compute smooth contrast color based on local brightness."""
    if not roi.size:
        return (255, 255, 255)
    avg = np.mean(roi, axis=(0, 1))
    brightness = np.mean(avg)
    fade_factor = np.clip(abs(128 - brightness) / 128, 0.3, 1.0)
    contrast_color = tuple(int((255 - c) * fade_factor + c * (1 - fade_factor)) for c in avg)
    return tuple(int(np.clip(x, 0, 255)) for x in contrast_color)


def auto_orient_clip(clip: mpy.VideoFileClip, video_path: Path) -> mpy.VideoFileClip:
    """Auto-fix orientation based on metadata."""
    rotation = get_rotation(video_path)
    if rotation == 90:
        clip = clip.rotate(-90)
    elif rotation == 270:
        clip = clip.rotate(90)
    elif rotation == 180:
        clip = clip.rotate(180)
    return clip


def render_tracked_effect(
    video_in: Path,
    video_out: Path,
    *,
    fps: float | None,
    pts_per_beat: int,
    ambient_rate: float,
    jitter_px: float,
    life_frames: int,
    min_size: int,
    max_size: int,
    neighbor_links: int,
    orb_fast_threshold: int,
    bell_width: float,
    seed: int | None,
    text_size: float,
    text_color: str | tuple[int, int, int],
    remove_text: bool,
    no_fill: bool,
    ignore_audio: bool,
    video_threshold: float,
    line_mode: str,
    max_box_size: int | None,
):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    with mpy.VideoFileClip(str(video_in)) as clip:
        clip = auto_orient_clip(clip, video_in)
        fps = fps or clip.fps
        frame_w, frame_h = clip.size

        # Onset detection
        if ignore_audio or clip.audio is None:
            logging.info("⚙️ Using video-based onset detection (threshold=%.2f)", video_threshold)
            onset_times = _detect_onsets_video(clip, threshold=video_threshold)
        else:
            wav_path = _extract_audio(video_in)
            onset_times = _detect_onsets_audio(wav_path)
        logging.info("%d onsets detected", len(onset_times))

        orb = cv2.ORB_create(nfeatures=1500, fastThreshold=orb_fast_threshold)
        active, onset_idx, prev_gray = [], 0, None
        color_mode_negative = isinstance(text_color, str) and text_color.lower() == "negative"

        def make_frame(t: float):
            nonlocal prev_gray, onset_idx, active
            frame = clip.get_frame(t)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape

            # Track existing points
            if prev_gray is not None and active:
                prev_pts = np.array([p.pos for p in active], dtype=np.float32).reshape(-1, 1, 2)
                next_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pts, None)
                new_active = []
                for tp, new_pt, ok in zip(active, next_pts.reshape(-1, 2), status.reshape(-1)):
                    if ok and 0 <= new_pt[0] < w and 0 <= new_pt[1] < h and tp.life > 0:
                        tp.pos = new_pt
                        tp.life -= 1
                        if jitter_px > 0:
                            tp.pos += np.random.normal(0, jitter_px, size=2)
                            tp.pos = np.clip(tp.pos, [0, 0], [w - 1, h - 1])
                        new_active.append(tp)
                active = new_active

            # Spawn new points on onset
            while onset_idx < len(onset_times) and t >= onset_times[onset_idx]:
                kps = orb.detect(gray, None)
                kps = sorted(kps, key=lambda k: k.response, reverse=True)
                for kp in kps[:random.randint(1, pts_per_beat)]:
                    x, y = kp.pt
                    if all(np.linalg.norm(tp.pos - (x, y)) >= 10 for tp in active):
                        s = _sample_size_bell(min_size, max_size, bell_width)
                        if max_box_size:
                            s = min(s, max_box_size)
                        active.append(TrackedPoint((x, y), life_frames, s))
                onset_idx += 1

            # Ambient points
            if ambient_rate > 0:
                for _ in range(np.random.poisson(ambient_rate / fps)):
                    x, y = random.uniform(0, w), random.uniform(0, h)
                    s = _sample_size_bell(min_size, max_size, bell_width)
                    if max_box_size:
                        s = min(s, max_box_size)
                    active.append(TrackedPoint((x, y), life_frames, s))

            # Line modes
            if line_mode != "none" and len(active) > 1:
                coords = [tp.pos for tp in active]
                for i, p in enumerate(coords):
                    dists = sorted(
                        [(j, np.linalg.norm(p - coords[j])) for j in range(len(coords)) if j != i],
                        key=lambda x: x[1],
                    )
                    if line_mode == "all":
                        nearest = dists[:neighbor_links]
                    elif line_mode == "near":
                        nearest = [d for d in dists if d[1] < 80][:neighbor_links]
                    else:
                        nearest = []
                    for j, _ in nearest:
                        cv2.line(frame, tuple(p.astype(int)), tuple(coords[j].astype(int)), (200, 200, 255), 1)

            # Draw boxes and text (if enabled)
            for tp in active:
                x, y, s = *tp.pos, tp.size
                tl, br = (max(0, int(x - s // 2)), max(0, int(y - s // 2))), (min(w - 1, int(x + s // 2)), min(h - 1, int(y + s // 2)))

                if not no_fill:
                    roi = frame[tl[1]:br[1], tl[0]:br[0]]
                    if roi.size:
                        frame[tl[1]:br[1], tl[0]:br[0]] = 255 - roi
                cv2.rectangle(frame, tl, br, (200, 200, 255), 1)

                if not remove_text:
                    if color_mode_negative:
                        roi = frame[max(0, tl[1] - 3):min(h, br[1] + 3), max(0, tl[0] - 3):min(w, br[0] + 3)]
                        txt_color = _adaptive_contrast_color(roi)
                    else:
                        txt_color = text_color

                    hex_text = f"0x{random.randint(0, 0xFFFFFFFF):08X}"
                    text_pos = (br[0] + 5, min(br[1], h - 10))
                    cv2.putText(frame, hex_text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, text_size, txt_color, 1, cv2.LINE_AA)

            prev_gray = gray
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        out_clip = mpy.VideoClip(make_frame, duration=clip.duration)
        out_clip = out_clip.set_audio(clip.audio if not ignore_audio else None).set_fps(fps)
        out_clip.write_videofile(str(video_out), codec="libx264", audio_codec="aac" if not ignore_audio else None)

    logging.info("✅ Render complete: %s", video_out)


def main():
    parser = argparse.ArgumentParser(
        description="Render onset-tracked video effects with motion, text, and visual options.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-i", "--input", help="Input video file path", required=True)
    parser.add_argument("-o", "--output", default="output.mp4", help="Output video path")
    parser.add_argument("-t", "--text-size", type=float, default=0.4, help="Hex text size")
    parser.add_argument("-c", "--text-color", nargs="+", default=(255, 255, 255),
                        help='Text color (B G R) or "negative" for adaptive auto-contrast')
    parser.add_argument("-r", "--remove-text", action="store_true", help="Remove text overlay entirely")
    parser.add_argument("-n", "--no-fill", action="store_true", help="Disable box inversion fill")
    parser.add_argument("-a", "--ignore-audio", action="store_true", help="Ignore audio and use video intensity threshold")
    parser.add_argument("-vth", "--video-threshold", type=float, default=1.0, help="Video intensity change threshold")
    parser.add_argument("-l", "--line-mode", choices=["all", "none", "near"], default="all",
                        help="Connection line mode")
    parser.add_argument("-m", "--max-box-size", type=int, default=None, help="Set maximum box size (override)")

    args = parser.parse_args()

    if len(args.text_color) == 1 and args.text_color[0].lower() == "negative":
        text_color = "negative"
    else:
        text_color = tuple(map(int, args.text_color))

    if not os.path.isfile(args.input):
        print("❌ Error: Input file does not exist.")
        return

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    render_tracked_effect(
        video_in=Path(args.input),
        video_out=Path(args.output),
        fps=None,
        pts_per_beat=20,
        ambient_rate=5.0,
        jitter_px=0.5,
        life_frames=10,
        min_size=15,
        max_size=40,
        neighbor_links=3,
        orb_fast_threshold=20,
        bell_width=4.0,
        seed=None,
        text_size=args.text_size,
        text_color=text_color,
        remove_text=args.remove_text,
        no_fill=args.no_fill,
        ignore_audio=args.ignore_audio,
        video_threshold=args.video_threshold,
        line_mode=args.line_mode,
        max_box_size=args.max_box_size,
    )


if __name__ == "__main__":
    main()
