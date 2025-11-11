#!/usr/bin/env python3
from __future__ import annotations
import argparse, logging, os, random, tempfile, warnings
from pathlib import Path
import cv2, numpy as np, ffmpeg
from moviepy import VideoFileClip, VideoClip

# Try to import librosa, make it optional
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    librosa = None

warnings.filterwarnings("ignore", message="Warning: in file .* bytes wanted but 0 bytes read.*",
                        category=UserWarning, module="moviepy.video.io.ffmpeg_reader")


def _extract_audio(video_path: Path, sr: int = 22050) -> Path:
    tmp_dir = Path(tempfile.mkdtemp())
    wav_path = tmp_dir / "temp_audio.wav"
    with VideoFileClip(str(video_path)) as clip:
        if clip.audio is None:
            raise ValueError("No audio track found in the video.")
        clip.audio.write_audiofile(str(wav_path), fps=sr, logger=None)
    return wav_path


def _detect_onsets_audio(wav_path: Path, sr: int = 22050) -> np.ndarray:
    if not LIBROSA_AVAILABLE:
        logging.warning("⚠️ Librosa not available, using video-based detection instead")
        return np.array([])  # Return empty array if librosa is not available
    
    y, _ = librosa.load(str(wav_path), sr=sr)
    return librosa.onset.onset_detect(y=y, sr=sr, units="time")


def _detect_onsets_video(clip: VideoFileClip, threshold: float, fps_sample: int = 10) -> np.ndarray:
    step = int(clip.fps // fps_sample) if clip.fps > fps_sample else 1
    prev_lum, onsets = None, []
    for t in np.arange(0, clip.duration, 1 / clip.fps * step):
        gray = cv2.cvtColor(clip.get_frame(t), cv2.COLOR_RGB2GRAY)
        lum = np.mean(gray)
        if prev_lum is not None and abs(lum - prev_lum) > threshold:
            onsets.append(t)
        prev_lum = lum
    return np.array(onsets, dtype=np.float32)


def _detect_onsets_time_based(clip: VideoFileClip, interval: float) -> np.ndarray:
    """Generate onsets at regular time intervals."""
    onset_times = []
    t = 0.0
    while t < clip.duration:
        onset_times.append(t)
        t += interval
    return np.array(onset_times, dtype=np.float32)


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
    _next_id = 0  # Class variable to generate unique IDs

    def __init__(self, pos: tuple[float, float], life: int, size: int, text_rate_ms: float = 0):
        self.id = TrackedPoint._next_id
        TrackedPoint._next_id += 1
        self.pos = np.array(pos, dtype=np.float32)
        self.life = life
        self.size = size
        self.text_rate_ms = text_rate_ms  # Text change rate in milliseconds
        self.last_text_update = 0  # Track time of last text update
        self.current_text = f"0x{random.randint(0, 0xFFFFFFFF):08X}"  # Current text


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


def _detect_contours(gray_frame, min_area=50, max_area=5000):
    """Detect contours in a grayscale frame and return keypoints at their centers."""
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray_frame, (5, 5), 0)
    
    # Use adaptive threshold to handle varying lighting
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    
    # Alternative: Use Canny edge detection
    # edges = cv2.Canny(blurred, 30, 100)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    keypoints = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area < area < max_area:  # Filter by area
            # Calculate the center of the contour
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                # Create a keypoint at the center of the contour
                kp = cv2.KeyPoint(float(cx), float(cy), size=np.sqrt(area))
                keypoints.append(kp)
    
    return keypoints


def _detect_hough_lines(gray_frame, threshold=100, min_line_length=30, max_line_gap=10):
    """Detect lines in a grayscale frame using Hough transform and return keypoints at line centers."""
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray_frame, (5, 5), 0)
    
    # Use Canny edge detection to find edges
    edges = cv2.Canny(blurred, 50, 150)
    
    # Apply HoughLinesP to detect line segments
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=threshold, minLineLength=min_line_length, maxLineGap=max_line_gap)
    
    keypoints = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Calculate the center of the line
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            
            # Calculate the length of the line to use as the keypoint size
            length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            
            # Create a keypoint at the center of the line
            kp = cv2.KeyPoint(float(cx), float(cy), size=length)
            keypoints.append(kp)
    
    return keypoints


def _detect_hough_circles(gray_frame, min_radius=10, max_radius=50, param1=50, param2=30):
    """Detect circles in a grayscale frame using HoughCircles transform."""
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray_frame, (9, 9), 2)
    
    # The input gray_frame is already grayscale, so we use it directly
    # Apply HoughCircles to detect circles
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=20,
        param1=param1,
        param2=param2,
        minRadius=min_radius,
        maxRadius=max_radius
    )
    
    keypoints = []
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            # Create a keypoint at the center of the circle
            kp = cv2.KeyPoint(float(x), float(y), float(r*2))  # Using diameter as size
            keypoints.append(kp)
    
    return keypoints


def _detect_background_subtraction(gray_frame, bg_subtractor):
    """Detect moving objects using background subtraction."""
    # Apply background subtraction to get foreground mask
    fg_mask = bg_subtractor.apply(gray_frame)
    
    # Clean up the mask with morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
    
    # Find contours in the foreground mask
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    keypoints = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 50:  # Filter out small noise
            # Calculate the center of the contour
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                # Create a keypoint at the center of the moving object
                kp = cv2.KeyPoint(float(cx), float(cy), size=np.sqrt(area))
                keypoints.append(kp)
    
    return keypoints


def _detect_dense_optical_flow(prev_gray, curr_gray, threshold=15.0):
    """Detect areas of high motion using dense optical flow."""
    # Calculate dense optical flow using Farneback method
    flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    
    # Calculate motion magnitude
    magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
    
    # Threshold to find areas with significant motion
    motion_mask = (magnitude > threshold).astype(np.uint8) * 255
    
    # Clean up the mask with morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel)
    motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, kernel)
    
    # Find contours in the motion mask
    contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    keypoints = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 30:  # Filter out small regions
            # Calculate the center of the contour
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                # Create a keypoint at the center of the motion region
                kp = cv2.KeyPoint(float(cx), float(cy), size=np.sqrt(area))
                keypoints.append(kp)
    
    return keypoints


def _detect_color_range(bgr_frame, lower_color=(0, 50, 50), upper_color=(10, 255, 255), min_area=30):
    """Detect objects within a specific color range in HSV space."""
    # Convert BGR to HSV
    hsv = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2HSV)
    
    # Create mask for the specified color range
    mask = cv2.inRange(hsv, lower_color, upper_color)
    
    # Clean up the mask with morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # Find contours in the color mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    keypoints = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_area:  # Filter out small regions
            # Calculate the center of the contour
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                # Create a keypoint at the center of the colored object
                kp = cv2.KeyPoint(float(cx), float(cy), size=np.sqrt(area))
                keypoints.append(kp)
    
    return keypoints


def _detect_edge_density(gray_frame, grid_size=32, threshold_factor=0.1):
    """Detect areas with high edge density using grid-based analysis."""
    # Apply Canny edge detection
    edges = cv2.Canny(gray_frame, 50, 150)

    h, w = edges.shape
    keypoints = []

    # Divide the frame into a grid and calculate edge density in each cell
    for y in range(0, h, grid_size):
        for x in range(0, w, grid_size):
            # Define the grid cell
            cell = edges[y:y+grid_size, x:x+grid_size]
            edge_count = np.sum(cell > 0)
            total_pixels = cell.shape[0] * cell.shape[1]

            # Calculate edge density
            if total_pixels > 0:
                edge_density = edge_count / total_pixels

                # If edge density is above threshold, add a keypoint at the center of the cell
                if edge_density > threshold_factor:
                    center_x = x + grid_size // 2
                    center_y = y + grid_size // 2
                    # Use edge density as the keypoint size (normalized)
                    kp = cv2.KeyPoint(float(center_x), float(center_y), size=edge_density * 100)
                    keypoints.append(kp)

    return keypoints


def _detect_temporal_difference(gray_frame, frame_history, frame_delay=1, threshold=30):
    """Detect motion using temporal differencing (frame comparison)."""
    # Add current frame to history
    frame_history.append(gray_frame.copy())

    keypoints = []

    # Only proceed if we have enough frames for comparison
    if len(frame_history) >= frame_delay:
        # Get the reference frame (frame_delay steps ago)
        reference_frame = frame_history[0]

        # Get the current frame (most recent)
        current_frame = frame_history[-1]

        # Calculate absolute difference
        diff_frame = cv2.absdiff(current_frame, reference_frame)

        # Apply threshold to create binary motion mask
        _, thresholded_diff = cv2.threshold(diff_frame, threshold, 255, cv2.THRESH_BINARY)

        # Clean up the motion mask with morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        thresholded_diff = cv2.morphologyEx(thresholded_diff, cv2.MORPH_OPEN, kernel)
        thresholded_diff = cv2.morphologyEx(thresholded_diff, cv2.MORPH_CLOSE, kernel)

        # Find contours in the motion mask
        contours, _ = cv2.findContours(thresholded_diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 50:  # Filter out small motion regions
                # Calculate the center of the contour
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    # Create a keypoint at the center of the motion region
                    kp = cv2.KeyPoint(float(cx), float(cy), size=np.sqrt(area))
                    keypoints.append(kp)

    return keypoints


def draw_curved_line(img, start_point, end_point, distance, color, thickness, max_curvature=0.5):
    """
    Draw a curved line between two points with random curvature up to max_curvature.
    """
    import math
    
    start = np.array(start_point, dtype=np.float32)
    end = np.array(end_point, dtype=np.float32)
    
    # Calculate midpoint
    mid = (start + end) / 2.0
    
    # Calculate perpendicular vector to the line
    direction = end - start
    perpendicular = np.array([-direction[1], direction[0]], dtype=np.float32)
    
    # Normalize the perpendicular vector
    length = np.linalg.norm(perpendicular)
    if length == 0:
        cv2.line(img, start_point, end_point, color, thickness)
        return
    
    perpendicular = perpendicular / length
    
    # Use random curvature up to max_curvature
    # Randomly decide if curve goes "up" or "down" from the line
    random_factor = (2 * np.random.random() - 1)  # Random value between -1 and 1
    curvature = random_factor * max_curvature
    
    # Scale the perpendicular vector by distance and random curvature
    control_offset = perpendicular * (np.linalg.norm(direction) * 0.1 * curvature)
    
    # Calculate control point
    control_point = mid + control_offset
    
    # Draw a quadratic Bézier curve
    # We'll approximate it by drawing multiple short straight lines
    num_segments = 20
    for i in range(num_segments):
        t = i / float(num_segments)
        t_next = (i + 1) / float(num_segments)
        
        # Quadratic Bézier formula: B(t) = (1-t)²P0 + 2(1-t)tP1 + t²P2
        # where P0=start, P1=control, P2=end
        point1 = ((1-t)**2) * start + 2*(1-t)*t * control_point + (t**2) * end
        point2 = ((1-t_next)**2) * start + 2*(1-t_next)*t_next * control_point + (t_next**2) * end
        
        cv2.line(img, tuple(point1.astype(int)), tuple(point2.astype(int)), color, thickness)


def auto_orient_clip(clip: VideoFileClip, video_path: Path) -> VideoFileClip:
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
    text_rate_ms: float,
    remove_text: bool,
    no_fill: bool,
    ignore_audio: bool,
    video_threshold: float,
    line_distance: int,
    line_stability: bool,
    curved_lines: bool,
    max_curvature: float,
    max_box_size: int | None,
    detection_method: str,
    contour_min_area: int,
    contour_max_area: int,
    hough_lines_threshold: int,
    hough_lines_min_length: int,
    hough_lines_max_gap: int,
    hough_circles_min_radius: int,
    hough_circles_max_radius: int,
    hough_circles_param1: int,
    hough_circles_param2: int,
    bg_sub_learning_rate: float,
    bg_sub_detect_shadows: bool,
    optical_flow_threshold: float,
    color_lower_hsv: list[int],
    color_upper_hsv: list[int],
    edge_density_threshold: float,
    edge_density_grid_size: int,
    temporal_diff_delay: int,
    temporal_diff_threshold: int,
):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    with VideoFileClip(str(video_in)) as clip:
        clip = auto_orient_clip(clip, video_in)
        fps = fps or clip.fps
        frame_w, frame_h = clip.size

        # Onset detection
        if ignore_audio or clip.audio is None or not LIBROSA_AVAILABLE:
            logging.info("⚙️ Using time-based onset detection (interval=%.2f seconds)", video_threshold)
            onset_times = _detect_onsets_time_based(clip, interval=video_threshold)
        else:
            wav_path = _extract_audio(video_in)
            onset_times = _detect_onsets_audio(wav_path)
        logging.info("%d onsets detected", len(onset_times))

        # Initialize detection method and required objects
        if detection_method == "orb":
            detector = cv2.ORB_create(nfeatures=1500, fastThreshold=orb_fast_threshold)
        elif detection_method == "bgs":  # bg-subtraction
            detect_shadows = bg_sub_detect_shadows
            learning_rate = bg_sub_learning_rate if bg_sub_learning_rate > 0 else -1  # -1 for default
            bg_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=detect_shadows, varThreshold=150)
        elif detection_method == "td":  # temporal-difference
            from collections import deque
            frame_history = deque(maxlen=temporal_diff_delay)
        elif detection_method in ["contour", "hl", "hc", "dof", "cr", "ed"]:
            # These detection methods don't need initialization, we'll call the functions directly
            pass
        
        active, onset_idx, prev_gray = [], 0, None
        active_connections = set()  # Track persistent line connections by point ID pairs
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
                if detection_method == "orb":
                    kps = detector.detect(gray, None)
                    kps = sorted(kps, key=lambda k: k.response, reverse=True) if kps else []
                elif detection_method == "contour":
                    kps = _detect_contours(gray, min_area=contour_min_area, max_area=contour_max_area)
                    # Sort by keypoint size
                    kps = sorted(kps, key=lambda k: k.size if hasattr(k, 'size') else 0, reverse=True) if kps else []
                elif detection_method == "hl":  # hough-lines
                    kps = _detect_hough_lines(gray, threshold=hough_lines_threshold, min_line_length=hough_lines_min_length, max_line_gap=hough_lines_max_gap)
                    # Sort by keypoint size (line length)
                    kps = sorted(kps, key=lambda k: k.size if hasattr(k, 'size') else 0, reverse=True) if kps else []
                elif detection_method == "hc":  # hough-circles
                    kps = _detect_hough_circles(gray, min_radius=hough_circles_min_radius, max_radius=hough_circles_max_radius, param1=hough_circles_param1, param2=hough_circles_param2)
                    # Sort by keypoint size (circle diameter)
                    kps = sorted(kps, key=lambda k: k.size if hasattr(k, 'size') else 0, reverse=True) if kps else []
                elif detection_method == "bgs":  # bg-subtraction
                    kps = _detect_background_subtraction(gray, bg_subtractor)
                    # Sort by keypoint size
                    kps = sorted(kps, key=lambda k: k.size if hasattr(k, 'size') else 0, reverse=True) if kps else []
                elif detection_method == "dof":  # dense-optical-flow
                    if prev_gray is not None:  # Only run if we have a previous frame
                        kps = _detect_dense_optical_flow(prev_gray, gray, threshold=optical_flow_threshold)
                        # Sort by keypoint size
                        kps = sorted(kps, key=lambda k: k.size if hasattr(k, 'size') else 0, reverse=True) if kps else []
                    else:
                        kps = []
                elif detection_method == "cr":  # color-range
                    kps = _detect_color_range(frame, lower_color=tuple(color_lower_hsv), upper_color=tuple(color_upper_hsv), min_area=contour_min_area)  # Pass BGR frame to color detection
                    # Sort by keypoint size
                    kps = sorted(kps, key=lambda k: k.size if hasattr(k, 'size') else 0, reverse=True) if kps else []
                elif detection_method == "ed":  # edge-density
                    kps = _detect_edge_density(gray, threshold_factor=edge_density_threshold, grid_size=edge_density_grid_size)
                    # Sort by keypoint size
                    kps = sorted(kps, key=lambda k: k.size if hasattr(k, 'size') else 0, reverse=True) if kps else []
                elif detection_method == "td":  # temporal-difference
                    kps = _detect_temporal_difference(gray, frame_history, frame_delay=temporal_diff_delay, threshold=temporal_diff_threshold)
                    # Sort by keypoint size
                    kps = sorted(kps, key=lambda k: k.size if hasattr(k, 'size') else 0, reverse=True) if kps else []
                
                for kp in kps[:random.randint(1, pts_per_beat)]:
                    # All detection methods now return KeyPoint objects with .pt attribute
                    x, y = kp.pt
                    # Ensure x, y are scalars
                    x, y = float(x), float(y)
                    if all(np.linalg.norm(tp.pos - (x, y)) >= 10 for tp in active):
                        s = _sample_size_bell(min_size, max_size, bell_width)
                        if max_box_size:
                            s = min(s, max_box_size)
                        active.append(TrackedPoint((x, y), life_frames, s, text_rate_ms))
                onset_idx += 1

            # Ambient points
            if ambient_rate > 0:
                for _ in range(np.random.poisson(ambient_rate / fps)):
                    x, y = random.uniform(0, w), random.uniform(0, h)
                    s = _sample_size_bell(min_size, max_size, bell_width)
                    if max_box_size:
                        s = min(s, max_box_size)
                    active.append(TrackedPoint((x, y), life_frames, s, text_rate_ms))

            # Update persistent line connections
            if line_distance > 0 and len(active) > 1:
                active_ids = {tp.id for tp in active}  # Set of currently active point IDs

                # Remove connections where both points are no longer active
                active_connections_copy = active_connections.copy()
                for conn in active_connections_copy:
                    id1, id2 = conn
                    if id1 not in active_ids and id2 not in active_ids:
                        active_connections.discard(conn)

                # Add new connections between close active points
                coords = [tp.pos for tp in active]
                point_id_to_pos = {tp.id: tp.pos for tp in active}

                for i, tp1 in enumerate(active):
                    for j, tp2 in enumerate(active):
                        if i >= j:  # Avoid duplicate connections and self-connections
                            continue

                        dist = np.linalg.norm(tp1.pos - tp2.pos)
                        frame_max_dimension = max(frame_w, frame_h)

                        # Check if points should be connected based on distance
                        should_connect = False
                        if line_distance >= 100:  # "all" behavior
                            should_connect = True
                        else:  # Distance-based behavior
                            distance_threshold = (line_distance / 100.0) * frame_max_dimension
                            should_connect = dist < distance_threshold

                        if should_connect:
                            # Add connection as frozenset to ensure order doesn't matter
                            connection = frozenset([tp1.id, tp2.id])
                            active_connections.add(connection)

            # Draw persistent line connections
            for conn in active_connections:
                id1, id2 = conn
                # Find positions of both points (they should both be active if connection exists)
                pos1 = None
                pos2 = None
                for tp in active:
                    if tp.id == id1:
                        pos1 = tp.pos
                    elif tp.id == id2:
                        pos2 = tp.pos

                if pos1 is not None and pos2 is not None:
                    dist = np.linalg.norm(pos1 - pos2)
                    if curved_lines:
                        # Draw curved lines with random curvature
                        start_point = tuple(pos1.astype(int))
                        end_point = tuple(pos2.astype(int))
                        draw_curved_line(frame, start_point, end_point, dist, (200, 200, 255), 1, max_curvature)
                    else:
                        cv2.line(frame, tuple(pos1.astype(int)), tuple(pos2.astype(int)), (200, 200, 255), 1)

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

                    # Update text based on text_rate_ms
                    current_time_ms = t * 1000  # Convert frame time from seconds to milliseconds
                    if tp.text_rate_ms > 0 and (current_time_ms - tp.last_text_update) >= tp.text_rate_ms:
                        tp.current_text = f"0x{random.randint(0, 0xFFFFFFFF):08X}"
                        tp.last_text_update = current_time_ms
                    
                    hex_text = tp.current_text
                    text_pos = (br[0] + 5, min(br[1], h - 10))
                    cv2.putText(frame, hex_text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, text_size, txt_color, 1, cv2.LINE_AA)

            prev_gray = gray
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        out_clip = VideoClip(make_frame, duration=clip.duration)
        out_clip = out_clip.with_audio(clip.audio if not ignore_audio else None).with_fps(fps)
        out_clip.write_videofile(str(video_out), codec="libx264", audio_codec="aac" if not ignore_audio else None)

    logging.info("✅ Render complete: %s", video_out)


def analyze_all_detection_methods(video_path: Path, sample_frames: int = 20, force_max: bool = False, pts_per_beat: int = 20, life_frames: int = 10, ambient_rate: float = 5.0, ignore_audio: bool = False, video_threshold: float = 1.0):
    """Analyze all detection methods with maximum sensitivity settings and simulate actual box generation."""
    mode_desc = "Maximum Sensitivity (Forced)" if force_max else "Maximum Sensitivity Settings"
    print(f"🔍 Detection Method Analysis ({mode_desc})")
    print(f"Video: {video_path.name}")

    with VideoFileClip(str(video_path)) as clip:
        clip = auto_orient_clip(clip, video_path)
        total_frames = int(clip.fps * clip.duration)
        sample_frames = min(max(1, sample_frames), min(100, total_frames))  # Clamp to reasonable range

        print(f"Sampling {sample_frames} frames from {total_frames} total frames")
        print()

        # Onset detection for simulation
        if ignore_audio or clip.audio is None or not LIBROSA_AVAILABLE:
            print("⚙️ Using time-based onset detection (interval=%.2f seconds)" % video_threshold)
            onset_times = _detect_onsets_time_based(clip, interval=video_threshold)
        else:
            wav_path = _extract_audio(video_path)
            onset_times = _detect_onsets_audio(wav_path)
        print(f"{len(onset_times)} onsets detected")

        # Define maximum sensitivity settings for each method
        max_settings = {
            "orb": {"nfeatures": 1500, "fastThreshold": 1},
            "contour": {"min_area": 1, "max_area": 10000},
            "hl": {"threshold": 10, "min_line_length": 1, "max_line_gap": 1},
            "hc": {"min_radius": 1, "max_radius": 100, "param1": 10, "param2": 5},
            "bgs": {"learning_rate": 0.001, "detect_shadows": True},
            "dof": {"threshold": 1.0},
            "cr": {"lower_color": (0, 0, 0), "upper_color": (179, 255, 255), "min_area": 1},
            "ed": {"threshold_factor": 0.01, "grid_size": 8},
            "td": {"frame_delay": 1, "threshold": 5}
        }

        # Initialize background subtractor for BGS
        bg_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=True, varThreshold=150)

        # Initialize temporal difference frame history
        from collections import deque
        frame_history = deque(maxlen=1)

        method_names_display = {
            "orb": "ORB Features",
            "contour": "Contour Detection",
            "hl": "Hough Lines",
            "hc": "Hough Circles",
            "bgs": "Background Subtraction",
            "dof": "Dense Optical Flow",
            "cr": "Color Range",
            "ed": "Edge Density",
            "td": "Temporal Difference"
        }

        results = {}

        # For each detection method, simulate the full process
        for method_name in ["orb", "contour", "hl", "hc", "bgs", "dof", "cr", "ed", "td"]:
            print(f"⏳ Analyzing {method_names_display[method_name]}...")
            total_keypoints = 0
            total_boxes_spawned = 0
            total_active_boxes = 0
            
            # Simulate the entire process for this method
            active_points = []
            onset_idx = 0
            prev_gray = None

            # Process each frame to detect keypoints and simulate box spawning
            for frame_time in np.linspace(0, clip.duration, total_frames if force_max else sample_frames, endpoint=False):
                frame = clip.get_frame(frame_time)
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

                settings = max_settings[method_name]

                # Get keypoints for this method
                if method_name == "orb":
                    detector = cv2.ORB_create(nfeatures=settings["nfeatures"], fastThreshold=settings["fastThreshold"])
                    keypoints = detector.detect(gray, None)
                elif method_name == "contour":
                    keypoints = _detect_contours(gray, min_area=settings["min_area"], max_area=settings["max_area"])
                elif method_name == "hl":
                    keypoints = _detect_hough_lines(gray, threshold=settings["threshold"],
                                                   min_line_length=settings["min_line_length"],
                                                   max_line_gap=settings["max_line_gap"])
                elif method_name == "hc":
                    keypoints = _detect_hough_circles(gray, min_radius=settings["min_radius"],
                                                     max_radius=settings["max_radius"],
                                                     param1=settings["param1"], param2=settings["param2"])
                elif method_name == "bgs":
                    keypoints = _detect_background_subtraction(gray, bg_subtractor)
                elif method_name == "dof":
                    if prev_gray is not None:
                        keypoints = _detect_dense_optical_flow(prev_gray, gray, threshold=settings["threshold"])
                    else:
                        keypoints = []
                elif method_name == "cr":
                    keypoints = _detect_color_range(frame_bgr, lower_color=settings["lower_color"],
                                                   upper_color=settings["upper_color"], min_area=settings["min_area"])
                elif method_name == "ed":
                    keypoints = _detect_edge_density(gray, threshold_factor=settings["threshold_factor"],
                                                   grid_size=settings["grid_size"])
                elif method_name == "td":
                    keypoints = _detect_temporal_difference(gray, frame_history,
                                                          frame_delay=settings["frame_delay"],
                                                          threshold=settings["threshold"])

                # Count total keypoints found
                total_keypoints += len(keypoints) if keypoints else 0

                # Simulate spawning boxes on onsets
                while onset_idx < len(onset_times) and frame_time >= onset_times[onset_idx]:
                    if keypoints:
                        # Simulate the same logic as in the main rendering function
                        # Sort keypoints by response/size if available
                        if method_name in ["contour", "hl", "hc", "bgs", "dof", "cr", "ed", "td"]:
                            # Sort by keypoint size/response if available
                            keypoints = sorted(keypoints, key=lambda k: k.size if hasattr(k, 'size') else 0, reverse=True) if keypoints else []
                        # Spawn points from detected keypoints, similar to the main function
                        num_points_to_spawn = random.randint(1, pts_per_beat)
                        for kp in keypoints[:num_points_to_spawn]:
                            x, y = kp.pt
                            x, y = float(x), float(y)
                            # Only add if not too close to an existing point
                            if all(np.linalg.norm(tp.pos - (x, y)) >= 10 for tp in active_points):
                                # Create a simulated TrackedPoint
                                active_points.append(TrackedPoint((x, y), life_frames, _sample_size_bell(15, 40, 4.0)))
                                total_boxes_spawned += 1
                    
                    onset_idx += 1

                # Simulate ambient points
                if ambient_rate > 0:
                    for _ in range(np.random.poisson(ambient_rate / clip.fps)):
                        x, y = random.uniform(0, gray.shape[1]), random.uniform(0, gray.shape[0])  # Use frame dimensions
                        s = _sample_size_bell(15, 40, 4.0)
                        active_points.append(TrackedPoint((x, y), life_frames, s))
                        total_boxes_spawned += 1

                # Update active points: reduce life and remove dead ones
                active_points = [tp for tp in active_points if tp.life > 0]
                for tp in active_points:
                    tp.life -= 1
                
                total_active_boxes += len(active_points)
                prev_gray = gray

            # Store results for this method
            results[method_name] = {
                'keypoints': total_keypoints,
                'boxes_spawned': total_boxes_spawned,
                'avg_active_boxes': total_active_boxes / (total_frames if force_max else sample_frames) if (total_frames if force_max else sample_frames) > 0 else 0
            }
            
            print(f"   ✓ Found {total_keypoints} keypoints, spawned {total_boxes_spawned} boxes")

        # Display results in a formatted table
        print("\nDetection Method Analysis Results:")
        print("-" * 80)
        print(f"{'Method':<20} | {'Keypoints':>9} | {'Boxes Spawned':>12} | {'Avg Active':>10}")
        print("-" * 80)

        for method, counts in results.items():
            print(f"{method_names_display[method]:<20} | {counts['keypoints']:>9,} | {counts['boxes_spawned']:>12,} | {counts['avg_active_boxes']:>10.1f}")

        print("-" * 80)
        print("💡 Keypoints: Raw features detected by each method")
        print("💡 Boxes Spawned: Potential boxes that would be created during actual rendering")
        print("💡 Avg Active: Average number of boxes visible simultaneously")
        print("💡 Higher box generation counts indicate methods that work well for this video content.")


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
    parser.add_argument("-tr", "--text-rate", type=float, default=0.0, help="Text change rate in milliseconds (0 = change every frame)")
    parser.add_argument("-r", "--remove-text", action="store_true", help="Remove text overlay entirely")
    parser.add_argument("-n", "--no-fill", action="store_true", help="Disable box inversion fill")
    parser.add_argument("-a", "--ignore-audio", action="store_true", help="Ignore audio and use video intensity threshold")
    parser.add_argument("-th", "--threshold", type=float, default=1.0, help="Time interval between onsets (seconds) when --ignore-audio is used")
    parser.add_argument("-ld", "--life-duration", type=float, default=333.0, help="Duration in milliseconds that tracked points remain active (333ms = ~10 frames at 30fps)")
    parser.add_argument("-l", "--line-distance", type=int, default=80,
                        help="Distance threshold for connecting points (0 for none, higher values for larger distances)")
    parser.add_argument("--line-stability", action="store_true", help="Enable stable line connections that persist even when points move (reduces blinking)")
    parser.add_argument("--curved-lines", action="store_true", help="Enable curved lines with distance-based curvature")
    parser.add_argument("--max-curvature", type=float, default=0.5, help="Maximum curvature for curved lines (0.0 to 1.0)")
    parser.add_argument("-m", "--max-box-size", type=int, default=None, help="Set maximum box size (override)")
    parser.add_argument("--detection-method", type=str, default="orb", choices=["orb", "contour", "hl", "hc", "bgs", "dof", "cr", "ed", "td"],
                        help="Detection method: 'orb' (default) for ORB feature detection, 'contour' for contour detection, 'hl' for Hough line detection, 'hc' for Hough circle detection, 'bgs' for background subtraction, 'dof' for dense optical flow, 'cr' for color-based detection, 'ed' for edge density detection, 'td' for temporal difference motion detection")
    
    # Detection method specific parameters
    parser.add_argument("--ct-min-area", type=int, default=50, help="Minimum contour area for contour detection")
    parser.add_argument("--ct-max-area", type=int, default=5000, help="Maximum contour area for contour detection")
    parser.add_argument("--hl-threshold", type=int, default=100, help="Threshold parameter for Hough lines detection")
    parser.add_argument("--hl-min-length", type=int, default=30, help="Minimum line length for Hough lines detection")
    parser.add_argument("--hl-max-gap", type=int, default=10, help="Maximum gap between line segments for Hough lines detection")
    parser.add_argument("--hc-min-radius", type=int, default=10, help="Minimum circle radius for Hough circles detection")
    parser.add_argument("--hc-max-radius", type=int, default=50, help="Maximum circle radius for Hough circles detection")
    parser.add_argument("--hc-param1", type=int, default=50, help="Parameter 1 for Hough circles detection")
    parser.add_argument("--hc-param2", type=int, default=30, help="Parameter 2 for Hough circles detection")
    parser.add_argument("--bgs-learning-rate", type=float, default=0.0, help="Learning rate for background subtraction (0.0 = auto)")
    parser.add_argument("--bgs-detect-shadows", action="store_true", help="Enable shadow detection for background subtraction")
    parser.add_argument("--dof-threshold", type=float, default=15.0, help="Motion threshold for dense optical flow detection")
    parser.add_argument("--cr-lower-hsv", type=int, nargs=3, default=[0, 50, 50], help="Lower HSV bounds for color detection [H, S, V]")
    parser.add_argument("--cr-upper-hsv", type=int, nargs=3, default=[10, 255, 255], help="Upper HSV bounds for color detection [H, S, V]")
    parser.add_argument("--ed-threshold", type=float, default=0.1, help="Edge density threshold factor")
    parser.add_argument("--ed-grid-size", type=int, default=32, help="Grid cell size for edge density analysis")
    parser.add_argument("--td-delay", type=int, default=1, help="Frame delay for temporal difference motion detection (1 = fast movements, higher = slow movements)")
    parser.add_argument("--td-threshold", type=int, default=30, help="Threshold for temporal difference motion detection")
    parser.add_argument("-va", "--verbose-all", action="store_true", help="Analyze all detection methods with maximum sensitivity (no video output)")
    parser.add_argument("--sample-frames", type=int, default=20, help="Number of frames to sample for --verbose-all analysis (1-100)")
    parser.add_argument("--max", action="store_true", help="Force maximum sensitivity for all detection methods (overrides other settings)")

    args = parser.parse_args()

    if len(args.text_color) == 1 and args.text_color[0].lower() == "negative":
        text_color = "negative"
    else:
        text_color = tuple(map(int, args.text_color))

    if not os.path.isfile(args.input):
        print("❌ Error: Input file does not exist.")
        return

    # Log info about librosa availability
    if not LIBROSA_AVAILABLE:
        logging.warning("⚠️ Librosa not available. Audio-based onset detection will be disabled.")

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Convert life duration from milliseconds to frames
    with VideoFileClip(str(args.input)) as temp_clip:
        temp_clip = auto_orient_clip(temp_clip, Path(args.input))
        fps = temp_clip.fps
        life_frames = max(1, int((args.life_duration / 1000.0) * fps))  # Convert ms to frames, minimum 1 frame

    # Handle verbose-all analysis mode
    if args.verbose_all:
        analyze_all_detection_methods(
            video_path=Path(args.input),
            sample_frames=args.sample_frames,
            force_max=args.max,
            pts_per_beat=20,
            life_frames=life_frames,
            ambient_rate=5.0,
            ignore_audio=args.ignore_audio,
            video_threshold=args.threshold
        )
        return

    render_tracked_effect(
        video_in=Path(args.input),
        video_out=Path(args.output),
        fps=None,
        pts_per_beat=20,
        ambient_rate=5.0,
        jitter_px=0.5,
        life_frames=life_frames,
        min_size=15,
        max_size=40,
        neighbor_links=3,
        orb_fast_threshold=20,
        bell_width=4.0,
        seed=None,
        text_size=args.text_size,
        text_color=text_color,
        text_rate_ms=args.text_rate,
        remove_text=args.remove_text,
        no_fill=args.no_fill,
        ignore_audio=args.ignore_audio,
        video_threshold=args.threshold,
        line_distance=args.line_distance,
        line_stability=args.line_stability,
        curved_lines=args.curved_lines,
        max_curvature=args.max_curvature,
        max_box_size=args.max_box_size,
        detection_method=args.detection_method,
        contour_min_area=args.ct_min_area,
        contour_max_area=args.ct_max_area,
        hough_lines_threshold=args.hl_threshold,
        hough_lines_min_length=args.hl_min_length,
        hough_lines_max_gap=args.hl_max_gap,
        hough_circles_min_radius=args.hc_min_radius,
        hough_circles_max_radius=args.hc_max_radius,
        hough_circles_param1=args.hc_param1,
        hough_circles_param2=args.hc_param2,
        bg_sub_learning_rate=args.bgs_learning_rate,
        bg_sub_detect_shadows=args.bgs_detect_shadows,
        optical_flow_threshold=args.dof_threshold,
        color_lower_hsv=args.cr_lower_hsv,
        color_upper_hsv=args.cr_upper_hsv,
        edge_density_threshold=args.ed_threshold,
        edge_density_grid_size=args.ed_grid_size,
        temporal_diff_delay=args.td_delay,
        temporal_diff_threshold=args.td_threshold,
    )


if __name__ == "__main__":
    main()
