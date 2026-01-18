import time
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any, Tuple

import cv2
import numpy as np


@dataclass
class Annotation:
    id: int
    x: float                 # pixel x in current frame
    y: float                 # pixel y in current frame
    pinned: bool = True      # future feature; currently always pinned
    lost: bool = False       # if tracking fails
    color_bgr: Tuple[int, int, int] = (0, 255, 255)  # yellow in BGR
    pts: Optional[np.ndarray] = None                 # tracked feature points (Nx1x2 float32)


class MotionTracker:
    """
    A simple motion-tracked annotation engine.

    - Reads MP4 frames sequentially.
    - When an annotation is added, it finds feature points near that location.
    - Each frame, it tracks those feature points using LK Optical Flow (calcOpticalFlowPyrLK).
    - The annotation position is updated by the median displacement of its tracked features.
    - Draws the annotation on the frame.
    - Encodes frame to JPEG for MJPEG streaming.
    """

    def __init__(self, video_path: str):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)

        self.frame_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        self.frame_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 30.0)

        self.prev_gray = None
        self.prev_frame = None

        self.annotations: List[Annotation] = []
        self.next_id = 1

        # Tunables
        self.roi_size = 80  # pixels around click to search features
        self.min_features = 8
        self.max_features = 80

        # LK params
        self.lk_win = (21, 21)
        self.lk_max_level = 3
        self.lk_criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)

        # Feature detector params
        self.gftt_params = dict(
            maxCorners=self.max_features,
            qualityLevel=0.01,
            minDistance=5,
            blockSize=7
        )

        # Prime first frame
        self._read_first_frame()

    # ---------------- Public API ----------------

    def get_info(self) -> Dict[str, Any]:
        return {
            "video": self.video_path,
            "frame_w": self.frame_w,
            "frame_h": self.frame_h,
            "fps": self.fps,
            "annotations": len(self.annotations),
        }

    def reset(self) -> None:
        """Reset everything: annotations + video position."""
        self.annotations = []
        self.next_id = 1
        self._reopen_video()
        self._read_first_frame()

    def loop_video(self) -> None:
        """If the video ended, restart it (keep annotations)."""
        self._reopen_video()
        self._read_first_frame()

    def add_annotation_norm(self, x_norm: float, y_norm: float) -> Dict[str, Any]:
        """
        Add annotation using normalized coordinates (0..1).
        """
        # Ensure we have a frame to reference
        if self.prev_frame is None:
            self._read_first_frame()

        x_px = float(x_norm * max(1, self.frame_w))
        y_px = float(y_norm * max(1, self.frame_h))

        ann = Annotation(id=self.next_id, x=x_px, y=y_px)
        self.next_id += 1

        # Try to initialize features near that point
        ann.pts = self._init_features_near(x_px, y_px, self.prev_gray)

        if ann.pts is None or len(ann.pts) < self.min_features:
            ann.lost = True

        self.annotations.append(ann)
        return self._ann_to_dict(ann)

    def step_encoded_jpeg(self) -> Optional[bytes]:
        """
        Read next frame, update annotation positions, draw overlay, return JPEG bytes.
        Returns None if frame cannot be read (end of video).
        """
        ok, frame = self.cap.read()
        if not ok or frame is None:
            return None

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Update tracking if we have a previous gray frame
        if self.prev_gray is not None:
            self._update_annotations(self.prev_gray, gray)

        # Draw overlay annotations onto current frame
        out = frame.copy()
        self._draw_annotations(out)

        # Update prev frame
        self.prev_gray = gray
        self.prev_frame = frame

        # Encode to JPEG
        ok2, buf = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if not ok2:
            return None
        return buf.tobytes()

    # ---------------- Internals ----------------

    def _reopen_video(self) -> None:
        try:
            self.cap.release()
        except Exception:
            pass
        self.cap = cv2.VideoCapture(self.video_path)

        self.frame_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        self.frame_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 30.0)

        self.prev_gray = None
        self.prev_frame = None

    def _read_first_frame(self) -> None:
        ok, frame = self.cap.read()
        if not ok or frame is None:
            self.prev_frame = None
            self.prev_gray = None
            return
        self.prev_frame = frame
        self.prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # If frame_w/h not known, infer from frame
        h, w = self.prev_gray.shape[:2]
        self.frame_w = self.frame_w or w
        self.frame_h = self.frame_h or h

    def _init_features_near(self, x: float, y: float, gray: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """
        Find features near (x,y) within a square ROI, return points as Nx1x2 float32.
        """
        if gray is None:
            return None

        h, w = gray.shape[:2]
        half = self.roi_size // 2

        x0 = int(max(0, x - half))
        y0 = int(max(0, y - half))
        x1 = int(min(w, x + half))
        y1 = int(min(h, y + half))

        if x1 <= x0 + 5 or y1 <= y0 + 5:
            return None

        roi = gray[y0:y1, x0:x1]

        pts = cv2.goodFeaturesToTrack(roi, mask=None, **self.gftt_params)
        if pts is None:
            return None

        # Convert ROI coords to full-frame coords
        pts[:, 0, 0] += x0
        pts[:, 0, 1] += y0

        return pts.astype(np.float32)

    def _update_annotations(self, prev_gray: np.ndarray, gray: np.ndarray) -> None:
        """
        For each annotation, track its points forward and update its (x,y).
        """
        for ann in self.annotations:
            if ann.lost or ann.pts is None or len(ann.pts) == 0:
                continue

            # Track points using LK optical flow
            new_pts, status, _err = cv2.calcOpticalFlowPyrLK(
                prev_gray,
                gray,
                ann.pts,
                None,
                winSize=self.lk_win,
                maxLevel=self.lk_max_level,
                criteria=self.lk_criteria
            )

            if new_pts is None or status is None:
                ann.lost = True
                ann.pts = None
                continue

            good_old = ann.pts[status.flatten() == 1]
            good_new = new_pts[status.flatten() == 1]

            if len(good_new) < self.min_features:
                # Try re-detect features near current location
                fresh = self._init_features_near(ann.x, ann.y, gray)
                if fresh is None or len(fresh) < self.min_features:
                    ann.lost = True
                    ann.pts = None
                else:
                    ann.pts = fresh
                continue

            # Compute displacement of tracked points (new - old)
            disp = (good_new - good_old).reshape(-1, 2)
            dx = float(np.median(disp[:, 0]))
            dy = float(np.median(disp[:, 1]))

            # Update annotation position
            ann.x += dx
            ann.y += dy

            # Clamp within frame
            ann.x = float(np.clip(ann.x, 0, self.frame_w - 1))
            ann.y = float(np.clip(ann.y, 0, self.frame_h - 1))

            # Keep updated points for next step
            ann.pts = good_new.reshape(-1, 1, 2).astype(np.float32)

            # Optional: refresh points sometimes (helps reduce drift)
            # Here we do a light refresh if points drop too low
            if len(ann.pts) < self.min_features * 2:
                fresh = self._init_features_near(ann.x, ann.y, gray)
                if fresh is not None and len(fresh) >= self.min_features:
                    ann.pts = fresh

    def _draw_annotations(self, frame: np.ndarray) -> None:
        """
        Draw circle + label. If lost, draw red X.
        """
        for ann in self.annotations:
            x = int(round(ann.x))
            y = int(round(ann.y))

            if ann.lost:
                # draw an X
                cv2.line(frame, (x - 10, y - 10), (x + 10, y + 10), (0, 0, 255), 2)
                cv2.line(frame, (x - 10, y + 10), (x + 10, y - 10), (0, 0, 255), 2)
                cv2.putText(frame, f"#{ann.id} LOST", (x + 12, y - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                continue

            cv2.circle(frame, (x, y), 8, ann.color_bgr, -1)
            cv2.circle(frame, (x, y), 10, (0, 0, 0), 2)
            cv2.putText(frame, f"#{ann.id}", (x + 12, y - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, ann.color_bgr, 2)

            # Debug: show features
            # (Comment out if you don't want to see dots)
            if ann.pts is not None:
                for p in ann.pts.reshape(-1, 2):
                    px, py = int(p[0]), int(p[1])
                    cv2.circle(frame, (px, py), 2, (255, 255, 255), -1)

    def _ann_to_dict(self, ann: Annotation) -> Dict[str, Any]:
        d = asdict(ann)
        # pts is numpy; remove it from JSON
        d.pop("pts", None)
        # tuple not JSON-friendly in some cases
        d["color_bgr"] = list(ann.color_bgr)
        return d

