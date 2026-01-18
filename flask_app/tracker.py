from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple

import cv2
import numpy as np


@dataclass
class Annotation:
    id: int
    x: float
    y: float
    lost: bool = False
    pts: Optional[np.ndarray] = None  # Nx1x2 feature points


class MotionTracker:
    """
    Reads an MP4, tracks features near annotation points using LK optical flow,
    updates annotation positions each frame, and draws dots on frames.
    """

    def __init__(self, video_path: str):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)

        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        self.frame_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        self.frame_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 30.0)

        self.prev_gray = None
        self.annotations: List[Annotation] = []
        self.next_id = 1

        # Tunables
        self.roi_size = 90
        self.min_features = 10
        self.max_features = 80

        self.gftt_params = dict(
            maxCorners=self.max_features,
            qualityLevel=0.01,
            minDistance=6,
            blockSize=7
        )

        self.lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )

        # Prime first frame
        ok, frame = self.cap.read()
        if not ok or frame is None:
            raise RuntimeError("Video opened but could not read first frame.")
        self.prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # If metadata missing, infer from frame
        h, w = self.prev_gray.shape[:2]
        self.frame_w = self.frame_w or w
        self.frame_h = self.frame_h or h

        # rewind to start for streaming
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def reset(self):
        self.annotations = []
        self.next_id = 1
        self.cap.release()
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video: {self.video_path}")

        ok, frame = self.cap.read()
        if not ok or frame is None:
            raise RuntimeError("Could not read first frame after reset.")
        self.prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def loop_video(self):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def get_info(self) -> Dict[str, Any]:
        return {
            "video": self.video_path,
            "frame_w": self.frame_w,
            "frame_h": self.frame_h,
            "fps": self.fps,
            "annotations": len(self.annotations),
        }

    def add_annotation_norm(self, x_norm: float, y_norm: float) -> Dict[str, Any]:
        x = float(x_norm * self.frame_w)
        y = float(y_norm * self.frame_h)

        ann = Annotation(id=self.next_id, x=x, y=y)
        self.next_id += 1

        # init features near the point
        ann.pts = self._init_features_near(self.prev_gray, x, y)
        if ann.pts is None or len(ann.pts) < self.min_features:
            ann.lost = True
            ann.pts = None

        self.annotations.append(ann)
        return {"id": ann.id, "x": ann.x, "y": ann.y, "lost": ann.lost}

    def step_encoded_jpeg(self) -> Optional[bytes]:
        ok, frame = self.cap.read()
        if not ok or frame is None:
            return None

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # update annotation positions with LK flow
        if self.prev_gray is not None:
            self._update_annotations(self.prev_gray, gray)

        # draw overlay onto frame
        self._draw(frame)

        self.prev_gray = gray

        ok2, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if not ok2:
            return None
        return buf.tobytes()

    def _init_features_near(self, gray: np.ndarray, x: float, y: float) -> Optional[np.ndarray]:
        h, w = gray.shape[:2]
        half = self.roi_size // 2
        x0 = max(0, int(x - half))
        y0 = max(0, int(y - half))
        x1 = min(w, int(x + half))
        y1 = min(h, int(y + half))

        if x1 <= x0 + 5 or y1 <= y0 + 5:
            return None

        roi = gray[y0:y1, x0:x1]
        pts = cv2.goodFeaturesToTrack(roi, mask=None, **self.gftt_params)
        if pts is None:
            return None

        pts[:, 0, 0] += x0
        pts[:, 0, 1] += y0
        return pts.astype(np.float32)

    def _update_annotations(self, prev_gray: np.ndarray, gray: np.ndarray):
        for ann in self.annotations:
            if ann.lost or ann.pts is None or len(ann.pts) == 0:
                continue

            new_pts, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, ann.pts, None, **self.lk_params)
            if new_pts is None or st is None:
                ann.lost = True
                ann.pts = None
                continue

            good_old = ann.pts[st.flatten() == 1]
            good_new = new_pts[st.flatten() == 1]

            if len(good_new) < self.min_features:
                # try re-init
                fresh = self._init_features_near(gray, ann.x, ann.y)
                if fresh is None or len(fresh) < self.min_features:
                    ann.lost = True
                    ann.pts = None
                else:
                    ann.pts = fresh
                    ann.lost = False
                continue

            disp = (good_new - good_old).reshape(-1, 2)
            dx = float(np.median(disp[:, 0]))
            dy = float(np.median(disp[:, 1]))

            ann.x = float(np.clip(ann.x + dx, 0, self.frame_w - 1))
            ann.y = float(np.clip(ann.y + dy, 0, self.frame_h - 1))
            ann.pts = good_new.reshape(-1, 1, 2).astype(np.float32)

    def _draw(self, frame: np.ndarray):
        for ann in self.annotations:
            x, y = int(round(ann.x)), int(round(ann.y))
            if ann.lost:
                cv2.putText(frame, f"#{ann.id} LOST", (x + 10, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.line(frame, (x - 10, y - 10), (x + 10, y + 10), (0, 0, 255), 2)
                cv2.line(frame, (x - 10, y + 10), (x + 10, y - 10), (0, 0, 255), 2)
            else:
                cv2.circle(frame, (x, y), 8, (0, 255, 255), -1)
                cv2.circle(frame, (x, y), 10, (0, 0, 0), 2)
                cv2.putText(frame, f"#{ann.id}", (x + 10, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

