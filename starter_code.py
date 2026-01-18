
import cv2
import numpy as np

# ---------- Parameters ----------
FEATURE_PARAMS = dict(
    maxCorners=50,
    qualityLevel=0.3,
    minDistance=7,
    blockSize=7
)

LK_PARAMS = dict(
    winSize=(15, 15),
    maxLevel=2,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
)

# ---------- Load video ----------
cap = cv2.VideoCapture("Lapchole1.mp4")  # replace with your video

ret, first_frame = cap.read()
if not ret:
    raise RuntimeError("Could not read video")

# ---------- INITIAL ANNOTATION (hardcoded for MVP) ----------
# x, y, width, height
bbox = [200, 200, 150, 150]

old_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

# Mask to detect features only inside the annotation box
mask = np.zeros_like(old_gray)
x, y, w, h = bbox
mask[y:y+h, x:x+w] = 255

# Detect feature points inside box
p0 = cv2.goodFeaturesToTrack(old_gray, mask=mask, **FEATURE_PARAMS)

# ---------- Main loop ----------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Track feature points
    p1, st, err = cv2.calcOpticalFlowPyrLK(
        old_gray, frame_gray, p0, None, **LK_PARAMS
    )

    # Keep only valid points
    good_new = p1[st == 1]
    good_old = p0[st == 1]

    if len(good_new) > 0:
        # Compute median motion
        dx = np.median(good_new[:, 0] - good_old[:, 0])
        dy = np.median(good_new[:, 1] - good_old[:, 1])

        # Move annotation box
        bbox[0] += int(dx)
        bbox[1] += int(dy)

        # Update points relative to new bbox
        p0 = good_new.reshape(-1, 1, 2)

    # Draw annotation box
    x, y, w, h = map(int, bbox)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Draw feature points
    for pt in p0:
        cx, cy = pt.ravel()
        cv2.circle(frame, (int(cx), int(cy)), 3, (0, 0, 255), -1)

    cv2.imshow("Tracked Annotation", frame)

    if cv2.waitKey(30) & 0xFF == 27:  # ESC to quit
        break

    old_gray = frame_gray.copy()

cap.release()
cv2.destroyAllWindows()