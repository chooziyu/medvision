import os
import threading
import time
from flask import Flask, Response, jsonify, render_template, request

from tracker import MotionTracker

app = Flask(__name__)

# -------- Config --------
DEFAULT_VIDEO = os.path.join(app.root_path, "static", "sample.mp4")
VIDEO_PATH = os.environ.get("VIDEO_PATH", DEFAULT_VIDEO)

# We keep one tracker instance for the whole server
tracker_lock = threading.Lock()
tracker = MotionTracker(VIDEO_PATH)


@app.get("/")
def index():
    # Your UI file must be: flask_app/templates/index.html
    return render_template("index.html")


@app.get("/api/ping")
def ping():
    with tracker_lock:
        info = tracker.get_info()
    return jsonify({"ok": True, **info})


@app.post("/api/add_annotation")
def add_annotation():
    """
    Body JSON:
      { "x": 0.42, "y": 0.58 }   # normalized coords 0..1
    """
    data = request.get_json(force=True, silent=True) or {}
    if "x" not in data or "y" not in data:
        return jsonify({"ok": False, "error": "Missing x or y"}), 400

    try:
        x = float(data["x"])
        y = float(data["y"])
    except Exception:
        return jsonify({"ok": False, "error": "x and y must be numbers"}), 400

    if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
        return jsonify({"ok": False, "error": "x and y must be normalized 0..1"}), 400

    with tracker_lock:
        ann = tracker.add_annotation_norm(x, y)

    return jsonify({"ok": True, "annotation": ann})


@app.post("/api/reset")
def reset():
    with tracker_lock:
        tracker.reset()
        info = tracker.get_info()
    return jsonify({"ok": True, **info})


def mjpeg_generator():
    """
    Streams frames as multipart/x-mixed-replace (MJPEG).
    Browser reads it as a continuously updating <img>.
    """
    while True:
        with tracker_lock:
            frame_bytes = tracker.step_encoded_jpeg()

        if frame_bytes is None:
            # End of video or read failure: try a small pause, then loop video
            time.sleep(0.1)
            with tracker_lock:
                tracker.loop_video()
            continue

        # MJPEG frame format
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")


@app.get("/stream")
def stream():
    return Response(
        mjpeg_generator(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )

@app.post("/api/test")
def api_test():
    return jsonify({"ok": True, "message": "Frontend reached backend!"})

if __name__ == "__main__":
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    app.run(host=host, port=port, debug=True, threaded=True)
