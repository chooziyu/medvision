import os
import subprocess
import threading
import time
from collections import deque
from flask import Flask, Response, jsonify, render_template, request

from tracker import MotionTracker

app = Flask(__name__)

VIDEO_PATH = os.environ.get("VIDEO_PATH", os.path.join(app.root_path, "static", "sample.mp4"))

tracker_lock = threading.Lock()
tracker = MotionTracker(VIDEO_PATH)
notebook_lock = threading.Lock()
notebook_proc = None
notebook_last_error = None
notebook_log_lock = threading.Lock()
notebook_log_lines = deque(maxlen=500)

NOTEBOOK_PATH = os.environ.get(
    "NOTEBOOK_PATH",
    "/Users/carolinechueh/Desktop/holoray-ui/Microscopy.ipynb",
)


def _launch_notebook():
    global notebook_proc
    global notebook_last_error
    if notebook_proc and notebook_proc.poll() is None:
        return

    cmd = ["python3", "-m", "notebook", NOTEBOOK_PATH]
    try:
        notebook_proc = subprocess.Popen(
            cmd,
            cwd=os.path.dirname(NOTEBOOK_PATH),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        notebook_last_error = None
        threading.Thread(target=_read_notebook_output, args=(notebook_proc,), daemon=True).start()
    except FileNotFoundError:
        notebook_last_error = "python3 not found"
    except Exception:
        notebook_last_error = "launch failed"


def _append_notebook_log(line: str) -> None:
    if not line:
        return
    with notebook_log_lock:
        notebook_log_lines.append(line)


def _read_notebook_output(proc: subprocess.Popen) -> None:
    if proc.stdout is None:
        return
    for line in iter(proc.stdout.readline, ""):
        _append_notebook_log(line.rstrip())
    _append_notebook_log("Notebook process exited.")


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/ping")
def ping():
    with tracker_lock:
        info = tracker.get_info()
    return jsonify({"ok": True, **info})


@app.post("/api/add_annotation")
def add_annotation():
    data = request.get_json(force=True, silent=True) or {}
    if "x" not in data or "y" not in data:
        return jsonify({"ok": False, "error": "Need JSON: {x,y} normalized 0..1"}), 400

    x = float(data["x"])
    y = float(data["y"])
    if not (0 <= x <= 1 and 0 <= y <= 1):
        return jsonify({"ok": False, "error": "x,y must be 0..1"}), 400

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
    while True:
        with tracker_lock:
            jpg = tracker.step_encoded_jpeg()

        if jpg is None:
            time.sleep(0.05)
            with tracker_lock:
                tracker.loop_video()
            continue

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")


@app.get("/stream")
def stream():
    return Response(mjpeg_generator(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.post("/api/test")
def api_test():
    return jsonify({"ok": True, "message": "Frontend reached backend!"})

@app.post("/api/launch_notebook")
def launch_notebook():
    with notebook_lock:
        _launch_notebook()
        running = bool(notebook_proc and notebook_proc.poll() is None)
        return jsonify({"ok": running, "running": running, "error": notebook_last_error})


@app.get("/api/notebook_status")
def notebook_status():
    with notebook_lock:
        running = bool(notebook_proc and notebook_proc.poll() is None)
        return jsonify({"ok": True, "running": running, "error": notebook_last_error})


@app.get("/api/notebook_logs")
def notebook_logs():
    tail = request.args.get("tail", default=200, type=int)
    with notebook_log_lock:
        lines = list(notebook_log_lines)[-tail:]
    return jsonify({"ok": True, "lines": lines})


if __name__ == "__main__":
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    app.run(host=host, port=port, debug=True, threaded=True)
