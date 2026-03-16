import os
import re
import cv2  # optional if installed; handled below
import json
import math
import time
import socket
import shutil
import random
import psutil
import atexit
import threading
import platform
import statistics
import subprocess
from collections import deque
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, Body
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn

# ------------------------------------------------------------
# Optional dependencies
# ------------------------------------------------------------
ENABLE_WEBCAM = os.getenv("ENABLE_WEBCAM", "0") == "1"
ENABLE_MIC = os.getenv("ENABLE_MIC", "0") == "1"

try:
    import cv2  # type: ignore
except Exception:
    cv2 = None
    ENABLE_WEBCAM = False

try:
    import numpy as np  # type: ignore
    import sounddevice as sd  # type: ignore
except Exception:
    np = None
    sd = None
    ENABLE_MIC = False

# ------------------------------------------------------------
# Config
# ------------------------------------------------------------
HOST = "127.0.0.1"
PORT = 8765
ROOM_NAME = os.getenv("ROOM_NAME", socket.gethostname())
ROUTER_IP_ENV = os.getenv("ROUTER_IP", "").strip() or None

DATA_DIR = os.path.abspath(".")
LAYOUT_FILE = os.path.join(DATA_DIR, "roomsense_layout.json")
HISTORY_FILE = os.path.join(DATA_DIR, "roomsense_history.json")
STATE_FILE = os.path.join(DATA_DIR, "roomsense_state.json")

SAMPLE_INTERVAL_SEC = 1.0
SAVE_EVERY_SEC = 10.0
HISTORY_MAX_POINTS = 24 * 60 * 60  # about 24h at 1 sample/sec if wanted
UI_HISTORY_POINTS = 1800           # last ~30 min shown in chart
MAP_TRAIL_MAX = 400
EVENTS_MAX = 100

DEFAULT_LAYOUT = {
    "canvas": {"width": 1100, "height": 700},
    "sensor": {"x": 550, "y": 350},
    "walls": [],  # list of {"x1","y1","x2","y2"}
    "distance_scale": 160,
    "show_grid": True
}

# ------------------------------------------------------------
# Shared state
# ------------------------------------------------------------
lock = threading.Lock()
stop_flag = False
last_save_ts = 0.0

layout = DEFAULT_LAYOUT.copy()

events = deque(maxlen=EVENTS_MAX)
history = deque(maxlen=HISTORY_MAX_POINTS)
map_points = deque(maxlen=MAP_TRAIL_MAX)

state = {
    "room_name": ROOM_NAME,
    "router_ip": None,
    "occupied": False,
    "motion": False,
    "people_estimate": 0,
    "moving_people_estimate": 0,
    "confidence": 0.0,
    "score_total": 0.0,
    "score_ping": 0.0,
    "score_rssi": 0.0,
    "score_neighbors": 0.0,
    "score_camera": 0.0,
    "score_mic": 0.0,
    "ping_ms": None,
    "wifi_rssi_dbm": None,
    "arp_neighbors": 0,
    "webcam_motion": 0.0,
    "mic_level": 0.0,
    "cam_x": 0.0,
    "cam_y": 0.0,
    "distance_scale": DEFAULT_LAYOUT["distance_scale"],
    "calibrating": False,
    "calibration_remaining": 0,
    "sensors": {
        "ping": False,
        "wifi_rssi": False,
        "arp_neighbors": True,
        "webcam": ENABLE_WEBCAM,
        "microphone": ENABLE_MIC,
    }
}

baseline = {
    "ping_ms": None,
    "wifi_rssi_dbm": None,
    "arp_neighbors": None,
    "webcam_motion": 0.0,
    "mic_level": 0.0,
}

calibration_until = 0.0

# ------------------------------------------------------------
# Utility
# ------------------------------------------------------------
def now() -> float:
    return time.time()

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def safe_mean(values: List[float], default: float = 0.0) -> float:
    return statistics.mean(values) if values else default

def safe_stdev(values: List[float], default: float = 0.0) -> float:
    return statistics.stdev(values) if len(values) >= 2 else default

def push_event(msg: str) -> None:
    stamp = time.strftime("%H:%M:%S")
    events.appendleft(f"[{stamp}] {msg}")

def load_json(path: str, default: Any) -> Any:
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return default

def save_json(path: str, data: Any) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    os.replace(tmp, path)

def which(name: str) -> Optional[str]:
    return shutil.which(name)

# ------------------------------------------------------------
# Persistence
# ------------------------------------------------------------
def load_persisted() -> None:
    global layout

    loaded_layout = load_json(LAYOUT_FILE, DEFAULT_LAYOUT)
    if isinstance(loaded_layout, dict):
        layout = loaded_layout
        layout.setdefault("canvas", {"width": 1100, "height": 700})
        layout.setdefault("sensor", {"x": 550, "y": 350})
        layout.setdefault("walls", [])
        layout.setdefault("distance_scale", 160)
        layout.setdefault("show_grid", True)

    loaded_history = load_json(HISTORY_FILE, [])
    if isinstance(loaded_history, list):
        for item in loaded_history[-UI_HISTORY_POINTS:]:
            history.append(item)

    loaded_state = load_json(STATE_FILE, {})
    if isinstance(loaded_state, dict):
        state["distance_scale"] = loaded_state.get("distance_scale", layout["distance_scale"])

def persist_all() -> None:
    with lock:
        save_json(LAYOUT_FILE, layout)
        save_json(HISTORY_FILE, list(history)[-UI_HISTORY_POINTS:])
        save_json(STATE_FILE, {
            "distance_scale": state["distance_scale"],
            "router_ip": state["router_ip"],
        })

atexit.register(persist_all)

# ------------------------------------------------------------
# Router / network probes
# ------------------------------------------------------------
def run_cmd(cmd: List[str], timeout: int = 3) -> str:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return (p.stdout or "") + ("\n" + p.stderr if p.stderr else "")
    except Exception:
        return ""

def detect_router_ip() -> Optional[str]:
    if ROUTER_IP_ENV:
        return ROUTER_IP_ENV

    system = platform.system().lower()

    try:
        if "windows" in system:
            out = run_cmd(["ipconfig"])
            gateways = re.findall(r"Default Gateway[ .:]*([0-9]+\.[0-9]+\.[0-9]+\.[0-9]+)", out)
            for g in gateways:
                if g and g != "0.0.0.0":
                    return g

            out = run_cmd(["route", "print", "0.0.0.0"])
            m = re.search(r"0\.0\.0\.0\s+0\.0\.0\.0\s+(\d+\.\d+\.\d+\.\d+)", out)
            if m:
                return m.group(1)

        if which("ip"):
            out = run_cmd(["ip", "route"])
            for line in out.splitlines():
                if line.startswith("default via "):
                    parts = line.split()
                    if len(parts) >= 3:
                        return parts[2]

        if which("route"):
            out = run_cmd(["route", "-n"])
            for line in out.splitlines():
                parts = line.split()
                if len(parts) >= 2 and parts[0] == "0.0.0.0":
                    return parts[1]

        if "darwin" in system:
            out = run_cmd(["route", "-n", "get", "default"])
            m = re.search(r"gateway:\s+(\d+\.\d+\.\d+\.\d+)", out)
            if m:
                return m.group(1)
    except Exception:
        pass

    return None

def get_ping_ms(host: Optional[str]) -> Optional[float]:
    if not host:
        return None

    system = platform.system().lower()

    try:
        if "windows" in system:
            out = run_cmd(["ping", "-n", "1", "-w", "1000", host], timeout=3)
            m = re.search(r"time[=<]\s*(\d+)\s*ms", out, re.IGNORECASE)
            return float(m.group(1)) if m else None
        else:
            out = run_cmd(["ping", "-c", "1", "-W", "1", host], timeout=3)
            m = re.search(r"time=(\d+(\.\d+)?)\s*ms", out)
            return float(m.group(1)) if m else None
    except Exception:
        return None

def get_wifi_rssi() -> Optional[float]:
    system = platform.system().lower()

    try:
        if "windows" in system:
            out = run_cmd(["netsh", "wlan", "show", "interfaces"])
            m = re.search(r"^\s*Signal\s*:\s*(\d+)%", out, re.MULTILINE)
            if m:
                pct = float(m.group(1))
                return -100 + (pct * 0.5)

        elif "darwin" in system:
            airport = "/System/Library/PrivateFrameworks/Apple80211.framework/Versions/Current/Resources/airport"
            if os.path.exists(airport):
                out = run_cmd([airport, "-I"])
                m = re.search(r"agrCtlRSSI:\s*(-?\d+)", out)
                if m:
                    return float(m.group(1))

        else:
            if which("nmcli"):
                out = run_cmd(["nmcli", "-t", "-f", "ACTIVE,SIGNAL", "dev", "wifi"])
                for line in out.splitlines():
                    if line.startswith("yes:"):
                        pct = float(line.split(":")[1])
                        return -100 + (pct * 0.5)

            if which("iwconfig"):
                out = run_cmd(["iwconfig"])
                m = re.search(r"Signal level=(-?\d+)\s*dBm", out)
                if m:
                    return float(m.group(1))
    except Exception:
        pass

    return None

def get_arp_neighbor_count() -> int:
    system = platform.system().lower()

    try:
        if "windows" in system:
            out = run_cmd(["arp", "-a"])
            ips = set(re.findall(r"(\d+\.\d+\.\d+\.\d+)", out))
            return len(ips)

        out = run_cmd(["arp", "-a"])
        ips = set(re.findall(r"\((\d+\.\d+\.\d+\.\d+)\)", out))
        if ips:
            return len(ips)

        if which("ip"):
            out = run_cmd(["ip", "neigh"])
            ips = set(re.findall(r"(\d+\.\d+\.\d+\.\d+)", out))
            return len(ips)
    except Exception:
        pass

    return 0

# ------------------------------------------------------------
# Optional webcam / mic sensors
# ------------------------------------------------------------
class WebcamSensor:
    def __init__(self) -> None:
        self.available = False
        self.motion = 0.0
        self.cx = 0.0
        self.cy = 0.0
        self._cap = None
        self._prev = None

        if ENABLE_WEBCAM and cv2 is not None:
            t = threading.Thread(target=self._run, daemon=True)
            t.start()

    def _run(self) -> None:
        try:
            self._cap = cv2.VideoCapture(0)
            if not self._cap or not self._cap.isOpened():
                self.available = False
                return

            self.available = True

            while not stop_flag:
                ok, frame = self._cap.read()
                if not ok:
                    time.sleep(0.1)
                    continue

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, (9, 9), 0)

                if self._prev is not None:
                    diff = cv2.absdiff(self._prev, gray)
                    self.motion = float(diff.mean())

                    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
                    moments = cv2.moments(thresh)

                    if moments["m00"] > 0:
                        mx = moments["m10"] / moments["m00"]
                        my = moments["m01"] / moments["m00"]
                        h, w = gray.shape
                        self.cx = ((mx / w) * 2.0) - 1.0
                        self.cy = ((my / h) * 2.0) - 1.0
                    else:
                        self.cx = 0.0
                        self.cy = 0.0

                self._prev = gray
                time.sleep(0.08)
        except Exception:
            self.available = False

    def close(self) -> None:
        try:
            if self._cap is not None:
                self._cap.release()
        except Exception:
            pass

class MicSensor:
    def __init__(self) -> None:
        self.available = False
        self.level = 0.0
        self._stream = None

        if ENABLE_MIC and sd is not None and np is not None:
            try:
                self._stream = sd.InputStream(callback=self._callback, channels=1, samplerate=16000)
                self._stream.start()
                self.available = True
            except Exception:
                self.available = False

    def _callback(self, indata, frames, timing, status) -> None:
        try:
            self.level = float(np.sqrt(np.mean(np.square(indata))))
        except Exception:
            self.level = 0.0

    def close(self) -> None:
        try:
            if self._stream is not None:
                self._stream.stop()
                self._stream.close()
        except Exception:
            pass

webcam = WebcamSensor()
mic = MicSensor()

atexit.register(webcam.close)
atexit.register(mic.close)

# ------------------------------------------------------------
# Calibration
# ------------------------------------------------------------
def start_calibration(seconds: int = 15) -> None:
    global calibration_until
    with lock:
        calibration_until = now() + seconds
        state["calibrating"] = True
        state["calibration_remaining"] = seconds
    push_event("Calibration started. Keep room as empty/still as possible.")

def finish_calibration() -> None:
    with lock:
        recent = list(history)[-30:]
        if recent:
            ping_vals = [x["ping_ms"] for x in recent if x["ping_ms"] is not None]
            rssi_vals = [x["wifi_rssi_dbm"] for x in recent if x["wifi_rssi_dbm"] is not None]
            arp_vals = [float(x["arp_neighbors"]) for x in recent]
            cam_vals = [float(x["webcam_motion"]) for x in recent]
            mic_vals = [float(x["mic_level"]) for x in recent]

            baseline["ping_ms"] = safe_mean(ping_vals) if ping_vals else baseline["ping_ms"]
            baseline["wifi_rssi_dbm"] = safe_mean(rssi_vals) if rssi_vals else baseline["wifi_rssi_dbm"]
            baseline["arp_neighbors"] = safe_mean(arp_vals) if arp_vals else baseline["arp_neighbors"]
            baseline["webcam_motion"] = safe_mean(cam_vals, 0.0)
            baseline["mic_level"] = safe_mean(mic_vals, 0.0)

        state["calibrating"] = False
        state["calibration_remaining"] = 0

    push_event("Calibration complete.")

# ------------------------------------------------------------
# Motion map helpers
# ------------------------------------------------------------
def estimate_activity_point(score_total: float, cam_x: float, cam_y: float, motion: bool) -> Dict[str, Any]:
    sensor = layout.get("sensor", {"x": 550, "y": 350})
    distance_scale = float(state["distance_scale"])

    # Honest behavior:
    # - if webcam provides centroid, use that for direction
    # - otherwise use ring mode around sensor, not fake directional placement
    if webcam.available and motion:
        dist = clamp(25 + score_total * (distance_scale / 9.0), 20, distance_scale * 1.8)
        px = float(sensor["x"]) + cam_x * dist
        py = float(sensor["y"]) + cam_y * dist
        return {
            "mode": "dot",
            "x": px,
            "y": py,
            "radius": clamp(16 + score_total * 2.2, 16, 80),
            "strength": score_total,
            "ts": now(),
        }
    else:
        return {
            "mode": "ring",
            "x": float(sensor["x"]),
            "y": float(sensor["y"]),
            "radius": clamp(30 + score_total * (distance_scale / 8.0), 24, distance_scale * 1.8),
            "strength": score_total,
            "ts": now(),
        }

# ------------------------------------------------------------
# Inference
# ------------------------------------------------------------
ping_window = deque(maxlen=20)
rssi_window = deque(maxlen=20)
arp_window = deque(maxlen=20)
cam_window = deque(maxlen=20)
mic_window = deque(maxlen=20)

_prev = {
    "occupied": None,
    "motion": None,
    "people_estimate": None,
    "moving_people_estimate": None,
}

def update_inference(ping_ms: Optional[float], wifi_rssi_dbm: Optional[float], arp_neighbors: int) -> None:
    global last_save_ts, calibration_until

    if ping_ms is not None:
        ping_window.append(float(ping_ms))
    if wifi_rssi_dbm is not None:
        rssi_window.append(float(wifi_rssi_dbm))
    arp_window.append(float(arp_neighbors))
    cam_window.append(float(webcam.motion if webcam.available else 0.0))
    mic_window.append(float(mic.level if mic.available else 0.0))

    ping_mean = safe_mean(list(ping_window))
    ping_std = safe_stdev(list(ping_window))
    rssi_mean = safe_mean(list(rssi_window))
    rssi_std = safe_stdev(list(rssi_window))
    arp_mean = safe_mean(list(arp_window))
    cam_mean = safe_mean(list(cam_window))
    mic_mean = safe_mean(list(mic_window))

    if baseline["ping_ms"] is None and len(ping_window) >= 8:
        baseline["ping_ms"] = ping_mean
    if baseline["wifi_rssi_dbm"] is None and len(rssi_window) >= 8:
        baseline["wifi_rssi_dbm"] = rssi_mean
    if baseline["arp_neighbors"] is None and len(arp_window) >= 8:
        baseline["arp_neighbors"] = arp_mean

    score_ping = 0.0
    if baseline["ping_ms"] is not None and ping_window:
        score_ping = clamp(abs(ping_mean - float(baseline["ping_ms"])) * 0.45 + ping_std * 0.9, 0, 10)

    score_rssi = 0.0
    if baseline["wifi_rssi_dbm"] is not None and rssi_window:
        score_rssi = clamp(abs(rssi_mean - float(baseline["wifi_rssi_dbm"])) * 0.65 + rssi_std * 1.1, 0, 10)

    score_neighbors = 0.0
    if baseline["arp_neighbors"] is not None:
        score_neighbors = clamp(abs(arp_mean - float(baseline["arp_neighbors"])) * 1.2, 0, 10)

    score_camera = 0.0
    if webcam.available:
        score_camera = clamp((cam_mean - float(baseline["webcam_motion"])) * 1.8, 0, 10)

    score_mic = 0.0
    if mic.available:
        score_mic = clamp((mic_mean - float(baseline["mic_level"])) * 350.0, 0, 10)

    available_weight = 0.20 + 0.20 + 0.15
    if webcam.available:
        available_weight += 0.30
    if mic.available:
        available_weight += 0.15

    score_total = (
        score_ping * 0.20
        + score_rssi * 0.20
        + score_neighbors * 0.15
        + score_camera * 0.30
        + score_mic * 0.15
    ) / max(0.01, available_weight)

    motion = (
        score_ping > 1.6
        or score_rssi > 1.5
        or score_camera > 1.5
        or score_mic > 1.8
    )

    occupied = score_total > 1.25 or motion or score_neighbors > 1.4

    # Total people estimate: very rough
    people_estimate = 0
    if occupied:
        people_estimate = 1

    if baseline["arp_neighbors"] is not None:
        extra = max(0, int(round(arp_mean - float(baseline["arp_neighbors"]))))
        if extra > 0:
            people_estimate = max(people_estimate, min(5, 1 + extra))

    strong_signals = sum(1 for x in [score_ping, score_rssi, score_camera, score_mic] if x > 2.7)
    if strong_signals >= 2:
        people_estimate = max(people_estimate, 2)
    if strong_signals >= 3:
        people_estimate = max(people_estimate, 3)

    # Moving people estimate:
    # - webcam motion helps a lot
    # - otherwise conservative estimate
    moving_people_estimate = 0
    if motion:
        moving_people_estimate = 1

    if webcam.available:
        if score_camera > 6.0:
            moving_people_estimate = max(moving_people_estimate, 2)
        if score_camera > 8.0:
            moving_people_estimate = max(moving_people_estimate, 3)
    else:
        if strong_signals >= 2 and people_estimate >= 2:
            moving_people_estimate = 2

    moving_people_estimate = min(moving_people_estimate, people_estimate)

    confidence = clamp(score_total / 3.0, 0.0, 1.0)

    with lock:
        state["occupied"] = occupied
        state["motion"] = motion
        state["people_estimate"] = int(people_estimate)
        state["moving_people_estimate"] = int(moving_people_estimate)
        state["confidence"] = round(confidence, 3)
        state["score_total"] = round(score_total, 3)
        state["score_ping"] = round(score_ping, 3)
        state["score_rssi"] = round(score_rssi, 3)
        state["score_neighbors"] = round(score_neighbors, 3)
        state["score_camera"] = round(score_camera, 3)
        state["score_mic"] = round(score_mic, 3)
        state["ping_ms"] = ping_ms
        state["wifi_rssi_dbm"] = wifi_rssi_dbm
        state["arp_neighbors"] = arp_neighbors
        state["webcam_motion"] = round(webcam.motion if webcam.available else 0.0, 3)
        state["mic_level"] = round(mic.level if mic.available else 0.0, 5)
        state["cam_x"] = round(webcam.cx if webcam.available else 0.0, 3)
        state["cam_y"] = round(webcam.cy if webcam.available else 0.0, 3)
        state["distance_scale"] = layout.get("distance_scale", 160)
        state["sensors"] = {
            "ping": ping_ms is not None,
            "wifi_rssi": wifi_rssi_dbm is not None,
            "arp_neighbors": True,
            "webcam": webcam.available,
            "microphone": mic.available,
        }

        item = {
            "ts": now(),
            "occupied": occupied,
            "motion": motion,
            "people_estimate": int(people_estimate),
            "moving_people_estimate": int(moving_people_estimate),
            "confidence": round(confidence, 3),
            "score_total": round(score_total, 3),
            "score_ping": round(score_ping, 3),
            "score_rssi": round(score_rssi, 3),
            "score_neighbors": round(score_neighbors, 3),
            "score_camera": round(score_camera, 3),
            "score_mic": round(score_mic, 3),
            "ping_ms": ping_ms,
            "wifi_rssi_dbm": wifi_rssi_dbm,
            "arp_neighbors": arp_neighbors,
            "webcam_motion": round(webcam.motion if webcam.available else 0.0, 3),
            "mic_level": round(mic.level if mic.available else 0.0, 5),
        }
        history.append(item)

        if motion or occupied:
            map_points.append(estimate_activity_point(score_total, state["cam_x"], state["cam_y"], motion))

        if _prev["occupied"] is not None and _prev["occupied"] != occupied:
            push_event("Occupancy changed to OCCUPIED." if occupied else "Occupancy changed to EMPTY.")
        if _prev["motion"] is not None and _prev["motion"] != motion:
            push_event("Motion started." if motion else "Motion stopped.")
        if _prev["people_estimate"] is not None and _prev["people_estimate"] != people_estimate:
            push_event(f"People estimate changed to {people_estimate}.")
        if _prev["moving_people_estimate"] is not None and _prev["moving_people_estimate"] != moving_people_estimate:
            push_event(f"Moving people estimate changed to {moving_people_estimate}.")

        _prev["occupied"] = occupied
        _prev["motion"] = motion
        _prev["people_estimate"] = people_estimate
        _prev["moving_people_estimate"] = moving_people_estimate

    # calibration countdown
    if calibration_until > 0:
        remaining = int(max(0, calibration_until - now()))
        with lock:
            state["calibrating"] = remaining > 0
            state["calibration_remaining"] = remaining
        if remaining <= 0:
            calibration_until = 0.0
            finish_calibration()

    # periodic save
    if now() - last_save_ts > SAVE_EVERY_SEC:
        persist_all()

# ------------------------------------------------------------
# Sampling thread
# ------------------------------------------------------------
def sampling_loop() -> None:
    global last_save_ts
    router_ip = detect_router_ip()
    with lock:
        state["router_ip"] = router_ip

    if router_ip:
        push_event(f"Detected router/default gateway: {router_ip}")
    else:
        push_event("No router/default gateway detected. Ping-based sensing disabled.")

    push_event(f"RoomSense started for '{ROOM_NAME}'")

    while not stop_flag:
        ping_ms = get_ping_ms(router_ip) if router_ip else None
        wifi_rssi_dbm = get_wifi_rssi()
        arp_neighbors = get_arp_neighbor_count()

        update_inference(ping_ms, wifi_rssi_dbm, arp_neighbors)
        last_save_ts = now()
        time.sleep(SAMPLE_INTERVAL_SEC)

# ------------------------------------------------------------
# API
# ------------------------------------------------------------
load_persisted()

app = FastAPI()

@app.on_event("startup")
def on_startup() -> None:
    t = threading.Thread(target=sampling_loop, daemon=True)
    t.start()

@app.get("/api/state")
def api_state():
    with lock:
        return JSONResponse({
            **state,
            "events": list(events),
            "baseline": baseline,
        })

@app.get("/api/history")
def api_history():
    with lock:
        return JSONResponse({
            "items": list(history)[-UI_HISTORY_POINTS:],
        })

@app.get("/api/layout")
def api_layout():
    with lock:
        return JSONResponse(layout)

@app.post("/api/layout")
def api_layout_save(payload: Dict[str, Any] = Body(...)):
    with lock:
        if "canvas" in payload and isinstance(payload["canvas"], dict):
            layout["canvas"] = payload["canvas"]
        if "sensor" in payload and isinstance(payload["sensor"], dict):
            layout["sensor"] = payload["sensor"]
        if "walls" in payload and isinstance(payload["walls"], list):
            layout["walls"] = payload["walls"]
        if "distance_scale" in payload:
            try:
                layout["distance_scale"] = int(payload["distance_scale"])
                state["distance_scale"] = int(payload["distance_scale"])
            except Exception:
                pass
        if "show_grid" in payload:
            layout["show_grid"] = bool(payload["show_grid"])

    persist_all()
    push_event("Layout saved.")
    return {"ok": True}

@app.post("/api/layout/reset")
def api_layout_reset():
    global layout
    with lock:
        layout = json.loads(json.dumps(DEFAULT_LAYOUT))
        state["distance_scale"] = layout["distance_scale"]
    persist_all()
    push_event("Layout reset.")
    return {"ok": True}

@app.get("/api/map")
def api_map():
    with lock:
        return JSONResponse({
            "points": list(map_points),
            "sensor": layout.get("sensor", {"x": 550, "y": 350}),
            "distance_scale": layout.get("distance_scale", 160),
        })

@app.post("/api/calibrate")
def api_calibrate():
    start_calibration(15)
    return {"ok": True}

@app.post("/api/clear_history")
def api_clear_history():
    with lock:
        history.clear()
        map_points.clear()
        events.clear()
    persist_all()
    push_event("History cleared.")
    return {"ok": True}

@app.post("/api/distance/{value}")
def api_distance(value: int):
    with lock:
        value = int(clamp(value, 40, 500))
        layout["distance_scale"] = value
        state["distance_scale"] = value
    persist_all()
    return {"ok": True, "distance_scale": value}

# ------------------------------------------------------------
# HTML UI
# ------------------------------------------------------------
HTML = r"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>RoomSense Floorplan</title>
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <style>
    :root{
      --bg:#0d0f14;
      --panel:#171a22;
      --panel2:#11141b;
      --text:#eef2f7;
      --muted:#a5adba;
      --line:#2b3140;
      --accent:#67b7ff;
      --good:#4ade80;
      --warn:#fbbf24;
      --bad:#fb7185;
      --hot:rgba(255,90,90,0.24);
      --hotcore:rgba(255,120,120,0.92);
    }
    *{box-sizing:border-box}
    body{
      margin:0;
      background:linear-gradient(180deg,#0a0d13,#121722);
      color:var(--text);
      font-family:Inter,Segoe UI,Arial,sans-serif;
    }
    .wrap{
      max-width:1500px;
      margin:0 auto;
      padding:18px;
    }
    h1{
      margin:0 0 6px;
      font-size:34px;
      font-weight:800;
    }
    .sub{
      color:var(--muted);
      margin-bottom:16px;
    }
    .toolbar{
      display:flex;
      gap:10px;
      flex-wrap:wrap;
      align-items:center;
      margin-bottom:14px;
    }
    .toolbar button, .toolbar select{
      background:var(--panel);
      color:var(--text);
      border:1px solid var(--line);
      border-radius:12px;
      padding:10px 12px;
      cursor:pointer;
    }
    .toolbar button.primary{
      background:var(--accent);
      color:#07121e;
      border-color:transparent;
      font-weight:700;
    }
    .toolbar .pill{
      background:var(--panel);
      border:1px solid var(--line);
      border-radius:999px;
      padding:8px 12px;
      color:var(--muted);
      font-size:14px;
    }
    .grid{
      display:grid;
      grid-template-columns:repeat(12,1fr);
      gap:14px;
    }
    .card{
      background:linear-gradient(180deg,#171a22,#141821);
      border:1px solid var(--line);
      border-radius:18px;
      padding:16px;
      box-shadow:0 12px 26px rgba(0,0,0,.22);
    }
    .span-3{grid-column:span 3}
    .span-4{grid-column:span 4}
    .span-5{grid-column:span 5}
    .span-6{grid-column:span 6}
    .span-7{grid-column:span 7}
    .span-8{grid-column:span 8}
    .span-12{grid-column:span 12}
    @media (max-width:1200px){
      .span-3,.span-4,.span-5,.span-6,.span-7,.span-8,.span-12{grid-column:span 12}
    }
    .label{
      color:var(--muted);
      font-size:12px;
      text-transform:uppercase;
      letter-spacing:.08em;
      margin-bottom:8px;
    }
    .big{
      font-size:38px;
      font-weight:800;
      line-height:1.05;
    }
    .mid{
      font-size:26px;
      font-weight:700;
    }
    .good{color:var(--good)}
    .warn{color:var(--warn)}
    .bad{color:var(--bad)}
    .small{
      color:var(--muted);
      font-size:13px;
    }
    .row{
      display:flex;
      justify-content:space-between;
      gap:10px;
      align-items:center;
      margin:10px 0;
    }
    .bar{
      width:100%;
      height:12px;
      border-radius:999px;
      overflow:hidden;
      background:#0b0f16;
      border:1px solid var(--line);
    }
    .fill{
      width:0%;
      height:100%;
      background:linear-gradient(90deg,var(--accent),#9be7ff);
    }
    .editorbar{
      display:flex;
      gap:10px;
      flex-wrap:wrap;
      align-items:center;
      margin-bottom:12px;
    }
    .editorbar input[type=range]{
      width:220px;
    }
    canvas{
      width:100%;
      display:block;
      background:#0f131b;
      border:1px solid var(--line);
      border-radius:16px;
    }
    #mapCanvas{
      cursor:crosshair;
      min-height:620px;
    }
    .events{
      max-height:300px;
      overflow:auto;
      white-space:pre-wrap;
      font-family:ui-monospace,SFMono-Regular,Consolas,monospace;
      font-size:13px;
      line-height:1.5;
    }
    .sensorlist{
      display:grid;
      gap:10px;
    }
    .sensorrow{
      display:flex;
      justify-content:space-between;
      background:#0f131b;
      border:1px solid var(--line);
      border-radius:12px;
      padding:10px 12px;
    }
    .dot{
      display:inline-block;
      width:10px;
      height:10px;
      border-radius:999px;
      margin-right:8px;
      background:#666;
    }
    .dot.on{background:var(--good)}
    .dot.off{background:var(--bad)}
  </style>
</head>
<body>
<div class="wrap">
  <h1>RoomSense Floorplan</h1>
  <div class="sub">Draw your walls manually, place the sensor, then watch live motion/occupancy estimates and history curves. Walls are manual because plain home Wi-Fi cannot detect walls by itself.</div>

  <div class="toolbar">
    <button class="primary" onclick="saveLayout()">Save Layout</button>
    <button onclick="resetLayout()">Reset Layout</button>
    <button onclick="calibrate()">Calibrate Empty Room</button>
    <button onclick="clearHistory()">Clear History</button>
    <div class="pill" id="roomPill">Room: -</div>
    <div class="pill" id="routerPill">Router: -</div>
    <div class="pill" id="calibPill">Calibration: idle</div>
  </div>

  <div class="grid">
    <div class="card span-3">
      <div class="label">Occupied</div>
      <div class="big" id="occupied">-</div>
    </div>

    <div class="card span-3">
      <div class="label">Motion</div>
      <div class="big" id="motion">-</div>
    </div>

    <div class="card span-3">
      <div class="label">People Estimate</div>
      <div class="big" id="people">-</div>
    </div>

    <div class="card span-3">
      <div class="label">Moving People Estimate</div>
      <div class="big" id="movingPeople">-</div>
    </div>

    <div class="card span-8">
      <div class="label">Floorplan + Activity Map</div>
      <div class="editorbar">
        <label>Mode
          <select id="toolMode">
            <option value="wall">Draw Walls</option>
            <option value="sensor">Place Sensor</option>
            <option value="erase">Erase Wall</option>
          </select>
        </label>

        <label>
          Distance
          <input id="distanceSlider" type="range" min="40" max="500" step="5" value="160" oninput="distanceChanged(this.value)">
          <span id="distanceValue">160</span>
        </label>

        <label>
          <input type="checkbox" id="showGrid" checked onchange="drawMap()"> Show grid
        </label>

        <div class="small">Wall tool: click-drag to draw a wall segment.</div>
      </div>
      <canvas id="mapCanvas" width="1100" height="700"></canvas>
    </div>

    <div class="card span-4">
      <div class="label">Signal Scores</div>

      <div class="row"><div>Fused total</div><div id="vTotal">0</div></div>
      <div class="bar"><div class="fill" id="bTotal"></div></div>

      <div class="row"><div>Ping</div><div id="vPing">0</div></div>
      <div class="bar"><div class="fill" id="bPing"></div></div>

      <div class="row"><div>Wi-Fi RSSI</div><div id="vRssi">0</div></div>
      <div class="bar"><div class="fill" id="bRssi"></div></div>

      <div class="row"><div>Neighbors</div><div id="vNbr">0</div></div>
      <div class="bar"><div class="fill" id="bNbr"></div></div>

      <div class="row"><div>Camera</div><div id="vCam">0</div></div>
      <div class="bar"><div class="fill" id="bCam"></div></div>

      <div class="row"><div>Microphone</div><div id="vMic">0</div></div>
      <div class="bar"><div class="fill" id="bMic"></div></div>

      <div class="row" style="margin-top:16px"><div>Confidence</div><div id="confidence">0%</div></div>

      <div class="label" style="margin-top:18px">Sensors</div>
      <div class="sensorlist" id="sensorList"></div>
    </div>

    <div class="card span-7">
      <div class="label">History Curves</div>
      <canvas id="historyCanvas" width="1000" height="320"></canvas>
      <div class="small" style="margin-top:8px;">Blue = total score, green = people estimate, orange = moving people estimate. This persists while the app is running and is also saved to disk periodically.</div>
    </div>

    <div class="card span-5">
      <div class="label">Latest Raw Values</div>
      <div class="events" id="latestBlock"></div>
    </div>

    <div class="card span-12">
      <div class="label">Event Log</div>
      <div class="events" id="eventsBlock"></div>
    </div>
  </div>
</div>

<script>
  let appState = null;
  let layout = null;
  let historyItems = [];
  let mapItems = [];

  let drawingWall = false;
  let wallStart = null;

  const mapCanvas = document.getElementById("mapCanvas");
  const mapCtx = mapCanvas.getContext("2d");
  const historyCanvas = document.getElementById("historyCanvas");
  const historyCtx = historyCanvas.getContext("2d");

  function cls(v, yes="good", no="bad"){ return v ? yes : no; }
  function txt(v, yes="YES", no="NO"){ return v ? yes : no; }
  function pct(v,max=10){ return Math.max(0, Math.min(100, (Number(v||0)/max)*100)); }

  function setBar(name, value){
    const v = Number(value || 0);
    document.getElementById("v"+name).innerText = v.toFixed(2);
    document.getElementById("b"+name).style.width = pct(v) + "%";
  }

  async function fetchJSON(url, options){
    const r = await fetch(url, options || {});
    return await r.json();
  }

  async function loadEverything(){
    const [s, l, h, m] = await Promise.all([
      fetchJSON("/api/state"),
      fetchJSON("/api/layout"),
      fetchJSON("/api/history"),
      fetchJSON("/api/map")
    ]);
    appState = s;
    layout = l;
    historyItems = h.items || [];
    mapItems = m.points || [];
    syncUi();
    drawMap();
    drawHistory();
  }

  function syncUi(){
    document.getElementById("roomPill").innerText = "Room: " + (appState.room_name || "-");
    document.getElementById("routerPill").innerText = "Router: " + (appState.router_ip || "not found");
    document.getElementById("calibPill").innerText = appState.calibrating
      ? ("Calibration: " + appState.calibration_remaining + "s remaining")
      : "Calibration: idle";

    const occ = document.getElementById("occupied");
    occ.innerText = txt(appState.occupied, "OCCUPIED", "EMPTY");
    occ.className = "big " + cls(appState.occupied);

    const mot = document.getElementById("motion");
    mot.innerText = txt(appState.motion, "MOVING", "STILL");
    mot.className = "big " + (appState.motion ? "warn" : "good");

    document.getElementById("people").innerText = String(appState.people_estimate ?? 0);
    document.getElementById("movingPeople").innerText = String(appState.moving_people_estimate ?? 0);
    document.getElementById("confidence").innerText = Math.round((appState.confidence || 0) * 100) + "%";

    setBar("Total", appState.score_total);
    setBar("Ping", appState.score_ping);
    setBar("Rssi", appState.score_rssi);
    setBar("Nbr", appState.score_neighbors);
    setBar("Cam", appState.score_camera);
    setBar("Mic", appState.score_mic);

    document.getElementById("distanceSlider").value = layout.distance_scale || 160;
    document.getElementById("distanceValue").innerText = layout.distance_scale || 160;
    document.getElementById("showGrid").checked = !!layout.show_grid;

    const sensors = appState.sensors || {};
    const sensorNames = [
      ["ping", "Ping/router"],
      ["wifi_rssi", "Wi-Fi RSSI"],
      ["arp_neighbors", "ARP neighbors"],
      ["webcam", "Webcam"],
      ["microphone", "Microphone"]
    ];
    document.getElementById("sensorList").innerHTML = sensorNames.map(([key, label])=>{
      const on = !!sensors[key];
      return `<div class="sensorrow"><div><span class="dot ${on ? "on":"off"}"></span>${label}</div><div>${on ? "available" : "off"}</div></div>`;
    }).join("");

    const latest = {
      ping_ms: appState.ping_ms,
      wifi_rssi_dbm: appState.wifi_rssi_dbm,
      arp_neighbors: appState.arp_neighbors,
      webcam_motion: appState.webcam_motion,
      mic_level: appState.mic_level,
      cam_x: appState.cam_x,
      cam_y: appState.cam_y,
      baseline: appState.baseline
    };
    document.getElementById("latestBlock").textContent = JSON.stringify(latest, null, 2);
    document.getElementById("eventsBlock").textContent = (appState.events || []).join("\n");
  }

  function getMousePos(canvas, evt){
    const r = canvas.getBoundingClientRect();
    return {
      x: (evt.clientX - r.left) * (canvas.width / r.width),
      y: (evt.clientY - r.top) * (canvas.height / r.height)
    };
  }

  function distancePointToSegment(px, py, x1, y1, x2, y2){
    const dx = x2 - x1;
    const dy = y2 - y1;
    if(dx === 0 && dy === 0){
      return Math.hypot(px - x1, py - y1);
    }
    const t = Math.max(0, Math.min(1, ((px-x1)*dx + (py-y1)*dy) / (dx*dx + dy*dy)));
    const x = x1 + t * dx;
    const y = y1 + t * dy;
    return Math.hypot(px - x, py - y);
  }

  mapCanvas.addEventListener("mousedown", (evt)=>{
    if(!layout) return;
    const mode = document.getElementById("toolMode").value;
    const p = getMousePos(mapCanvas, evt);

    if(mode === "sensor"){
      layout.sensor = {x: p.x, y: p.y};
      drawMap();
      return;
    }

    if(mode === "erase"){
      const walls = layout.walls || [];
      let bestIndex = -1;
      let bestDist = 999999;
      walls.forEach((w, i)=>{
        const d = distancePointToSegment(p.x, p.y, w.x1, w.y1, w.x2, w.y2);
        if(d < bestDist){
          bestDist = d;
          bestIndex = i;
        }
      });
      if(bestIndex >= 0 && bestDist < 15){
        walls.splice(bestIndex, 1);
        drawMap();
      }
      return;
    }

    if(mode === "wall"){
      drawingWall = true;
      wallStart = p;
    }
  });

  mapCanvas.addEventListener("mouseup", (evt)=>{
    if(!layout) return;
    const mode = document.getElementById("toolMode").value;
    if(mode !== "wall" || !drawingWall || !wallStart) return;

    const p = getMousePos(mapCanvas, evt);
    drawingWall = false;

    if(Math.hypot(p.x - wallStart.x, p.y - wallStart.y) > 5){
      layout.walls.push({
        x1: wallStart.x,
        y1: wallStart.y,
        x2: p.x,
        y2: p.y
      });
      drawMap();
    }

    wallStart = null;
  });

  mapCanvas.addEventListener("mousemove", (evt)=>{
    if(!layout) return;
    if(drawingWall && wallStart){
      drawMap(getMousePos(mapCanvas, evt));
    }
  });

  function drawMap(previewEnd){
    if(!layout) return;

    mapCanvas.width = layout.canvas.width || 1100;
    mapCanvas.height = layout.canvas.height || 700;

    mapCtx.clearRect(0,0,mapCanvas.width,mapCanvas.height);

    if(layout.show_grid){
      mapCtx.strokeStyle = "#1e2531";
      mapCtx.lineWidth = 1;
      for(let x=0; x<mapCanvas.width; x+=40){
        mapCtx.beginPath();
        mapCtx.moveTo(x,0);
        mapCtx.lineTo(x,mapCanvas.height);
        mapCtx.stroke();
      }
      for(let y=0; y<mapCanvas.height; y+=40){
        mapCtx.beginPath();
        mapCtx.moveTo(0,y);
        mapCtx.lineTo(mapCanvas.width,y);
        mapCtx.stroke();
      }
    }

    // distance rings
    const sx = layout.sensor.x;
    const sy = layout.sensor.y;
    const ds = layout.distance_scale || 160;

    mapCtx.strokeStyle = "rgba(103,183,255,0.16)";
    mapCtx.lineWidth = 1.5;
    [ds*0.5, ds, ds*1.5].forEach(r=>{
      mapCtx.beginPath();
      mapCtx.arc(sx, sy, r, 0, Math.PI*2);
      mapCtx.stroke();
    });

    // activity map items
    const nowTs = Date.now()/1000;
    for(const p of mapItems){
      const age = Math.max(0, nowTs - (p.ts || nowTs));
      const fade = Math.max(0.05, 1.0 - age/35.0);

      if(p.mode === "dot"){
        mapCtx.fillStyle = `rgba(255,90,90,${0.10 * fade})`;
        mapCtx.beginPath();
        mapCtx.arc(p.x, p.y, (p.radius || 20) * 1.8, 0, Math.PI*2);
        mapCtx.fill();

        mapCtx.fillStyle = `rgba(255,120,120,${0.85 * fade})`;
        mapCtx.beginPath();
        mapCtx.arc(p.x, p.y, Math.max(5, (p.radius || 20) * 0.28), 0, Math.PI*2);
        mapCtx.fill();
      } else {
        mapCtx.strokeStyle = `rgba(255,120,120,${0.22 * fade})`;
        mapCtx.lineWidth = 3;
        mapCtx.beginPath();
        mapCtx.arc(p.x, p.y, p.radius || 40, 0, Math.PI*2);
        mapCtx.stroke();
      }
    }

    // walls
    mapCtx.strokeStyle = "#cbd5e1";
    mapCtx.lineWidth = 5;
    mapCtx.lineCap = "round";
    for(const w of layout.walls || []){
      mapCtx.beginPath();
      mapCtx.moveTo(w.x1, w.y1);
      mapCtx.lineTo(w.x2, w.y2);
      mapCtx.stroke();
    }

    // preview wall
    if(drawingWall && wallStart && previewEnd){
      mapCtx.strokeStyle = "rgba(255,255,255,0.5)";
      mapCtx.lineWidth = 3;
      mapCtx.beginPath();
      mapCtx.moveTo(wallStart.x, wallStart.y);
      mapCtx.lineTo(previewEnd.x, previewEnd.y);
      mapCtx.stroke();
    }

    // sensor
    mapCtx.fillStyle = "#ffffff";
    mapCtx.beginPath();
    mapCtx.arc(sx, sy, 8, 0, Math.PI*2);
    mapCtx.fill();

    mapCtx.fillStyle = "#67b7ff";
    mapCtx.font = "bold 14px Arial";
    mapCtx.fillText("Sensor", sx + 12, sy - 12);

    // live status text
    mapCtx.fillStyle = "rgba(255,255,255,0.86)";
    mapCtx.font = "13px Arial";
    mapCtx.fillText(`People: ${appState ? appState.people_estimate : 0}`, 16, 22);
    mapCtx.fillText(`Moving: ${appState ? appState.moving_people_estimate : 0}`, 16, 40);
    mapCtx.fillText(`Score: ${appState ? appState.score_total : 0}`, 16, 58);
  }

  function drawHistory(){
    historyCtx.clearRect(0,0,historyCanvas.width,historyCanvas.height);

    historyCtx.strokeStyle = "#202733";
    for(let i=0;i<6;i++){
      const y = (historyCanvas.height/5) * i;
      historyCtx.beginPath();
      historyCtx.moveTo(0,y);
      historyCtx.lineTo(historyCanvas.width,y);
      historyCtx.stroke();
    }

    if(historyItems.length < 2) return;

    const scoreVals = historyItems.map(x=>Number(x.score_total || 0));
    const peopleVals = historyItems.map(x=>Number(x.people_estimate || 0));
    const movingVals = historyItems.map(x=>Number(x.moving_people_estimate || 0));

    const maxY = Math.max(5, ...scoreVals, ...peopleVals, ...movingVals);

    function drawLine(values, color, width=2){
      historyCtx.strokeStyle = color;
      historyCtx.lineWidth = width;
      historyCtx.beginPath();
      values.forEach((v,i)=>{
        const x = (i / Math.max(1, values.length-1)) * historyCanvas.width;
        const y = historyCanvas.height - (v / maxY) * historyCanvas.height;
        if(i===0) historyCtx.moveTo(x,y);
        else historyCtx.lineTo(x,y);
      });
      historyCtx.stroke();
    }

    drawLine(scoreVals, "#67b7ff", 2.5);
    drawLine(peopleVals, "#4ade80", 2.5);
    drawLine(movingVals, "#fbbf24", 2.5);

    historyCtx.fillStyle = "#67b7ff";
    historyCtx.fillRect(10,10,12,12);
    historyCtx.fillStyle = "#dbe6f3";
    historyCtx.fillText("Total score", 28,20);

    historyCtx.fillStyle = "#4ade80";
    historyCtx.fillRect(130,10,12,12);
    historyCtx.fillStyle = "#dbe6f3";
    historyCtx.fillText("People est.", 148,20);

    historyCtx.fillStyle = "#fbbf24";
    historyCtx.fillRect(250,10,12,12);
    historyCtx.fillStyle = "#dbe6f3";
    historyCtx.fillText("Moving people est.", 268,20);
  }

  async function saveLayout(){
    layout.show_grid = document.getElementById("showGrid").checked;
    await fetch("/api/layout", {
      method:"POST",
      headers: {"Content-Type":"application/json"},
      body: JSON.stringify(layout)
    });
  }

  async function resetLayout(){
    await fetch("/api/layout/reset", {method:"POST"});
    await loadEverything();
  }

  async function calibrate(){
    await fetch("/api/calibrate", {method:"POST"});
  }

  async function clearHistory(){
    await fetch("/api/clear_history", {method:"POST"});
    await loadEverything();
  }

  async function distanceChanged(value){
    document.getElementById("distanceValue").innerText = value;
    layout.distance_scale = Number(value);
    await fetch("/api/distance/" + value, {method:"POST"});
    drawMap();
  }

  async function tick(){
    await loadEverything();
  }

  tick();
  setInterval(tick, 1000);
</script>
</body>
</html>
"""

@app.get("/")
def ui():
    return HTMLResponse(HTML)

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
if __name__ == "__main__":
    print(f"Open http://{HOST}:{PORT}")
    print("Optional examples:")
    print("  ENABLE_WEBCAM=1 python app.py")
    print("  ENABLE_MIC=1 python app.py")
    print("  ENABLE_WEBCAM=1 ENABLE_MIC=1 python app.py")
    uvicorn.run(app, host=HOST, port=PORT)
