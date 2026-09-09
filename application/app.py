import tkinter as tk
from tkinter import ttk, messagebox

import threading
import queue
import time
from collections import deque

import requests
import numpy as np
import joblib

from scipy.signal import resample, find_peaks
from scipy.signal import butter, filtfilt
from scipy.signal import lfilter, lfilter_zi

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# ============================================================
# CONFIG — MATCHES ESP32 .INO
# ============================================================

DEFAULT_IP = "192.168.4.1"

# ESP32 sampling rates
ECG_FS = 250
PPG_FS = 100

# Display
DISPLAY_SECONDS = 1

N_ECG_DISPLAY = ECG_FS * 4
N_PPG_DISPLAY = PPG_FS * DISPLAY_SECONDS * 5

# ============================================================
# TIMING
# ============================================================

GUI_UPDATE_MS = 40

REQUEST_TIMEOUT = 1.5

FETCH_SLEEP_S = 0.03

# BPM calculation interval
BPM_UPDATE_INTERVAL = 0.5

# SpO2 calculation interval
SPO2_UPDATE_INTERVAL = 0.5

# ML calculation interval
ML_UPDATE_INTERVAL = 1.0


# ============================================================
# BPM LIMITS
# ============================================================

BPM_MIN = 40
BPM_MAX = 180


# ============================================================
# ECG BANDPASS FILTER
# ============================================================

_bp_b, _bp_a = butter(
    2,
    [
        0.5 / (ECG_FS / 2),
        40 / (ECG_FS / 2)
    ],
    btype="band"
)

_bp_zi = lfilter_zi(
    _bp_b,
    _bp_a
) * 0.0

_bp_initialized = False


# ============================================================
# ML MODEL
# ============================================================

MODEL_PATH = "model/ecg_model.joblib"

try:

    ecg_model = joblib.load(
        MODEL_PATH
    )

    print(
        "ML MODEL LOADED SUCCESSFULLY"
    )

except Exception as e:

    ecg_model = None

    print(
        "ML MODEL LOAD ERROR:",
        e
    )


# ============================================================
# LIVE ECG → MODEL INPUT
# ============================================================

def prepare_live_beat(
    ecg,
    r_peak,
    fs=250
):

    """
    Convert one live ECG beat into the same
    205-feature format used during training.

    Training:
        MIT-BIH = 360 Hz
        100 samples before R peak
        100 samples after R peak
        Total = 200 waveform samples

    Live:
        ESP32 ECG = 250 Hz

    We therefore extract the same time duration
    and resample to exactly 200 samples.
    """

    TRAIN_FS = 360

    half_duration = (
        100 / TRAIN_FS
    )

    half_samples = int(
        round(
            half_duration * fs
        )
    )

    start = (
        r_peak -
        half_samples
    )

    end = (
        r_peak +
        half_samples
    )

    if start < 0 or end > len(ecg):

        return None

    beat = np.asarray(
        ecg[start:end],
        dtype=float
    )

    # Resample to 200 samples
    beat = resample(
        beat,
        200
    )

    # Normalize
    beat = (
        beat -
        np.mean(beat)
    )

    sd = np.std(
        beat
    )

    if sd == 0:

        return None

    beat = (
        beat /
        sd
    )

    # Five statistical features
    features = [

        np.mean(beat),

        np.std(beat),

        np.min(beat),

        np.max(beat),

        np.ptp(beat)

    ]

    # Add 200 waveform samples
    features.extend(
        beat.tolist()
    )

    return np.asarray(
        features,
        dtype=float
    )


# ============================================================
# ECG R-PEAK DETECTION
# ============================================================

def detect_live_r_peaks(
    ecg,
    fs=250
):

    x = np.asarray(
        ecg,
        dtype=float
    )

    if len(x) < fs:

        return np.array(
            [],
            dtype=int
        )

    # Remove DC offset
    x = (
        x -
        np.mean(x)
    )

    # Minimum distance between beats
    #
    # 0.20 sec = 300 BPM maximum
    #

    min_distance = int(
        0.30 * fs
    )

    signal_std = np.std(
        x
    )

    prominence = max(
        signal_std * 0.30,
        1e-7
    )

    # Positive peaks
    peaks, properties = find_peaks(

        x,

        distance=min_distance,

        prominence=prominence

    )

    # Negative peaks
    negative_peaks, negative_properties = find_peaks(

        -x,

        distance=min_distance,

        prominence=prominence

    )

    # If no positive peaks
    if len(peaks) == 0:

        return negative_peaks

    # If no negative peaks
    if len(negative_peaks) == 0:

        return peaks

    # Determine ECG polarity
    positive_strength = np.mean(
        properties["prominences"]
    )

    negative_strength = np.mean(
        negative_properties["prominences"]
    )

    if negative_strength > positive_strength:

        return negative_peaks

    return peaks


# ============================================================
# OPTIONAL STREAM FILTER
# ============================================================

def bandpass_filter_stream(
    new_samples
):

    global _bp_zi
    global _bp_initialized

    new_samples = np.asarray(
        new_samples,
        dtype=float
    )

    if len(new_samples) == 0:

        return np.array([])

    if not _bp_initialized:

        _bp_zi = (
            lfilter_zi(
                _bp_b,
                _bp_a
            )
            *
            new_samples[0]
        )

        _bp_initialized = True

    filtered, _bp_zi = lfilter(

        _bp_b,
        _bp_a,

        new_samples,

        zi=_bp_zi

    )

    return filtered


# ============================================================
# ECG → ML PREDICTION
# ============================================================

def predict_live_ecg(
    ecg
):

    if ecg_model is None:

        return None

    if len(ecg) < ECG_FS * 2:

        return None

    try:

        filtered = filtfilt(

            _bp_b,
            _bp_a,

            ecg

        )

    except Exception as e:

        print(
            "ECG filtering error:",
            e
        )

        return None

    peaks = detect_live_r_peaks(

        filtered,

        fs=ECG_FS

    )

    print(
        "Detected R-peaks:",
        len(peaks)
    )

    if len(peaks) == 0:

        messagebox.showwarning(
        "ECG Alert",
        "No R-peak detected!\n\n"
        "Please check the ECG electrodes and sensor connection."
    )
    return None

    TRAIN_FS = 360

    half_duration = (
        100 / TRAIN_FS
    )

    half_samples = int(
        round(
            half_duration *
            ECG_FS
        )
    )

    # Only use peaks with enough
    # ECG after them
    valid_peaks = [

        int(p)

        for p in peaks

        if (
            p - half_samples >= 0
            and
            p + half_samples <= len(filtered)
        )

    ]

    if len(valid_peaks) == 0:

        return None

    # Newest valid peak
    r_peak = valid_peaks[-1]

    features = prepare_live_beat(

        filtered,

        r_peak,

        fs=ECG_FS

    )

    if features is None:

        return None

    if len(features) != 205:

        print(
            "ERROR: Expected 205 features, got",
            len(features)
        )

        return None

    try:

        model_input = (
            features.reshape(
                1,
                -1
            )
        )

        prediction = (
            ecg_model.predict(
                model_input
            )[0]
        )

        probabilities = (
            ecg_model.predict_proba(
                model_input
            )[0]
        )

        confidence = float(
            np.max(
                probabilities
            ) * 100
        )

        print(
            "Full probabilities:",
            probabilities
        )

        print(
            "Classes:",
            ecg_model.classes_
        )

        print(
            "ML:",
            prediction,
            "| Confidence:",
            round(
                confidence,
                2
            ),
            "%"
        )

        return {

            "prediction":
                str(prediction),

            "confidence":
                confidence,

            "r_peak":
                r_peak,

            "feature_count":
                len(features)

        }

    except Exception as e:

        print(
            "ML prediction error:",
            e
        )

        return None


# ============================================================
# BPM CALCULATION
# IR ONLY
# ============================================================

def calculate_bpm(
    ir_samples,
    fs=100
):

    """
    Calculate heart rate from MAX30102 IR only.

    No RED signal is used here.

    The algorithm:
        1. Uses recent IR samples
        2. Removes DC
        3. Smooths the waveform
        4. Detects pulse peaks
        5. Calculates beat-to-beat intervals
        6. Converts intervals to BPM
        7. Rejects unrealistic values
    """

    # Need at least 5 seconds
    if len(ir_samples) < fs * 5:

        return None

    x = np.asarray(
        ir_samples,
        dtype=float
    )

    # Use latest 8 seconds
    max_samples = fs * 8

    if len(x) > max_samples:

        x = x[-max_samples:]

    # Remove DC component
    x = (
        x -
        np.mean(x)
    )

    # Smooth the PPG
    window = max(
        3,
        int(0.08 * fs)
    )

    kernel = (
        np.ones(window)
        /
        window
    )

    smooth = np.convolve(

        x,

        kernel,

        mode="same"

    )

    signal_std = np.std(
        smooth
    )

    if signal_std < 1e-6:

        return None

    # Minimum peak distance
    #
    # 180 BPM =
    # 3 beats/sec
    #
    # Therefore:
    # 100 samples/sec / 3
    # ≈ 33 samples

    min_distance = int(
        fs * 60 / BPM_MAX
    )

    prominence = (
        signal_std *
        0.35
    )

    peaks, properties = find_peaks(

        smooth,

        distance=min_distance,

        prominence=prominence

    )

    if len(peaks) < 2:

        return None

    # Time between pulse peaks
    intervals = (
        np.diff(peaks)
        /
        fs
    )

    if len(intervals) == 0:

        return None

    # Convert intervals to BPM
    bpm_values = (
        60.0 /
        intervals
    )

    # Keep realistic BPM
    valid = bpm_values[

        (bpm_values >= BPM_MIN)
        &
        (bpm_values <= BPM_MAX)

    ]

    if len(valid) == 0:

        return None

    # Median is more stable
    median_bpm = np.median(
        valid
    )

    # Remove sudden outliers
    stable = valid[

        np.abs(
            valid -
            median_bpm
        ) < 25

    ]

    if len(stable) == 0:

        stable = valid

    bpm = float(
        np.median(stable)
    )

    return round(
        bpm,
        1
    )


# ============================================================
# SPO2 CALCULATION
# RED + IR
# ============================================================

def calculate_spo2(
    ir_samples,
    red_samples,
    fs=100
):

    """
    Prototype SpO2 calculation using
    MAX30102 RED + IR.

    IMPORTANT:
    This is NOT medical-grade.
    Proper calibration is required for
    real medical use.
    """

    # Need at least 4 seconds
    minimum_samples = (
        fs * 4
    )

    if len(ir_samples) < minimum_samples:

        return None

    if len(red_samples) < minimum_samples:

        return None

    # Both channels must have
    # matching lengths

    n = min(

        len(ir_samples),

        len(red_samples)

    )

    ir = np.asarray(

        ir_samples[-n:],

        dtype=float

    )

    red = np.asarray(

        red_samples[-n:],

        dtype=float

    )

    # Check for invalid values

    if not np.all(
        np.isfinite(ir)
    ):

        return None

    if not np.all(
        np.isfinite(red)
    ):

        return None

    # DC components

    ir_dc = np.mean(
        ir
    )

    red_dc = np.mean(
        red
    )

    if ir_dc <= 0:

        return None

    if red_dc <= 0:

        return None

    # AC components

    ir_ac = np.std(
        ir
    )

    red_ac = np.std(
        red
    )

    if ir_ac <= 0:

        return None

    if red_ac <= 0:

        return None

    # Ratio-of-ratios

    ratio = (

        (red_ac / red_dc)

        /

        (ir_ac / ir_dc)

    )

    # Simple prototype calibration
    #
    # NOT medical-grade

    spo2 = (
        110.0 -
        (25.0 * ratio)
    )

    # Reject impossible values

    if spo2 < 70:

        return None

    if spo2 > 100:

        return None

    return round(
        float(spo2),
        1
    )


# ============================================================
# SHARED STATE
# ============================================================

running = False
# R-peak alert state
r_peak_alert_active = False


# ESP32 indexes
pulse_since = 0
ecg_since = 0

# Queue from receiver → GUI
data_queue = queue.Queue()

# Rolling buffers
ppg_buffer = deque(
    maxlen=N_PPG_DISPLAY
)

red_buffer = deque(
    maxlen=N_PPG_DISPLAY
)

ecg_buffer = deque(
    maxlen=N_ECG_DISPLAY
)

# Received counters
received_ppg = 0
received_red = 0
received_ecg = 0

# Current vitals
current_bpm = None
current_spo2 = None

# Timing
last_bpm_update = 0
last_spo2_update = 0
last_ml_update = 0

# Last valid values
last_valid_bpm_time = 0
last_valid_spo2_time = 0


# ============================================================
# ESP32 REQUEST
# ============================================================

def fetch_samples(
    session,
    base_url,
    endpoint,
    since
):

    response = session.get(

        f"{base_url}{endpoint}",

        params={
            "since": since
        },

        timeout=REQUEST_TIMEOUT

    )

    response.raise_for_status()

    return response.json()


# ============================================================
# BACKGROUND RECEIVER
# ============================================================

def receiver(
    ip
):

    global running
    global pulse_since
    global ecg_since

    base_url = (
        f"http://{ip}"
    )

    session = requests.Session()

    while running:

        try:

            # =================================================
            # MAX30102
            # =================================================

            pulse_payload = fetch_samples(

                session,

                base_url,

                "/data",

                pulse_since

            )

            data_queue.put({

                "type":
                    "ppg",

                "payload":
                    pulse_payload

            })


            # =================================================
            # ECG
            # =================================================

            ecg_payload = fetch_samples(

                session,

                base_url,

                "/ecg",

                ecg_since

            )

            data_queue.put({

                "type":
                    "ecg",

                "payload":
                    ecg_payload

            })


            # =================================================
            # UPDATE INDEXES
            # =================================================

            pulse_since = int(

                pulse_payload.get(

                    "latestIndex",

                    pulse_since

                )

            )

            ecg_since = int(

                ecg_payload.get(

                    "latestIndex",

                    ecg_since

                )

            )


            data_queue.put({

                "type":
                    "connected"

            })


        except (
            requests.RequestException,
            ValueError
        ) as exc:

            data_queue.put({

                "type":
                    "error",

                "message":
                    str(exc)

            })

            time.sleep(
                0.2
            )

        time.sleep(
            FETCH_SLEEP_S
        )


# ============================================================
# TKINTER APPLICATION
# ============================================================

class CardiacMonitorApp:

    def __init__(
        self,
        root
    ):

        self.root = root

        self.root.title(
            "ESP32-S3 Cardiac Monitor — ECG + PPG + BPM + SpO2"
        )

        self.root.geometry(
            "1600x900"
        )

        try:

            self.root.state(
                "zoomed"
            )

        except tk.TclError:

            pass

        self.build_ui()

        self.root.protocol(
            "WM_DELETE_WINDOW",
            self.close
        )


    # ========================================================
    # BUILD UI
    # ========================================================

    def build_ui(
        self
    ):

        self.root.configure(
            bg="black"
        )

        style = ttk.Style()

        style.theme_use(
            "clam"
        )

        style.configure(
            ".",
            background="black",
            foreground="#39FF14"
        )

        style.configure(
            "TFrame",
            background="black"
        )

        style.configure(
            "TLabel",
            background="black",
            foreground="white"
        )

        style.configure(
            "TButton",
            background="#111111",
            foreground="#39FF14"
        )

        style.configure(
            "TEntry",
            fieldbackground="#111111",
            foreground="#39FF14"
        )


        # ====================================================
        # TITLE
        # ====================================================

        title_frame = ttk.Frame(
            self.root
        )

        title_frame.pack(
            fill="x",
            padx=20,
            pady=(12, 5)
        )

        ttk.Label(

            title_frame,

            text=
            "ESP32-S3 CARDIAC MONITOR",

            font=
            (
                "Arial",
                24,
                "bold"
            )

        ).pack(
            anchor="center"
        )


        # ====================================================
        # CONTROLS
        # ====================================================

        control_frame = ttk.Frame(
            self.root
        )

        control_frame.pack(
            fill="x",
            padx=15,
            pady=5
        )

        ttk.Label(

            control_frame,

            text="ESP32 IP:"

        ).pack(
            side="left",
            padx=4
        )


        self.ip_entry = ttk.Entry(

            control_frame,

            width=18

        )

        self.ip_entry.insert(

            0,

            DEFAULT_IP

        )

        self.ip_entry.pack(

            side="left",

            padx=4

        )


        self.start_button = ttk.Button(

            control_frame,

            text="START",

            command=self.start

        )

        self.start_button.pack(

            side="left",

            padx=4

        )


        self.stop_button = ttk.Button(

            control_frame,

            text="STOP",

            command=self.stop

        )

        self.stop_button.pack(

            side="left",

            padx=4

        )


        # ====================================================
        # STATUS
        # ====================================================

        info_frame = ttk.Frame(
            self.root
        )

        info_frame.pack(

            fill="x",

            padx=15,

            pady=5

        )


        self.status_var = tk.StringVar(

            value="Status: Ready"

        )


        ttk.Label(

            info_frame,

            textvariable=
            self.status_var,

            font=
            (
                "Arial",
                11
            )

        ).pack(
            side="left"
        )


        self.sample_var = tk.StringVar(

            value=
            "PC received — "
            "ECG: 0 | "
            "PPG: 0 | "
            "RED: 0"

        )


        ttk.Label(

            info_frame,

            textvariable=
            self.sample_var,

            font=
            (
                "Arial",
                11
            )

        ).pack(
            side="right"
        )


        # ====================================================
        # VITAL CARDS
        # ====================================================

        vitals_frame = tk.Frame(

            self.root,

            bg="black"

        )

        vitals_frame.pack(

            fill="x",

            padx=15,

            pady=8

        )


        # ====================================================
        # BPM CARD
        # ====================================================

        bpm_card = tk.Frame(

            vitals_frame,

            bg="black",

            highlightbackground="white",

            highlightthickness=2

        )

        bpm_card.pack(

            side="left",

            expand=True,

            fill="x",

            padx=8

        )


        tk.Label(

            bpm_card,

            text="HEART RATE",

            bg="black",

            fg="white",

            font=
            (
                "Arial",
                13,
                "bold"
            )

        ).pack(
            pady=(8, 2)
        )


        self.bpm_var = tk.StringVar(

            value="-- BPM"

        )


        tk.Label(

            bpm_card,

            textvariable=
            self.bpm_var,

            bg="black",

            fg="#39FF14",

            font=
            (
                "Arial",
                28,
                "bold"
            )

        ).pack(
            pady=(2, 8)
        )


        # ====================================================
        # SPO2 CARD
        # ====================================================

        spo2_card = tk.Frame(

            vitals_frame,

            bg="black",

            highlightbackground="white",

            highlightthickness=2

        )

        spo2_card.pack(

            side="left",

            expand=True,

            fill="x",

            padx=8

        )


        tk.Label(

            spo2_card,

            text="BLOOD OXYGEN",

            bg="black",

            fg="white",

            font=
            (
                "Arial",
                13,
                "bold"
            )

        ).pack(
            pady=(8, 2)
        )


        self.spo2_var = tk.StringVar(

            value="-- %"

        )


        tk.Label(

            spo2_card,

            textvariable=
            self.spo2_var,

            bg="black",

            fg="#39FF14",

            font=
            (
                "Arial",
                28,
                "bold"
            )

        ).pack(
            pady=(2, 8)
        )


        # ====================================================
        # ML RESULT
        # ====================================================

        ml_frame = ttk.Frame(

            self.root

        )

        ml_frame.pack(

            fill="x",

            padx=15,

            pady=5

        )


        self.ml_var = tk.StringVar(

            value=
            "ML Result: Waiting for ECG..."

        )


        ttk.Label(

            ml_frame,

            textvariable=
            self.ml_var,

            font=
            (
                "Arial",
                14,
                "bold"
            )

        ).pack(
            side="left"
        )


        # ====================================================
        # GRAPH AREA
        # ====================================================

        plot_frame = ttk.Frame(

            self.root

        )

        plot_frame.pack(

            fill="both",

            expand=True,

            padx=12,

            pady=8

        )


        # ====================================================
        # ECG GRAPH
        # ====================================================

        self.ecg_figure = Figure(

            figsize=(9, 7),

            dpi=100,

            facecolor="black"

        )


        self.ecg_axis = (

            self.ecg_figure
            .add_subplot(111)

        )


        self.ecg_axis.set_facecolor(
            "black"
        )


        self.ecg_axis.set_title(

            "ECG — AD8232",

            fontsize=18,

            fontweight="bold",

            color="white"

        )


        self.ecg_axis.set_xlabel(

            "Time (seconds ago)",

            fontsize=12,

            color="white"

        )


        self.ecg_axis.set_ylabel(

            "ADC value",

            fontsize=12,

            color="white"

        )


        self.ecg_axis.set_xlim(

            -DISPLAY_SECONDS,

            0

        )


        self.ecg_axis.tick_params(

            colors="white"

        )


        self.ecg_axis.grid(

            True,

            alpha=0.3

        )


        # White graph outline

        for spine in self.ecg_axis.spines.values():

            spine.set_color(
                "white"
            )


        self.ecg_line, = (

            self.ecg_axis.plot(

                [],

                [],

                linewidth=1.0,

                color="#39FF14"

            )

        )


        self.ecg_canvas = (

            FigureCanvasTkAgg(

                self.ecg_figure,

                master=plot_frame

            )

        )


        self.ecg_canvas.get_tk_widget().pack(

            side="left",

            fill="both",

            expand=True,

            padx=(0, 6)

        )


        # ====================================================
        # PPG GRAPH
        # ====================================================

        self.ppg_figure = Figure(

            figsize=(9, 7),

            dpi=100,

            facecolor="black"

        )


        self.ppg_axis = (

            self.ppg_figure
            .add_subplot(111)

        )


        self.ppg_axis.set_facecolor(
            "black"
        )


        self.ppg_axis.set_title(

            "PPG — MAX30102 IR",

            fontsize=18,

            fontweight="bold",

            color="white"

        )


        self.ppg_axis.set_xlabel(

            "Time (seconds ago)",

            fontsize=12,

            color="white"

        )


        self.ppg_axis.set_ylabel(

            "IR value",

            fontsize=12,

            color="white"

        )


        self.ppg_axis.set_xlim(

            -DISPLAY_SECONDS * 4,

            0

        )


        self.ppg_axis.tick_params(

            colors="white"

        )


        self.ppg_axis.grid(

            True,

            alpha=0.3

        )


        # White graph outline

        for spine in self.ppg_axis.spines.values():

            spine.set_color(
                "white"
            )


        self.ppg_line, = (

            self.ppg_axis.plot(

                [],

                [],

                linewidth=1.0,

                color="#39FF14"

            )

        )


        self.ppg_canvas = (

            FigureCanvasTkAgg(

                self.ppg_figure,

                master=plot_frame

            )

        )


        self.ppg_canvas.get_tk_widget().pack(

            side="left",

            fill="both",

            expand=True,

            padx=(6, 0)

        )


        # ====================================================
        # FOOTER
        # ====================================================

        ttk.Label(

            self.root,

            text=(

                "BPM = IR only  |  "

                "SpO2 = RED + IR  |  "

                "ECG = AD8232  |  "

                "ML = MIT-BIH model  |  "

                "Prototype — not medical-grade"

            ),

            font=
            (
                "Arial",
                9
            )

        ).pack(

            pady=(0, 8)

        )


    # ========================================================
    # START
    # ========================================================

    def start(
        self
    ):

        global running

        global pulse_since
        global ecg_since

        global received_ppg
        global received_red
        global received_ecg

        global current_bpm
        global current_spo2

        global last_bpm_update
        global last_spo2_update
        global last_ml_update

        global last_valid_bpm_time
        global last_valid_spo2_time


        if running:

            return


        ip = (
            self.ip_entry
            .get()
            .strip()
        )


        if not ip:

            messagebox.showerror(

                "Missing IP",

                "Enter the ESP32 IP address."

            )

            return


        # Reset state

        running = True

        pulse_since = 0
        ecg_since = 0

        received_ppg = 0
        received_red = 0
        received_ecg = 0

        current_bpm = None
        current_spo2 = None

        last_bpm_update = 0
        last_spo2_update = 0
        last_ml_update = 0

        last_valid_bpm_time = 0
        last_valid_spo2_time = 0


        # Clear buffers

        ppg_buffer.clear()

        red_buffer.clear()

        ecg_buffer.clear()


        # Clear queue

        while True:

            try:

                data_queue.get_nowait()

            except queue.Empty:

                break


        self.bpm_var.set(
            "-- BPM"
        )

        self.spo2_var.set(
            "-- %"
        )

        self.ml_var.set(
            "ML Result: Waiting for ECG..."
        )


        self.status_var.set(

            f"Status: Connecting to {ip}..."

        )


        # Start receiver

        threading.Thread(

            target=receiver,

            args=(ip,),

            daemon=True

        ).start()


        # Start GUI update

        self.update_gui()


    # ========================================================
    # STOP
    # ========================================================

    def stop(
        self
    ):

        global running

        running = False

        self.status_var.set(
            "Status: Stopped"
        )


    # ========================================================
    # GUI UPDATE
    # ========================================================

    def update_gui(
        self
    ):

        global running

        global received_ppg
        global received_red
        global received_ecg

        global current_bpm
        global current_spo2

        global last_bpm_update
        global last_spo2_update
        global last_ml_update

        global last_valid_bpm_time
        global last_valid_spo2_time


        if not running:

            return


        connected = False

        error_message = None


        # ====================================================
        # READ QUEUE
        # ====================================================

        while True:

            try:

                item = (
                    data_queue
                    .get_nowait()
                )

            except queue.Empty:

                break


            item_type = item.get(
                "type"
            )


            # =================================================
            # PPG + RED
            # =================================================

            if item_type == "ppg":

                payload = item[
                    "payload"
                ]


                # IR samples

                samples = payload.get(

                    "samples",

                    []

                )


                # RED samples
                #
                # This comes from the
                # modified ESP32 .INO

                red_samples = payload.get(

                    "redSamples",

                    []

                )


                # Store IR

                for value in samples:

                    try:

                        ppg_buffer.append(

                            float(value)

                        )

                        received_ppg += 1

                    except (
                        ValueError,
                        TypeError
                    ):

                        pass


                # Store RED

                for value in red_samples:

                    try:

                        red_buffer.append(

                            float(value)

                        )

                        received_red += 1

                    except (
                        ValueError,
                        TypeError
                    ):

                        pass


            # =================================================
            # ECG
            # =================================================

            elif item_type == "ecg":

                payload = item[
                    "payload"
                ]


                samples = payload.get(

                    "samples",

                    []

                )


                for value in samples:

                    try:

                        ecg_buffer.append(

                            float(value)

                        )

                        received_ecg += 1

                    except (
                        ValueError,
                        TypeError
                    ):

                        pass


            # =================================================
            # CONNECTED
            # =================================================

            elif item_type == "connected":

                connected = True


            # =================================================
            # ERROR
            # =================================================

            elif item_type == "error":

                error_message = item.get(

                    "message",

                    "Unknown error"

                )


        # ====================================================
        # STATUS
        # ====================================================

        if error_message:

            self.status_var.set(

                "Status: Connection error — "
                + error_message

            )

        elif connected:

            self.status_var.set(

                "Status: Receiving from ESP32"

            )


        # ====================================================
        # SAMPLE COUNTER
        # ====================================================

        self.sample_var.set(

            f"PC received — "

            f"ECG: {received_ecg} | "

            f"PPG: {received_ppg} | "

            f"RED: {received_red}"

        )


        now = time.time()


        # ====================================================
        # BPM
        # ====================================================

        if (
            now -
            last_bpm_update
            >=
            BPM_UPDATE_INTERVAL
        ):

            last_bpm_update = now


            # Need at least 5 seconds
            # of IR

            if len(ppg_buffer) >= PPG_FS * 5:

                bpm = calculate_bpm(

                    list(ppg_buffer),

                    PPG_FS

                )


                if bpm is not None:

                    current_bpm = bpm

                    last_valid_bpm_time = now

                    self.bpm_var.set(

                        f"{current_bpm:.1f} BPM"

                    )

                else:

                    # Keep last good value
                    # for a few seconds

                    if (
                        now -
                        last_valid_bpm_time
                        > 3
                    ):

                        self.bpm_var.set(
                            "-- BPM"
                        )


        # ====================================================
        # SPO2
        # ====================================================

        if (
            now -
            last_spo2_update
            >=
            SPO2_UPDATE_INTERVAL
        ):

            last_spo2_update = now


            # Need 4 seconds of BOTH
            # IR and RED

            if (

                len(ppg_buffer)
                >=
                PPG_FS * 4

                and

                len(red_buffer)
                >=
                PPG_FS * 4

            ):

                spo2 = calculate_spo2(

                    list(ppg_buffer),

                    list(red_buffer),

                    PPG_FS

                )


                if spo2 is not None:

                    current_spo2 = spo2

                    last_valid_spo2_time = now

                    self.spo2_var.set(

                        f"{current_spo2:.1f} %"

                    )

                else:

                    if (
                        now -
                        last_valid_spo2_time
                        > 3
                    ):

                        self.spo2_var.set(
                            "-- %"
                        )


        # ====================================================
        # ML ECG
        # ====================================================

        if (
            now -
            last_ml_update
            >=
            ML_UPDATE_INTERVAL
        ):

            last_ml_update = now


            if len(ecg_buffer) >= ECG_FS * 2:

                ecg_for_ml = np.asarray(

                    ecg_buffer,

                    dtype=float

                )


                ml_result = (
                    predict_live_ecg(
                        ecg_for_ml
                    )
                )


                if ml_result is not None:

                    prediction = (
                        ml_result[
                            "prediction"
                        ]
                    )

                    confidence = (
                        ml_result[
                            "confidence"
                        ]
                    )


                    self.ml_var.set(

                        f"ML Result: "
                        f"{prediction} "
                        f"| Confidence: "
                        f"{confidence:.1f}%"

                    )

                else:

                    self.ml_var.set(

                        "ML Result: Analyzing ECG..."

                    )


        # ====================================================
        # ECG GRAPH
        # ====================================================

        if len(ecg_buffer) > 0:

            ecg_values = np.asarray(

                ecg_buffer,

                dtype=float

            )


            ecg_time = (

                np.arange(
                    len(ecg_values)
                )
                /
                ECG_FS

            )


            ecg_time = (

                ecg_time -
                ecg_time[-1]

            )


            self.ecg_line.set_data(

                ecg_time,

                ecg_values

            )


            self.ecg_axis.set_xlim(

                -DISPLAY_SECONDS,

                0

            )


            ymin = np.min(
                ecg_values
            )

            ymax = np.max(
                ecg_values
            )


            if ymax == ymin:

                padding = 1

            else:

                padding = (

                    ymax -
                    ymin

                ) * 0.10


            self.ecg_axis.set_ylim(

                ymin - padding,

                ymax + padding

            )


            self.ecg_canvas.draw_idle()


        # ====================================================
        # PPG GRAPH
        # ====================================================

        if len(ppg_buffer) > 0:

            ppg_values = np.asarray(

                ppg_buffer,

                dtype=float

            )


            ppg_time = (

                np.arange(
                    len(ppg_values)
                )
                /
                PPG_FS

            )


            ppg_time = (

                ppg_time -
                ppg_time[-1]

            )


            self.ppg_line.set_data(

                ppg_time,

                ppg_values

            )


            self.ppg_axis.set_xlim(

                -DISPLAY_SECONDS * 4,

                0

            )


            ymin = np.min(
                ppg_values
            )

            ymax = np.max(
                ppg_values
            )


            if ymax == ymin:

                padding = 1

            else:

                padding = (

                    ymax -
                    ymin

                ) * 0.10


            self.ppg_axis.set_ylim(

                ymin - padding,

                ymax + padding

            )


            self.ppg_canvas.draw_idle()


        # ====================================================
        # NEXT GUI UPDATE
        # ====================================================

        self.root.after(

            GUI_UPDATE_MS,

            self.update_gui

        )


    # ========================================================
    # CLOSE
    # ========================================================

    def close(
        self
    ):

        global running

        running = False

        try:

            self.root.destroy()

        except Exception:

            pass


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    root = tk.Tk()

    app = CardiacMonitorApp(
        root
    )

    root.mainloop()