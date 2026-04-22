import cv2
import pickle
import cvzone
import numpy as np
import streamlit as st
from PIL import Image
import io

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Smart Parking Analyzer", layout="wide", initial_sidebar_state="expanded")

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Syne', sans-serif; background-color: #0d0d0d; color: #f0f0f0; }
.stApp { background: #0d0d0d; }

.hero { text-align: center; padding: 2rem 1rem 1rem; }
.hero h1 {
    font-size: 3rem; font-weight: 800; letter-spacing: -1px;
    background: linear-gradient(90deg, #00ff88, #00cfff);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin: 0;
}
.hero p { color: #888; font-family: 'Space Mono', monospace; font-size: 0.8rem; margin-top: 0.4rem; letter-spacing: 2px; text-transform: uppercase; }

.stats-bar {
    display: flex; justify-content: center; gap: 2rem;
    padding: 1.2rem 2rem; background: #161616;
    border: 1px solid #222; border-radius: 12px; margin: 1.2rem 0;
}
.stat-item { text-align: center; }
.stat-value { font-size: 2.2rem; font-weight: 800; line-height: 1; }
.stat-label { font-family: 'Space Mono', monospace; font-size: 0.65rem; letter-spacing: 2px; text-transform: uppercase; color: #555; margin-top: 0.3rem; }
.stat-free { color: #00ff88; } .stat-occ { color: #ff4444; } .stat-total { color: #00cfff; }

.img-label { font-family: 'Space Mono', monospace; font-size: 0.7rem; letter-spacing: 3px; text-transform: uppercase; color: #444; margin-bottom: 0.5rem; }

div[data-testid="stFileUploader"] { border: 2px dashed #2a2a2a !important; border-radius: 12px !important; background: #111 !important; padding: 1rem; }
div[data-testid="stFileUploader"]:hover { border-color: #00ff88 !important; }
.stButton > button { background: linear-gradient(135deg, #00ff88, #00cfff) !important; color: #000 !important; border: none !important; border-radius: 8px !important; font-family: 'Space Mono', monospace !important; font-weight: 700 !important; }
</style>
""", unsafe_allow_html=True)


# ── Load parking positions ────────────────────────────────────────────────────
@st.cache_resource
def load_positions():
    try:
        with open('CarParkPos', 'rb') as f:
            return pickle.load(f)
    except Exception:
        return []

posList = load_positions()


# ── Auto-calibrate threshold ──────────────────────────────────────────────────
def auto_calibrate(img_bgr: np.ndarray, pos_list: list):
    """
    Collect nonzero pixel counts for every space, then find the natural gap
    between 'empty' and 'occupied' clusters using a histogram valley method.
    Returns (suggested_threshold, all_counts)
    """
    imgGray      = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    imgBlur      = cv2.GaussianBlur(imgGray, (3, 3), 1)
    imgThreshold = cv2.adaptiveThreshold(imgBlur, 255,
                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                       cv2.THRESH_BINARY_INV, 25, 16)
    imgMedian    = cv2.medianBlur(imgThreshold, 5)
    kernel       = np.ones((3, 3), np.uint8)
    imgDilate    = cv2.dilate(imgMedian, kernel, iterations=1)

    counts = []
    for pos in pos_list:
        x, y = pos
        crop = imgDilate[y:y + 33, x:x + 79]
        counts.append(cv2.countNonZero(crop))

    if not counts:
        return 900, counts

    counts_arr = np.array(counts, dtype=float)

    # Build histogram with 30 bins
    hist, edges = np.histogram(counts_arr, bins=30)

    # Find the valley (minimum density) to the right of the first peak
    peak_idx   = int(np.argmax(hist))
    valley_val = None

    for i in range(peak_idx + 1, len(hist)):
        if i + 1 < len(hist) and hist[i] < hist[i - 1] and hist[i] <= hist[i + 1]:
            valley_val = int((edges[i] + edges[i + 1]) / 2)
            break

    # Fallback: midpoint between low-cluster median and high-cluster median
    if valley_val is None:
        median_all = float(np.median(counts_arr))
        low_vals   = counts_arr[counts_arr <= median_all]
        high_vals  = counts_arr[counts_arr > median_all]
        low_med    = int(np.median(low_vals))  if len(low_vals)  else 0
        high_med   = int(np.median(high_vals)) if len(high_vals) else int(median_all * 2)
        valley_val = (low_med + high_med) // 2

    valley_val = max(100, min(valley_val, 2500))
    return valley_val, [int(c) for c in counts_arr]


# ── Core analysis function ────────────────────────────────────────────────────
def analyze_parking(img_bgr: np.ndarray, threshold: int):
    imgGray      = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    imgBlur      = cv2.GaussianBlur(imgGray, (3, 3), 1)
    imgThreshold = cv2.adaptiveThreshold(imgBlur, 255,
                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                       cv2.THRESH_BINARY_INV, 25, 16)
    imgMedian    = cv2.medianBlur(imgThreshold, 5)
    kernel       = np.ones((3, 3), np.uint8)
    imgDilate    = cv2.dilate(imgMedian, kernel, iterations=1)

    annotated  = img_bgr.copy()
    free_count = 0

    for pos in posList:
        x, y  = pos
        crop  = imgDilate[y:y + 33, x:x + 79]
        count = cv2.countNonZero(crop)

        if count < threshold:
            color     = (0, 255, 0)
            thickness = 5
            free_count += 1
        else:
            color     = (0, 0, 255)
            thickness = 2

        cv2.rectangle(annotated, pos, (pos[0] + 79, pos[1] + 33), color, thickness)
        cvzone.putTextRect(annotated, str(count), (x, y + 33 - 3),
                           scale=1, thickness=2, offset=0, colorR=color)

    total    = len(posList)
    occupied = total - free_count
    cvzone.putTextRect(annotated, f'Free: {free_count}/{total}',
                       (100, 50), scale=3, thickness=5, offset=20, colorR=(0, 200, 0))

    return cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), free_count, occupied, total


# ── UI ────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>PARKVISION</h1>
    <p>Intelligent Parking Space Analyzer</p>
</div>
""", unsafe_allow_html=True)

uploaded = st.file_uploader("Upload a parking lot frame", type=["png", "jpg", "jpeg", "webp"])

if uploaded is not None:
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    img_bgr    = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img_bgr is None:
        st.error("Could not decode image.")
        st.stop()

    if len(posList) == 0:
        st.warning("⚠️ No parking positions loaded. Place `CarParkPos` in the same directory as app.py.")
        st.stop()

    # ── Auto-calibrate on this image ──
    suggested_thresh, all_counts = auto_calibrate(img_bgr, posList)

    # ── Sidebar controls ──
    st.sidebar.markdown("## 🎛️ Threshold Control")
    st.sidebar.markdown(f"**Auto-calibrated:** `{suggested_thresh}`")
    st.sidebar.caption("Computed by finding the natural valley between empty and occupied pixel-count clusters for this image.")

    threshold = st.sidebar.slider(
        "Occupancy Threshold",
        min_value=50,
        max_value=2500,
        value=suggested_thresh,
        step=10,
        help="Spaces with pixel count BELOW this = FREE (green). Raise if too many false-free; lower if too many false-occupied."
    )

    if all_counts:
        arr = np.array(all_counts)
        st.sidebar.markdown("---")
        st.sidebar.markdown("**Pixel count distribution:**")
        st.sidebar.markdown(f"""
| Stat | Value |
|------|-------|
| Min  | `{int(arr.min())}` |
| Max  | `{int(arr.max())}` |
| Mean | `{int(arr.mean())}` |
| Median | `{int(np.median(arr))}` |
        """)
        st.sidebar.caption("Counts below threshold → 🟢 Free · Counts above → 🔴 Occupied")

    # ── Analyze ──
    annotated_rgb, free, occupied, total = analyze_parking(img_bgr, threshold)
    pct_free = int((free / total) * 100) if total else 0

    # ── Stats bar ──
    st.markdown(f"""
    <div class="stats-bar">
        <div class="stat-item"><div class="stat-value stat-free">{free}</div><div class="stat-label">Free Spaces</div></div>
        <div class="stat-item"><div class="stat-value stat-occ">{occupied}</div><div class="stat-label">Occupied</div></div>
        <div class="stat-item"><div class="stat-value stat-total">{total}</div><div class="stat-label">Total Spots</div></div>
        <div class="stat-item"><div class="stat-value" style="color:#f0c040">{pct_free}%</div><div class="stat-label">Availability</div></div>
    </div>
    """, unsafe_allow_html=True)

    # ── Side-by-side images ──
    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.markdown("<div class='img-label'>Original Upload</div>", unsafe_allow_html=True)
        st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)
    with col2:
        st.markdown("<div class='img-label'>Annotated Analysis</div>", unsafe_allow_html=True)
        st.image(annotated_rgb, use_container_width=True)

    # ── Download ──
    buf = io.BytesIO()
    Image.fromarray(annotated_rgb).save(buf, format="PNG")
    buf.seek(0)
    st.download_button("⬇ Download Annotated Image", data=buf,
                       file_name="parking_analysis.png", mime="image/png")

else:
    st.markdown("""
    <div style="text-align:center; padding: 4rem 2rem; color: #333; font-family: 'Space Mono', monospace; font-size: 0.8rem; letter-spacing: 1px;">
        ↑ &nbsp; UPLOAD A FRAME FROM YOUR PARKING LOT VIDEO ABOVE<br><br>
        Supports PNG · JPG · WEBP
    </div>
    """, unsafe_allow_html=True)