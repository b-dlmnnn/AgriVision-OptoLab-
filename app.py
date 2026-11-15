import streamlit as st
import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from nucleic_acid_module import render_nucleic_acid_tab

st.set_page_config(page_title="AgriVision OptoLab | Smart Agriculture Quality & IoT", page_icon="🌱", layout="wide")

st.sidebar.header("⚙️ Opto-Lab Control Panel")
max_blemish_tol = st.sidebar.slider("Max Export Blemish Tolerance (%) [Configurable]", 1.0, 15.0, 4.5, 0.5)
min_conf_req = st.sidebar.slider("Human-in-the-Loop Confidence Gate (%)", 50, 95, 75, 5)

st.sidebar.subheader("🔦 Optical Robustness Perturbation")
illumination_shift = st.sidebar.slider("Chamber Illumination Shift (Brightness)", -40, 40, 0, 5)
sensor_noise_lvl = st.sidebar.slider("Optical Sensor Noise (Sigma)", 0, 20, 4, 2)

st.sidebar.markdown("---")
st.sidebar.info("**Academic Transparency Note:** This prototype uses physics-informed synthetic optical chamber images and simulated IoT sensor streams to validate the computer vision and grading pipeline prior to physical hardware deployment.")

st.title("🌱 AgriVision OptoLab: Opto-Mechatronic Agricultural Quality & IoT System")
st.markdown("**Automated Optical Grading, Specular Reflection Filtering & IoT Telemetry Prototype** — Designed for Smart Agriculture & Biomechatronic Research.")

def generate_produce_sample(sample_type="Fresh Grade A", seed=42, hard_boundary=False):
    rng = np.random.default_rng(seed)
    img = np.ones((240, 240, 3), dtype=np.uint8) * 245
    center, radius = (120, 120), 78
    colors_map = {
        "Fresh Grade A": (38, 42, 210),
        "Unripe / Early Harvest": (55, 175, 75),
        "Surface Blemish (Grade B)": (42, 65, 195),
        "Defective / Fungal Rot": (45, 70, 145)
    }
    base_c = list(colors_map.get(sample_type, (45, 70, 145)))
    if hard_boundary:
        base_c = [int(np.clip(c + rng.integers(-18, 18), 10, 240)) for c in base_c]
    cv2.circle(img, center, radius, tuple(base_c), -1)

    noise = rng.normal(0, 10 if hard_boundary else 7, img.shape).astype(np.int16)
    mask = np.zeros((240, 240), dtype=np.uint8)
    cv2.circle(mask, center, radius, 255, -1)
    img = np.where(mask[:, :, None] == 255, np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8), img)

    # Kontrollu LED Specular Highlight (Parlama)
    cv2.ellipse(img, (95, 92), (16, 9), -30, 0, 360, (235, 240, 252), -1)

    if sample_type == "Fresh Grade A" and hard_boundary:
        cv2.circle(img, (int(rng.integers(100, 140)), int(rng.integers(100, 140))), 5, (45, 75, 110), -1)
    elif sample_type == "Surface Blemish (Grade B)":
        n_spots = 2 if hard_boundary else 3
        for _ in range(n_spots):
            cv2.circle(img, (int(rng.integers(90, 150)), int(rng.integers(90, 150))), int(rng.integers(7, 12)), (35, 65, 95), -1)
    elif sample_type == "Defective / Fungal Rot":
        n_spots = 4 if hard_boundary else 7
        for _ in range(n_spots):
            cv2.circle(img, (int(rng.integers(80, 160)), int(rng.integers(80, 160))), int(rng.integers(10, 18)), (30, 42, 52), -1)
    return img

def apply_perturbation(bgr_img, brightness_delta=0, noise_sigma=0):
    rng = np.random.default_rng(99)
    perturbed = bgr_img.astype(np.int16) + brightness_delta
    if noise_sigma > 0:
        perturbed = perturbed + rng.normal(0, noise_sigma, bgr_img.shape).astype(np.int16)
    return np.clip(perturbed, 0, 255).astype(np.uint8)

def extract_optical_features(bgr_img):
    resized = cv2.resize(bgr_img, (240, 240))
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    # Arka plan segmentasyonu (Parlak beyaz arka plan haric)
    _, thresh = cv2.threshold(gray, 228, 255, cv2.THRESH_BINARY_INV)
    obj_mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    if np.count_nonzero(obj_mask) < 500:
        obj_mask = np.ones_like(gray) * 255

    # Specular Highlight (Parlama) tespiti ve maskeden cikarilmasi (V > 220 & S < 45)
    specular_mask = ((hsv[:, :, 2] > 220) & (hsv[:, :, 1] < 45) & (obj_mask > 0)).astype(np.uint8) * 255
    clean_mask = cv2.bitwise_and(obj_mask, cv2.bitwise_not(specular_mask))
    if np.count_nonzero(clean_mask) < 200:
        clean_mask = obj_mask

    h_vals = hsv[:, :, 0][clean_mask > 0].astype(np.float64)
    s_vals = hsv[:, :, 1][clean_mask > 0].astype(np.float64)
    v_vals = hsv[:, :, 2][clean_mask > 0].astype(np.float64)

    # Dairesel Hue Ortalamasi (OpenCV 0-179 olcegi -> 2*pi radyan)
    angles = (h_vals / 180.0) * 2.0 * np.pi
    sin_mean = np.mean(np.sin(angles))
    cos_mean = np.mean(np.cos(angles))
    circ_angle = np.arctan2(sin_mean, cos_mean)
    if circ_angle < 0:
        circ_angle += 2.0 * np.pi
    circular_hue = (circ_angle / (2.0 * np.pi)) * 180.0
    R = np.hypot(sin_mean, cos_mean)
    circular_std = np.sqrt(max(0.0, -2.0 * np.log(max(R, 1e-6)))) * (180.0 / (2.0 * np.pi))

    dark_pixels = np.sum((v_vals < 95) & (s_vals < 165))
    blemish_ratio = float(dark_pixels / (len(v_vals) + 1e-5)) * 100.0
    specular_pct = float(np.count_nonzero(specular_mask) / (np.count_nonzero(obj_mask) + 1e-5)) * 100.0

    edges = cv2.Canny(gray, 60, 150)
    edge_density = float(np.sum(edges[clean_mask > 0] > 0) / (len(v_vals) + 1e-5)) * 100.0

    return {
        "circular_hue_0_179": round(float(circular_hue), 2),
        "circular_hue_std": round(float(circular_std), 2),
        "mean_saturation": round(float(np.mean(s_vals)), 2),
        "mean_brightness": round(float(np.mean(v_vals)), 2),
        "blemish_ratio_pct": round(blemish_ratio, 2),
        "texture_edge_density": round(edge_density, 2),
    }, clean_mask, specular_mask, round(specular_pct, 2)

@st.cache_resource
def train_quality_model():
    classes = ["Fresh Grade A", "Unripe / Early Harvest", "Surface Blemish (Grade B)", "Defective / Fungal Rot"]
    rows, labels = [], []
    for idx, cls in enumerate(classes):
        for i in range(60):
            is_hard = (i % 3 == 0)  # %33 sinir/zor ornekler ile gercekci kalibrasyon
            sample = generate_produce_sample(cls, seed=idx * 200 + i, hard_boundary=is_hard)
            if is_hard:
                sample = apply_perturbation(sample, brightness_delta=(i % 5 - 2) * 8, noise_sigma=5)
            feats, _, _, _ = extract_optical_features(sample)
            rows.append(feats)
            labels.append(cls)
    df = pd.DataFrame(rows)
    X_train, X_test, y_train, y_test = train_test_split(df, labels, test_size=0.25, random_state=42, stratify=labels)
    clf = RandomForestClassifier(n_estimators=100, max_depth=6, min_samples_leaf=2, random_state=42).fit(X_train, y_train)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test) 
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    return clf, accuracy_score(y_test, y_pred), df.columns.tolist(), cm, classes
tab1, tab4, tab2, tab3 = st.tabs([
    "🔬 1. Optical Produce Quality Inspection",
    "🧬 2. Nucleic Acid Assay (LAMP/RPA)",
    "📡 3. IoT & Biological Sensor Telemetry",
    "📊 4. Model Validation & Confusion Matrix"
])

with tab1:
    col_left, col_right = st.columns([1, 1.35])
    with col_left:
        st.subheader("Image Acquisition & Robustness Test")
        input_mode = st.radio("Select Input Source:", ["Simulated Optical Chamber Sample", "Upload Custom Produce Image"], horizontal=True)
        if input_mode == "Simulated Optical Chamber Sample":
            selected_sample = st.selectbox("Select Conveyor Test Sample:", ["Fresh Grade A", "Surface Blemish (Grade B)", "Unripe / Early Harvest", "Defective / Fungal Rot"])
            sample_seed = st.slider("Sample Variation Seed", 1, 50, 12)
            raw_bgr = generate_produce_sample(selected_sample, seed=sample_seed, hard_boundary=(sample_seed % 4 == 0))
            sample_label = f"{selected_sample} (Seed {sample_seed})"
        else:
            uploaded = st.file_uploader("Upload an image (JPG/PNG)", type=["jpg", "jpeg", "png"])
            if uploaded is not None:
                raw_bgr = cv2.cvtColor(np.array(Image.open(uploaded).convert("RGB")), cv2.COLOR_RGB2BGR)
                sample_label = uploaded.name
            else:
                st.info("Showing default sample until an image is uploaded.")
                raw_bgr = generate_produce_sample("Fresh Grade A", seed=7)
                sample_label = "Default Sample (Fresh Grade A)"

        bgr_image = apply_perturbation(raw_bgr, illumination_shift, sensor_noise_lvl)
        st.image(cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB), caption=f"Chamber Capture (Illumination Shift: {illumination_shift:+d}, Noise: {sensor_noise_lvl})", use_container_width=True)

    with col_right:
        st.subheader("Automated Optical Feature & Grade Analysis")
        feats, clean_mask, spec_mask, spec_pct = extract_optical_features(bgr_image)
        input_df = pd.DataFrame([feats])[feature_names]
        pred_class = str(model.predict(input_df)[0])
        pred_probs = model.predict_proba(input_df)[0]
        confidence = float(np.max(pred_probs)) * 100.0

        low_conf_review = confidence < min_conf_req
        passes_export = ("Fresh" in pred_class) and (feats["blemish_ratio_pct"] <= max_blemish_tol) and (not low_conf_review)

        if low_conf_review:
            gate_status = "MANUAL REVIEW — Low Confidence Routing"
            st.warning(f"**Classification:** {pred_class} ({confidence:.1f}%)  |  **Decision:** {gate_status}")
        elif passes_export:
            gate_status = "PASS — Export Grade A"
            st.success(f"**Classification:** {pred_class} ({confidence:.1f}%)  |  **Decision:** {gate_status}")
        elif "Grade B" in pred_class or "Unripe" in pred_class:
            gate_status = "DIVERT — Secondary Sorting Line"
            st.warning(f"**Classification:** {pred_class} ({confidence:.1f}%)  |  **Decision:** {gate_status}")
        else:
            gate_status = "REJECT — Defective / Spoiled"
            st.error(f"**Classification:** {pred_class} ({confidence:.1f}%)  |  **Decision:** {gate_status}")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Blemish Area", f"{feats['blemish_ratio_pct']}%", f"Limit: {max_blemish_tol}%")
        m2.metric("Circular Hue", f"{feats['circular_hue_0_179']}", "Scale: 0–179")
        m3.metric("Specular Glare", f"{spec_pct}%", "Filtered Out")
        m4.metric("Edge Density", f"{feats['texture_edge_density']}%")

        v1, v2 = st.columns(2)
        v1.image(clean_mask, caption="Active Analysis Mask (Background & Glare Removed)", use_container_width=True)
        v2.image(spec_mask, caption="Filtered Specular Highlight Map (V>220, S<45)", use_container_width=True)

        prob_df = pd.DataFrame({"Quality Class": model.classes_, "Probability (%)": np.round(pred_probs * 100, 2)})
        fig_prob = px.bar(prob_df, x="Quality Class", y="Probability (%)", color="Quality Class", title="Calibrated Classifier Probability Distribution")
        fig_prob.update_layout(height=260, showlegend=False)
        st.plotly_chart(fig_prob, use_container_width=True)

        report_df = pd.DataFrame([{"Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "Sample_ID": sample_label, "Predicted_Grade": pred_class, "Confidence_Pct": round(confidence, 2), "Gate_Decision": gate_status, "Illumination_Shift": illumination_shift, "Specular_Glare_Pct": spec_pct, **feats}])
        st.download_button("📥 Download Optical Inspection Report (CSV)", report_df.to_csv(index=False).encode("utf-8"), "agrivision_inspection_report.csv", "text/csv")

with tab2:
    st.subheader("Simulated Greenhouse & Sorting Line IoT Telemetry")
    st.caption("Simulated multi-sensor stream demonstrating timestamp alignment, anomaly flags, and temperature-yield correlation.")
    rng_iot = np.random.default_rng(2027)
    hours = pd.date_range("2027-01-15 08:00", periods=48, freq="h")
    temp_c = 24.5 + np.sin(np.linspace(0, 4 * np.pi, 48)) * 2.4 + rng_iot.normal(0, 0.4, 48)
    humidity = 68.0 - np.sin(np.linspace(0, 4 * np.pi, 48)) * 5.0 + rng_iot.normal(0, 1.1, 48)
    grade_a_yield = np.clip(94.0 - np.maximum(0, temp_c - 26.0) * 4.8 + rng_iot.normal(0, 1.2, 48), 72, 99)
    iot_df = pd.DataFrame({"Timestamp": hours, "Chamber Temperature (°C)": np.round(temp_c, 2), "Relative Humidity (%)": np.round(humidity, 2), "Grade-A Optical Yield (%)": np.round(grade_a_yield, 2)})
    anomalies = iot_df[iot_df["Chamber Temperature (°C)"] > 26.5]

    k1, k2, k3 = st.columns(3)
    k1.metric("Avg Chamber Temp", f"{iot_df['Chamber Temperature (°C)'].mean():.1f} °C")
    k2.metric("Thermal Excursion Alerts", f"{len(anomalies)} events", "Threshold > 26.5°C")
    k3.metric("Mean Grade-A Yield", f"{iot_df['Grade-A Optical Yield (%)'].mean():.1f} %")

    fig_iot = go.Figure()
    fig_iot.add_trace(go.Scatter(x=iot_df["Timestamp"], y=iot_df["Grade-A Optical Yield (%)"], name="Grade-A Optical Yield (%)", line=dict(color="#10B981", width=3)))
    fig_iot.add_trace(go.Scatter(x=iot_df["Timestamp"], y=iot_df["Chamber Temperature (°C)"], name="Chamber Temp (°C)", yaxis="y2", line=dict(color="#EF4444", dash="dot")))
    fig_iot.update_layout(title="Simulated Thermal Stress vs. Optical Grade-A Yield", yaxis=dict(title="Optical Yield (%)"), yaxis2=dict(title="Temperature (°C)", overlaying="y", side="right"), height=370)
    st.plotly_chart(fig_iot, use_container_width=True)

with tab3:
    st.subheader("Opto-Mechatronic Pipeline & Validation Diagnostics")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"""
        - **Hold-out Test Accuracy:** `{model_acc * 100:.1f}%` (evaluated on noisy & boundary samples)
        - **Circular Hue Statistics:** Trigonometric mean (`arctan2`) on OpenCV `0–179` scale to prevent red-spectrum wrap-around bias.
        - **Specular Reflection Filtering:** Excludes high-brightness, low-saturation LED glare (`V > 220, S < 45`) from defect area calculation.
        - **Human-in-the-Loop Routing:** Predictions below `{min_conf_req}%` confidence are automatically flagged for manual operator inspection.
        """)
        imp_df = pd.DataFrame({"Optical Feature": feature_names, "Feature Importance": model.feature_importances_}).sort_values("Feature Importance", ascending=True)
        fig_imp = px.bar(imp_df, x="Feature Importance", y="Optical Feature", orientation="h", title="Random Forest Feature Importance")
        fig_imp.update_layout(height=300)
        st.plotly_chart(fig_imp, use_container_width=True)
    with c2:
        fig_cm = px.imshow(conf_mat, text_auto=True, x=class_labels, y=class_labels, color_continuous_scale="Greens", title="Hold-Out Validation Confusion Matrix")
        fig_cm.update_layout(height=420, xaxis_title="Predicted Grade", yaxis_title="True Grade")
        st.plotly_chart(fig_cm, use_container_width=True)
with tab4:
    render_nucleic_acid_tab()
