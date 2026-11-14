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
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="AgriVision OptoLab | Smart Agriculture Quality & IoT", page_icon="🌱", layout="wide")

st.sidebar.header("⚙️ Opto-Lab Control Panel")
max_blemish_tol = st.sidebar.slider("Max Export Blemish Tolerance (%)", 1.0, 15.0, 4.5, 0.5)
min_conf_req = st.sidebar.slider("Minimum Classifier Confidence (%)", 50, 95, 75, 5)
st.sidebar.markdown("---")
st.sidebar.caption("**System Pipeline:**\n1. HSV Optical Color Space\n2. Morphological Segmentation\n3. Random Forest Grading\n4. IoT Telemetry Sync")

st.title("🌱 AgriVision OptoLab: Opto-Mechatronic Agricultural Quality & IoT System")
st.markdown("**Automated Optical Grading, Biological Monitoring & Sensor Telemetry Prototype** — Designed for Smart Agriculture & Biomechatronic Quality Inspection Workflows.")

def generate_produce_sample(sample_type="Fresh Grade A", seed=42):
    rng = np.random.default_rng(seed)
    img = np.ones((240, 240, 3), dtype=np.uint8) * 245
    center, radius = (120, 120), 78
    colors_map = {
        "Fresh Grade A": (38, 42, 210),
        "Unripe / Early Harvest": (55, 175, 75),
        "Surface Blemish (Grade B)": (42, 65, 195),
        "Defective / Fungal Rot": (45, 70, 145)
    }
    cv2.circle(img, center, radius, colors_map.get(sample_type, (45, 70, 145)), -1)
    noise = rng.normal(0, 8, img.shape).astype(np.int16)
    mask = np.zeros((240, 240), dtype=np.uint8)
    cv2.circle(mask, center, radius, 255, -1)
    img = np.where(mask[:, :, None] == 255, np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8), img)
    cv2.ellipse(img, (95, 92), (16, 9), -30, 0, 360, (180, 190, 240), -1)
    if sample_type == "Surface Blemish (Grade B)":
        for _ in range(3):
            cv2.circle(img, (int(rng.integers(90, 150)), int(rng.integers(90, 150))), int(rng.integers(7, 12)), (35, 65, 95), -1)
    elif sample_type == "Defective / Fungal Rot":
        for _ in range(7):
            cv2.circle(img, (int(rng.integers(80, 160)), int(rng.integers(80, 160))), int(rng.integers(10, 18)), (30, 42, 52), -1)
    return img

def extract_optical_features(bgr_img):
    resized = cv2.resize(bgr_img, (240, 240))
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY_INV)
    mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    if np.count_nonzero(mask) < 500:
        mask = np.ones_like(gray) * 255
    h_vals, s_vals, v_vals = hsv[:, :, 0][mask > 0], hsv[:, :, 1][mask > 0], hsv[:, :, 2][mask > 0]
    dark_pixels = np.sum((v_vals < 95) & (s_vals < 160))
    blemish_ratio = float(dark_pixels / (len(v_vals) + 1e-5)) * 100.0
    edges = cv2.Canny(gray, 60, 150)
    edge_density = float(np.sum(edges[mask > 0] > 0) / (len(v_vals) + 1e-5)) * 100.0
    return {
        "mean_hue": round(float(np.mean(h_vals)), 2),
        "std_hue": round(float(np.std(h_vals)), 2),
        "mean_saturation": round(float(np.mean(s_vals)), 2),
        "mean_brightness": round(float(np.mean(v_vals)), 2),
        "blemish_ratio_pct": round(blemish_ratio, 2),
        "texture_edge_density": round(edge_density, 2),
    }, mask, edges

@st.cache_resource
def train_quality_model():
    classes = ["Fresh Grade A", "Unripe / Early Harvest", "Surface Blemish (Grade B)", "Defective / Fungal Rot"]
    rows, labels = [], []
    for idx, cls in enumerate(classes):
        for i in range(45):
            feats, _, _ = extract_optical_features(generate_produce_sample(cls, seed=idx * 100 + i))
            rows.append(feats)
            labels.append(cls)
    df = pd.DataFrame(rows)
    X_train, X_test, y_train, y_test = train_test_split(df, labels, test_size=0.25, random_state=42, stratify=labels)
    clf = RandomForestClassifier(n_estimators=120, random_state=42).fit(X_train, y_train)
    return clf, accuracy_score(y_test, clf.predict(X_test)), df.columns.tolist()

model, model_acc, feature_names = train_quality_model()
tab1, tab2, tab3 = st.tabs(["🔬 1. Optical Produce Quality Inspection", "📡 2. IoT & Biological Sensor Telemetry", "📊 3. Model Architecture & Validation"])

with tab1:
    col_left, col_right = st.columns([1, 1.3])
    with col_left:
        st.subheader("Image Acquisition Source")
        input_mode = st.radio("Select Input Method:", ["Use Simulated Conveyor Sample", "Upload Custom Produce Image"], horizontal=True)
        if input_mode == "Use Simulated Conveyor Sample":
            selected_sample = st.selectbox("Select Conveyor Test Sample:", ["Fresh Grade A", "Surface Blemish (Grade B)", "Unripe / Early Harvest", "Defective / Fungal Rot"])
            sample_seed = st.slider("Sample Variation Seed", 1, 50, 12)
            bgr_image = generate_produce_sample(selected_sample, seed=sample_seed)
            sample_label = f"{selected_sample} (Seed {sample_seed})"
        else:
            uploaded = st.file_uploader("Upload an image (JPG/PNG)", type=["jpg", "jpeg", "png"])
            if uploaded is not None:
                bgr_image = cv2.cvtColor(np.array(Image.open(uploaded).convert("RGB")), cv2.COLOR_RGB2BGR)
                sample_label = uploaded.name
            else:
                st.info("Showing default sample until an image is uploaded.")
                bgr_image = generate_produce_sample("Fresh Grade A", seed=7)
                sample_label = "Default Sample (Fresh Grade A)"
        st.image(cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB), caption="Optical Chamber Capture (RGB)", use_container_width=True)

    with col_right:
        st.subheader("Automated Optical Feature & Grade Analysis")
        feats, seg_mask, edge_map = extract_optical_features(bgr_image)
        input_df = pd.DataFrame([feats])[feature_names]
        pred_class = str(model.predict(input_df)[0])
        pred_probs = model.predict_proba(input_df)[0]
        confidence = float(np.max(pred_probs)) * 100.0
        passes_export = ("Fresh" in pred_class) and (feats["blemish_ratio_pct"] <= max_blemish_tol) and (confidence >= min_conf_req)
        gate_status = "PASS — Export Grade A" if passes_export else "HOLD / DIVERT — Secondary Sorting"
        if passes_export:
            st.success(f"**Classification:** {pred_class} ({confidence:.1f}%)  |  **Gate Decision:** {gate_status}")
        elif "Grade B" in pred_class or "Unripe" in pred_class or confidence < min_conf_req:
            st.warning(f"**Classification:** {pred_class} ({confidence:.1f}%)  |  **Gate Decision:** {gate_status}")
        else:
            st.error(f"**Classification:** {pred_class} ({confidence:.1f}%)  |  **Gate Decision:** {gate_status}")
        m1, m2, m3 = st.columns(3)
        m1.metric("Blemish Area Ratio", f"{feats['blemish_ratio_pct']}%", f"Limit: {max_blemish_tol}%")
        m2.metric("Mean Hue (HSV)", f"{feats['mean_hue']}°")
        m3.metric("Surface Edge Density", f"{feats['texture_edge_density']}%")
        v1, v2 = st.columns(2)
        v1.image(seg_mask, caption="Object Segmentation Mask", use_container_width=True)
        v2.image(edge_map, caption="Surface Defect & Edge Detection", use_container_width=True)
        prob_df = pd.DataFrame({"Quality Class": model.classes_, "Probability (%)": np.round(pred_probs * 100, 2)})
        fig_prob = px.bar(prob_df, x="Quality Class", y="Probability (%)", color="Quality Class", title="Classifier Confidence Distribution")
        fig_prob.update_layout(height=270, showlegend=False)
        st.plotly_chart(fig_prob, use_container_width=True)
        report_df = pd.DataFrame([{"Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "Sample_ID": sample_label, "Predicted_Grade": pred_class, "Confidence_Pct": round(confidence, 2), "Gate_Decision": gate_status, "Max_Blemish_Tolerance_Pct": max_blemish_tol, **feats}])
        st.download_button("📥 Download Optical Inspection Report (CSV)", report_df.to_csv(index=False).encode("utf-8"), "agrivision_inspection_report.csv", "text/csv")

with tab2:
    st.subheader("Real-Time Greenhouse & Sorting Line IoT Telemetry")
    rng_iot = np.random.default_rng(2027)
    hours = pd.date_range("2027-01-15 08:00", periods=48, freq="h")
    temp_c = 24.5 + np.sin(np.linspace(0, 4 * np.pi, 48)) * 2.2 + rng_iot.normal(0, 0.4, 48)
    humidity = 68.0 - np.sin(np.linspace(0, 4 * np.pi, 48)) * 5.0 + rng_iot.normal(0, 1.1, 48)
    grade_a_yield = np.clip(94.0 - np.maximum(0, temp_c - 26.0) * 4.5 + rng_iot.normal(0, 1.2, 48), 72, 99)
    iot_df = pd.DataFrame({"Timestamp": hours, "Chamber Temperature (°C)": np.round(temp_c, 2), "Relative Humidity (%)": np.round(humidity, 2), "Grade-A Optical Yield (%)": np.round(grade_a_yield, 2)})
    k1, k2, k3 = st.columns(3)
    k1.metric("Avg Chamber Temp", f"{iot_df['Chamber Temperature (°C)'].mean():.1f} °C", "-0.3 °C")
    k2.metric("Avg Relative Humidity", f"{iot_df['Relative Humidity (%)'].mean():.1f} %", "+1.2 %")
    k3.metric("Mean Grade-A Yield", f"{iot_df['Grade-A Optical Yield (%)'].mean():.1f} %", "+2.1 %")
    fig_iot = go.Figure()
    fig_iot.add_trace(go.Scatter(x=iot_df["Timestamp"], y=iot_df["Grade-A Optical Yield (%)"], name="Grade-A Optical Yield (%)", line=dict(color="#10B981", width=3)))
    fig_iot.add_trace(go.Scatter(x=iot_df["Timestamp"], y=iot_df["Chamber Temperature (°C)"], name="Chamber Temp (°C)", yaxis="y2", line=dict(color="#EF4444", dash="dot")))
    fig_iot.update_layout(title="Correlation Between Chamber Temperature & Optical Quality Yield", yaxis=dict(title="Optical Yield (%)"), yaxis2=dict(title="Temperature (°C)", overlaying="y", side="right"), height=380)
    st.plotly_chart(fig_iot, use_container_width=True)

with tab3:
    st.subheader("Opto-Mechatronic Pipeline & Model Diagnostics")
    st.markdown(f"- **Validation Accuracy (Hold-out Test Set):** `{model_acc * 100:.1f}%`\n- **Optical Color Space:** HSV (Hue-Saturation-Value) for controlled LED chamber lighting.\n- **Morphological Segmentation:** Binary inverse thresholding + morphological closing.\n- **Feature Engineering:** 6 quantitative optical & textural descriptors.")
    imp_df = pd.DataFrame({"Optical Feature": feature_names, "Feature Importance": model.feature_importances_}).sort_values("Feature Importance", ascending=True)
    fig_imp = px.bar(imp_df, x="Feature Importance", y="Optical Feature", orientation="h", title="Random Forest Feature Importance")
    fig_imp.update_layout(height=320)
    st.plotly_chart(fig_imp, use_container_width=True)
