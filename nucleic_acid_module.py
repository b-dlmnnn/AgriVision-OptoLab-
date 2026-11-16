import streamlit as st
import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from PIL import Image
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support


def _generate_synthetic_tube(condition: str, assay_mode: str, illum_shift: int, noise_sigma: int):
    h, w = 440, 320
    bg = np.full((h, w, 3), (242, 245, 248), dtype=np.uint8)

    x1, x2 = 105, 215
    y_top, y_liq_top, y_bot = 55, 165, 365
    cx = (x1 + x2) // 2
    radius = (x2 - x1) // 2

    roi_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(roi_mask, (x1 + 5, y_liq_top + 6), (x2 - 5, y_bot - radius), 255, -1)
    cv2.circle(roi_mask, (cx, y_bot - radius), radius - 5, 255, -1)

    if assay_mode == "Colorimetric LAMP (Phenol Red)":
        if condition == "Positive (Target Detected)":
            liquid_bgr = (35, 215, 245)
            true_copies = 10000.0
        elif condition == "Low-Copy Near LoD (100 copies/µL)":
            liquid_bgr = (75, 175, 240)
            true_copies = 100.0
        else:
            liquid_bgr = (125, 65, 225)
            true_copies = 0.0
    else:
        bg = np.full((h, w, 3), (24, 28, 34), dtype=np.uint8)
        if condition == "Positive (Target Detected)":
            liquid_bgr = (75, 245, 65)
            true_copies = 10000.0
        elif condition == "Low-Copy Near LoD (100 copies/µL)":
            liquid_bgr = (55, 155, 50)
            true_copies = 100.0
        else:
            liquid_bgr = (35, 65, 38)
            true_copies = 0.0

    img = bg.copy()
    cv2.rectangle(img, (x1, y_top), (x2, y_bot - radius), (185, 195, 205), 2)
    cv2.ellipse(img, (cx, y_bot - radius), (radius, radius), 0, 0, 180, (185, 195, 205), 2)
    cv2.ellipse(img, (cx, y_top), (radius, 10), 0, 0, 360, (165, 175, 185), 2)

    img = np.where(roi_mask[:, :, None] == 255, np.array(liquid_bgr, dtype=np.uint8), img)

    glare_overlay = np.zeros_like(img)
    cv2.rectangle(glare_overlay, (x1 + 12, y_liq_top + 14), (x1 + 21, y_bot - radius - 10), (255, 255, 255), -1)
    cv2.ellipse(glare_overlay, (cx, y_liq_top + 4), (radius - 8, 5), 0, 0, 360, (245, 248, 252), -1)
    img = cv2.addWeighted(img, 1.0, glare_overlay, 0.78, 0)

    img_float = img.astype(np.float32) + float(illum_shift)
    if noise_sigma > 0:
        rng = np.random.default_rng(42)
        noise = rng.normal(0, noise_sigma * 1.5, img_float.shape)
        img_float += noise
    img_final = np.clip(img_float, 0, 255).astype(np.uint8)

    return img_final, roi_mask, true_copies


def _extract_center_roi_from_upload(bgr_img: np.ndarray):
    resized = cv2.resize(bgr_img, (320, 440), interpolation=cv2.INTER_AREA)
    h, w = resized.shape[:2]
    roi_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(roi_mask, (115, 140), (205, 340), 255, -1)
    return resized, roi_mask


def _analyze_tube_roi(bgr_img: np.ndarray, roi_mask: np.ndarray, assay_mode: str, conf_gate: float):
    hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2LAB)

    h_ch, s_ch, v_ch = cv2.split(hsv)
    _, a_ch, b_ch = cv2.split(lab)

    glare_in_roi = ((v_ch > 220) & (s_ch < 45) & (roi_mask == 255)).astype(np.uint8) * 255
    valid_liquid_mask = (roi_mask == 255) & (glare_in_roi == 0)

    total_roi_px = max(int(np.sum(roi_mask == 255)), 1)
    glare_px = int(np.sum(glare_in_roi == 255))
    glare_pct = (glare_px / total_roi_px) * 100.0

    if np.sum(valid_liquid_mask) < 20:
        valid_liquid_mask = roi_mask == 255

    angles_rad = np.deg2rad(h_ch[valid_liquid_mask].astype(np.float32) * 2.0)
    mean_sin = float(np.mean(np.sin(angles_rad)))
    mean_cos = float(np.mean(np.cos(angles_rad)))
    circular_hue_deg = (np.rad2deg(np.arctan2(mean_sin, mean_cos)) + 360.0) % 360.0

    cielab_a = float(np.mean(a_ch[valid_liquid_mask].astype(np.float32) - 128.0))
    cielab_b = float(np.mean(b_ch[valid_liquid_mask].astype(np.float32) - 128.0))

    green_plane = bgr_img[:, :, 1]
    g_channel_au = float(np.mean(green_plane[valid_liquid_mask]))

    if assay_mode == "Colorimetric LAMP (Phenol Red)":
        logit = (cielab_b - 28.0) / 8.5
        prob_pos = 1.0 / (1.0 + np.exp(-np.clip(logit, -20, 20)))
        est_copies = 0.0 if prob_pos < 0.5 else float(np.clip(10 ** (1.5 + (cielab_b - 20.0) / 22.0), 10.0, 1e6))
    else:
        logit = (g_channel_au - 115.0) / 22.0
        prob_pos = 1.0 / (1.0 + np.exp(-np.clip(logit, -20, 20)))
        est_copies = 0.0 if prob_pos < 0.5 else float(np.clip(10 ** (1.5 + (g_channel_au - 95.0) / 55.0), 10.0, 1e6))

    confidence_pct = float(max(prob_pos, 1.0 - prob_pos) * 100.0)

    if confidence_pct < conf_gate:
        decision = "INCONCLUSIVE (Below Confidence Gate — Manual Review)"
        status_color = "#b45309"
        bg_tint = "rgba(245, 158, 11, 0.14)"
    elif prob_pos >= 0.5:
        decision = "POSITIVE (Target Signal Detected)"
        status_color = "#dc2626"
        bg_tint = "rgba(220, 38, 38, 0.14)"
    else:
        decision = "NEGATIVE / NTC (No Target Amplification)"
        status_color = "#15803d"
        bg_tint = "rgba(21, 128, 61, 0.14)"

    return {
        "circular_hue_deg": circular_hue_deg,
        "cielab_a": cielab_a,
        "cielab_b": cielab_b,
        "glare_pct": glare_pct,
        "g_channel_au": g_channel_au,
        "prob_pos": prob_pos,
        "confidence_pct": confidence_pct,
        "est_copies": est_copies,
        "decision": decision,
        "status_color": status_color,
        "bg_tint": bg_tint,
        "roi_mask": roi_mask,
        "glare_mask": glare_in_roi,
    }


def render_nucleic_acid_tab(min_conf_req=75, illumination_shift=0, sensor_noise_lvl=4, *args, **kwargs):
    st.subheader("🧪 Portable Nucleic Acid Assay Analyzer (LAMP / RPA)")
    st.caption(
        "Research prototype for optical colorimetry & fluorescence extraction — not a clinical diagnostic device. "
        "Sidebar illumination, noise, and confidence gate controls apply directly to this module."
    )

    col_ctrl, col_img, col_res = st.columns((1.05, 1.0, 1.25), gap="medium")

    with col_ctrl:
        st.markdown("#### 1. Assay & Input Source")
        assay_mode = st.radio(
            "Assay Detection Mode",
            ("Colorimetric LAMP (Phenol Red)", "Fluorescent RPA (SYBR Green / FAM)"),
            key="na_assay_mode",
        )

        input_source = st.radio(
            "Image Input Source",
            ("Rule-Based Synthetic Tube", "Upload Real Tube Photo (JPG / PNG)"),
            key="na_input_source",
        )

        uploaded_tube = None
        sample_condition = "Positive (Target Detected)"
        reaction_min = 30

        if input_source == "Rule-Based Synthetic Tube":
            sample_condition = st.selectbox(
                "Control / Sample Condition",
                (
                    "Positive (Target Detected)",
                    "Low-Copy Near LoD (100 copies/µL)",
                    "Negative Control — NTC (No-Template Control)",
                ),
                key="na_sample_cond",
            )
            reaction_min = st.slider(
                "Simulated Incubation Time (min)",
                min_value=0,
                max_value=45,
                value=30,
                step=5,
                key="na_time_min",
            )
            if reaction_min < 15 and sample_condition != "Negative Control — NTC (No-Template Control)":
                st.caption("⏱ Early reaction window (<15 min): amplification color shift is still developing.")
        else:
            uploaded_tube = st.file_uploader(
                "Upload Tube / Colored Solution Image",
                type=("jpg", "jpeg", "png"),
                key="na_tube_uploader",
                help="Supports smartphone photos of assay tubes or non-biological pH/dye validation tubes.",
            )
            st.caption("Non-biological validation mode: test with colored water / pH indicator + gray card.")

    if input_source == "Upload Real Tube Photo (JPG / PNG)" and uploaded_tube is not None:
        pil_img = Image.open(uploaded_tube).convert("RGB")
        bgr_raw = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        tube_bgr, roi_mask = _extract_center_roi_from_upload(bgr_raw)
        is_real_upload = True
    else:
        effective_cond = (
            "Negative Control — NTC (No-Template Control)"
            if (input_source == "Rule-Based Synthetic Tube" and reaction_min < 15)
            else sample_condition
        )
        tube_bgr, roi_mask, _ = _generate_synthetic_tube(
            effective_cond, assay_mode, illumination_shift, sensor_noise_lvl
        )
        is_real_upload = False

    metrics = _analyze_tube_roi(tube_bgr, roi_mask, assay_mode, float(min_conf_req))

    with col_img:
        st.markdown("#### 2. Optical Tube & ROI Masks")
        rgb_display = cv2.cvtColor(tube_bgr, cv2.COLOR_BGR2RGB)
        overlay_disp = rgb_display.copy()
        ys, xs = np.where(roi_mask == 255)
        if len(xs) > 0 and len(ys) > 0:
            cv2.rectangle(overlay_disp, (int(xs.min()), int(ys.min())), (int(xs.max()), int(ys.max())), (16, 185, 129), 2)

        st.image(
            overlay_disp,
            caption="Uploaded Tube (Green = Active Liquid ROI)" if is_real_upload else "Rule-Based Synthetic Tube (Green = ROI)",
            use_container_width=True,
        )

        m_col1, m_col2 = st.columns(2)
        with m_col1:
            st.image(metrics.get("roi_mask"), caption="Liquid ROI Mask", use_container_width=True)
        with m_col2:
            st.image(metrics.get("glare_mask"), caption="ROI-Only Glare Mask", use_container_width=True)

    with col_res:
        st.markdown("#### 3. Quantitative Signal & Decision")
        st.markdown(
            f"""
