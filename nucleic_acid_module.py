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
            liquid_bgr, true_copies = (35, 215, 245), 10000.0
        elif condition == "Low-Copy Near LoD (100 copies/µL)":
            liquid_bgr, true_copies = (75, 175, 240), 100.0
        else:
            liquid_bgr, true_copies = (125, 65, 225), 0.0
    else:
        bg = np.full((h, w, 3), (24, 28, 34), dtype=np.uint8)
        if condition == "Positive (Target Detected)":
            liquid_bgr, true_copies = (75, 245, 65), 10000.0
        elif condition == "Low-Copy Near LoD (100 copies/µL)":
            liquid_bgr, true_copies = (55, 155, 50), 100.0
        else:
            liquid_bgr, true_copies = (35, 65, 38), 0.0

    img = bg.copy()
    cv2.rectangle(img, (x1, y_top), (x2, y_bot - radius), (185, 195, 205), 2)
    cv2.ellipse(img, (cx, y_bot - radius), (radius, radius), 0, 0, 180, (185, 195, 205), 2)
    cv2.ellipse(img, (cx, y_top), (radius, 10), 0, 0, 360, (165, 175, 185), 2)

    mask_3d = np.repeat(np.expand_dims(roi_mask, axis=2), 3, axis=2)
    color_layer = np.full((h, w, 3), liquid_bgr, dtype=np.uint8)
    img = np.where(mask_3d == 255, color_layer, img)

    glare_overlay = np.zeros_like(img)
    cv2.rectangle(glare_overlay, (x1 + 12, y_liq_top + 14), (x1 + 21, y_bot - radius - 10), (255, 255, 255), -1)
    cv2.ellipse(glare_overlay, (cx, y_liq_top + 4), (radius - 8, 5), 0, 0, 360, (245, 248, 252), -1)
    img = cv2.addWeighted(img, 1.0, glare_overlay, 0.78, 0)

    img_float = img.astype(np.float32) + float(illum_shift)
    if noise_sigma > 0:
        rng = np.random.default_rng(42)
        img_float += rng.normal(0, noise_sigma * 1.5, img_float.shape)
    return np.clip(img_float, 0, 255).astype(np.uint8), roi_mask, true_copies


def _extract_center_roi_from_upload(bgr_img: np.ndarray):
    resized = cv2.resize(bgr_img, (320, 440), interpolation=cv2.INTER_AREA)
    h, w, _ = resized.shape
    roi_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(roi_mask, (115, 140), (205, 340), 255, -1)
    return resized, roi_mask


def _analyze_tube_roi(bgr_img: np.ndarray, roi_mask: np.ndarray, assay_mode: str, conf_gate: float):
    hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2LAB)
    h_ch, s_ch, v_ch = cv2.split(hsv)
    _, a_ch, b_ch = cv2.split(lab)
    _, green_plane, _ = cv2.split(bgr_img)

    glare_in_roi = ((v_ch > 220) & (s_ch < 45) & (roi_mask == 255)).astype(np.uint8) * 255
    valid_liquid_mask = (roi_mask == 255) & (glare_in_roi == 0)
    total_roi_px = max(int(np.sum(roi_mask == 255)), 1)
    glare_pct = (int(np.sum(glare_in_roi == 255)) / total_roi_px) * 100.0
    if np.sum(valid_liquid_mask) < 20:
        valid_liquid_mask = roi_mask == 255

    angles_rad = np.deg2rad(np.extract(valid_liquid_mask, h_ch).astype(np.float32) * 2.0)
    circular_hue_deg = (np.rad2deg(np.arctan2(float(np.mean(np.sin(angles_rad))), float(np.mean(np.cos(angles_rad))))) + 360.0) % 360.0
    cielab_a = float(np.mean(np.extract(valid_liquid_mask, a_ch).astype(np.float32) - 128.0))
    cielab_b = float(np.mean(np.extract(valid_liquid_mask, b_ch).astype(np.float32) - 128.0))
    g_channel_au = float(np.mean(np.extract(valid_liquid_mask, green_plane).astype(np.float32)))

    if assay_mode == "Colorimetric LAMP (Phenol Red)":
        prob_pos = 1.0 / (1.0 + np.exp(-np.clip((cielab_b - 28.0) / 8.5, -20, 20)))
        est_copies = 0.0 if prob_pos < 0.5 else float(np.clip(10 ** (1.5 + (cielab_b - 20.0) / 22.0), 10.0, 1e6))
    else:
        prob_pos = 1.0 / (1.0 + np.exp(-np.clip((g_channel_au - 115.0) / 22.0, -20, 20)))
        est_copies = 0.0 if prob_pos < 0.5 else float(np.clip(10 ** (1.5 + (g_channel_au - 95.0) / 55.0), 10.0, 1e6))

    confidence_pct = float(max(prob_pos, 1.0 - prob_pos) * 100.0)
    if confidence_pct < conf_gate:
        decision, status_color, bg_tint = "INCONCLUSIVE (Below Confidence Gate)", "#b45309", "rgba(245, 158, 11, 0.14)"
    elif prob_pos >= 0.5:
        decision, status_color, bg_tint = "POSITIVE (Target Signal Detected)", "#dc2626", "rgba(220, 38, 38, 0.14)"
    else:
        decision, status_color, bg_tint = "NEGATIVE / NTC (No Amplification)", "#15803d", "rgba(21, 128, 61, 0.14)"

    return {
        "circular_hue_deg": circular_hue_deg, "cielab_a": cielab_a, "cielab_b": cielab_b,
        "glare_pct": glare_pct, "g_channel_au": g_channel_au, "prob_pos": prob_pos,
        "confidence_pct": confidence_pct, "est_copies": est_copies, "decision": decision,
        "status_color": status_color, "bg_tint": bg_tint, "roi_mask": roi_mask, "glare_mask": glare_in_roi,
    }
