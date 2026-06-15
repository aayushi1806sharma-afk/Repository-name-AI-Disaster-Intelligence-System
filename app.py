import streamlit as st
import torch
import numpy as np
import cv2
import hashlib
import pandas as pd
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
from datetime import datetime
import sys
import os
import io

from weather_service import get_weather
from disaster_feed import get_disaster_alerts
from risk_engine import filter_nearby_alerts

# PDF generation
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, KeepTogether
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from model import DisasterCNN

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------
CLASS_NAMES = ['fire', 'flood', 'landslide', 'normal', 'smoke']
IMG_SIZE    = 128
MODEL_PATH  = "models/best_model.pth"
SAMPLE_DIR  = "sample_images"
device      = torch.device("cpu")

CLASS_INFO = {
    "fire":      {"emoji": "🔥", "color": "#E53E3E", "hex_dark": "#742a2a",
                  "message": "Fire detected. Immediate evacuation and fire department alert recommended."},
    "flood":     {"emoji": "🌊", "color": "#3182CE", "hex_dark": "#2a4365",
                  "message": "Flood detected. Move to higher ground and avoid flooded roads."},
    "landslide": {"emoji": "⛰️", "color": "#B7791F", "hex_dark": "#5a3e1b",
                  "message": "Landslide detected. Avoid slopes and evacuate the affected area."},
    "normal":    {"emoji": "✅", "color": "#276749", "hex_dark": "#1c4532",
                  "message": "No disaster detected. Area appears safe."},
    "smoke":     {"emoji": "💨", "color": "#4A5568", "hex_dark": "#2d3748",
                  "message": "Smoke detected. Check for nearby fire sources and ensure ventilation."},
}

RL_COLORS = {
    "fire":      colors.HexColor("#E53E3E"),
    "flood":     colors.HexColor("#3182CE"),
    "landslide": colors.HexColor("#B7791F"),
    "normal":    colors.HexColor("#276749"),
    "smoke":     colors.HexColor("#4A5568"),
}

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(
    page_title="AI Disaster Intelligence System",
    page_icon="🚨",
    layout="wide"
)

st.markdown("""
    <style>
    .header-banner {
        padding: 1.4rem 2rem; border-radius: 14px;
        margin-bottom: 1.5rem; border: 1px solid rgba(128,128,128,0.2);
        display: flex; align-items: center; gap: 16px;
    }
    .header-banner h1 { margin: 0; font-size: 1.8rem; }
    .header-banner p  { margin: 0.3rem 0 0 0; font-size: 0.9rem; opacity: 0.7; }
    .status-pill {
        display: inline-flex; align-items: center; gap: 6px;
        background: rgba(72,187,120,0.12); border: 1px solid rgba(72,187,120,0.3);
        border-radius: 20px; padding: 4px 12px;
        font-size: 12px; color: #38a169; margin-left: auto;
    }
    .alert-box {
        padding: 1rem 1.4rem; border-radius: 10px;
        font-size: 1rem; font-weight: 600;
        margin: 1rem 0; border: 1px solid;
        display: flex; align-items: flex-start; gap: 12px;
    }
    .alert-msg { font-size: 0.85rem; font-weight: 400; opacity: 0.85; margin-top: 3px; }
    .section-label {
        font-size: 11px; text-transform: uppercase;
        letter-spacing: 0.1em; opacity: 0.5;
        margin-bottom: 10px; margin-top: 6px;
    }
    .sample-caption {
        text-align: center; font-size: 0.82rem;
        font-weight: 600; margin: 0.3rem 0; opacity: 0.75;
    }
    .metric-row { display: flex; gap: 12px; margin-bottom: 18px; }
    .metric-card {
        flex: 1; border: 1px solid rgba(128,128,128,0.2);
        border-radius: 10px; padding: 14px 16px;
    }
    .metric-label { font-size: 11px; text-transform: uppercase; letter-spacing: 0.08em; opacity: 0.5; margin-bottom: 4px; }
    .metric-val   { font-size: 24px; font-weight: 600; }
    .metric-sub   { font-size: 11px; opacity: 0.5; margin-top: 2px; }

    /* Environmental risk section */
    .env-section {
        border: 1px solid rgba(128,128,128,0.2);
        border-radius: 12px; padding: 20px 24px; margin-top: 16px;
    }
    .env-title { font-size: 1rem; font-weight: 700; margin-bottom: 4px; }
    .env-sub   { font-size: 12px; opacity: 0.6; margin-bottom: 16px; }
    .env-badge {
        display: inline-block; font-size: 11px; font-weight: 600;
        padding: 3px 10px; border-radius: 20px;
        border: 1px solid; margin-right: 6px;
    }
    .combined-note {
        border: 1px dashed rgba(128,128,128,0.35);
        border-radius: 8px; padding: 12px 16px;
        font-size: 12px; opacity: 0.75; margin-top: 14px;
        line-height: 1.6;
    }

    /* Batch report */
    .sev-chip {
        display: inline-block; font-size: 11px; font-weight: 600;
        padding: 3px 10px; border-radius: 20px;
        border: 1px solid; margin-right: 6px; margin-bottom: 6px;
    }
    .dist-bar-row { display: flex; align-items: center; gap: 10px; margin-bottom: 8px; }
    .dist-dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }
    .dist-name { font-size: 12px; flex: 1; }
    .dist-bar-bg { width: 100px; height: 5px; background: rgba(128,128,128,0.15); border-radius: 3px; }
    .dist-bar-fill { height: 100%; border-radius: 3px; }
    .dist-count { font-size: 12px; opacity: 0.6; width: 30px; text-align: right; }
    </style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# THEME HELPER — matplotlib text color
# ---------------------------------------------------------
def get_text_color():
    """Return black or white based on Streamlit's active theme."""
    try:
        theme = st.get_option("theme.base") or "light"
    except Exception:
        theme = "light"
    return "#FFFFFF" if theme == "dark" else "#1A202C"


# ---------------------------------------------------------
# MODEL + GRAD-CAM
# ---------------------------------------------------------
@st.cache_resource
def load_model():
    m = DisasterCNN(num_classes=5, freeze_backbone=True)
    m.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    m.eval()
    return m


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients   = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, _, __, out):
        self.activations = out.detach()

    def _save_gradient(self, _, __, grad_out):
        self.gradients = grad_out[0].detach()

    def generate(self, input_tensor, class_idx):
        out = self.model(input_tensor)
        self.model.zero_grad()
        out[0, class_idx].backward()
        w   = self.gradients[0].mean(dim=(1, 2))
        act = self.activations[0]
        cam = torch.zeros(act.shape[1:], dtype=torch.float32)
        for i, wi in enumerate(w):
            cam += wi * act[i]
        cam = torch.relu(cam)
        cam -= cam.min()
        cam /= (cam.max() + 1e-8)
        return cam.numpy()


@st.cache_resource
def get_gradcam(_model):
    return GradCAM(_model, _model.backbone.features[-1])


model    = load_model()
grad_cam = get_gradcam(model)

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])
])


# ---------------------------------------------------------
# PREDICTION + GRADCAM
# ---------------------------------------------------------
def predict_image(image):
    t = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        out  = model(t)
        prob = torch.softmax(out, dim=1)
        conf, idx = torch.max(prob, 1)
    return CLASS_NAMES[idx.item()], conf.item() * 100, prob.cpu().numpy()[0], idx.item()


def generate_gradcam_overlay(image, class_idx):
    t = transform(image).unsqueeze(0).to(device)
    t.requires_grad_(True)
    cam         = grad_cam.generate(t, class_idx)
    cam_resized = cv2.resize(cam, image.size)
    heatmap     = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap     = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    img_np      = np.array(image).astype(np.uint8)
    return cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)


# ---------------------------------------------------------
# PLOTS  (theme-aware text color)
# ---------------------------------------------------------
def plot_probabilities(all_probs):
    tc  = get_text_color()
    fig, ax = plt.subplots(figsize=(5, 2.6))
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("none")
    bar_colors = [CLASS_INFO[c]["color"] for c in CLASS_NAMES]
    bars = ax.barh(CLASS_NAMES, all_probs * 100, color=bar_colors, height=0.55)
    ax.set_xlim(0, 105)
    ax.set_xlabel("Confidence (%)", fontsize=9, color=tc)
    ax.tick_params(labelsize=9, colors=tc)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for sp in ["left", "bottom"]:
        ax.spines[sp].set_color(tc)
        ax.spines[sp].set_alpha(0.3)
    for bar, prob in zip(bars, all_probs):
        ax.text(prob * 100 + 1.5, bar.get_y() + bar.get_height() / 2,
                f"{prob * 100:.1f}%", va="center", fontsize=9, color=tc)
    fig.tight_layout()
    return fig


def plot_class_distribution(counts):
    tc     = get_text_color()
    labels = [c for c in CLASS_NAMES if counts[c] > 0]
    sizes  = [counts[c] for c in labels]
    pie_colors = [CLASS_INFO[c]["color"] for c in labels]
    fig, ax = plt.subplots(figsize=(4, 4))
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("none")
    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=[f"{CLASS_INFO[c]['emoji']} {c.capitalize()}" for c in labels],
        autopct="%1.0f%%",
        colors=pie_colors,
        textprops={"fontsize": 10, "color": tc},
        wedgeprops={"linewidth": 1.5, "edgecolor": "white"},
        pctdistance=0.78
    )
    for at in autotexts:
        at.set_fontweight("bold")
        at.set_fontsize(10)
        at.set_color(tc)
    fig.tight_layout()
    return fig


def plot_env_gauge(env_score):
    """
    Semi-circular gauge showing ONLY the environmental risk score
    (weather + seismic) — completely independent of any image upload.
    """
    tc  = get_text_color()
    fig, ax = plt.subplots(figsize=(5, 3))
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("none")

    zones = [(0, 30, "#48BB78"), (30, 60, "#DD6B20"), (60, 100, "#E53E3E")]
    for start, end, col in zones:
        a1    = np.pi * (1 - start / 100)
        a2    = np.pi * (1 - end   / 100)
        theta = np.linspace(a1, a2, 50)
        ax.plot(np.cos(theta), np.sin(theta), color=col,
                linewidth=22, solid_capstyle="butt")

    needle = np.pi * (1 - env_score / 100)
    ax.plot([0, 0.75 * np.cos(needle)], [0, 0.75 * np.sin(needle)],
            color=tc, linewidth=3, solid_capstyle="round")
    ax.scatter([0], [0], color=tc, s=60, zorder=5)

    ax.text(0, -0.35, f"{env_score:.0f}", ha="center",
            fontsize=26, fontweight="bold", color=tc)
    ax.text(0, -0.55, "Environmental Risk (0–100)", ha="center",
            fontsize=9, alpha=0.6, color=tc)
    ax.text(-1.05, -0.05, "Low",  fontsize=9, alpha=0.6, color=tc)
    ax.text( 1.05, -0.05, "High", ha="right", fontsize=9, alpha=0.6, color=tc)

    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-0.7, 1.2)
    ax.axis("off")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------
# ENVIRONMENTAL RISK  (weather + seismic only, NO image)
# ---------------------------------------------------------
def calculate_env_risk(weather, alerts):
    """
    Pure area risk based solely on weather + nearby seismic activity.
    No image prediction involved.

    Returns a score 0–100.
    """
    score = 0.0

    if weather:
        try:
            humidity  = weather["main"]["humidity"]
            wind      = weather["wind"]["speed"]
            condition = weather["weather"][0]["main"].lower()
            temp      = weather["main"]["temp"]
        except (KeyError, IndexError):
            humidity = wind = temp = None
            condition = ""

        # Heat stress
        if temp is not None and temp > 38:
            score += 12
        elif temp is not None and temp > 32:
            score += 6

        # Rain / storm increases flood & landslide baseline risk
        if "thunderstorm" in condition or "storm" in condition:
            score += 20
        elif "rain" in condition or "drizzle" in condition:
            score += 14

        # Low humidity → dry / fire-prone environment
        if humidity is not None and humidity < 25:
            score += 12
        elif humidity is not None and humidity < 40:
            score += 6

        # High wind
        if wind is not None and wind > 12:
            score += 10
        elif wind is not None and wind > 7:
            score += 5

    # Seismic activity near the city
    if alerts:
        significant = [a for a in alerts if (a.get("magnitude") or 0) >= 5.0]
        close_major = [a for a in significant if a.get("distance_km", 9999) < 200]
        if close_major:
            score += 25
        elif significant:
            score += 15
        else:
            score += 5

    return round(min(score, 100), 1)


def env_risk_label(score):
    if score >= 60:
        return "🔴 HIGH", "#E53E3E"
    elif score >= 30:
        return "🟠 MODERATE", "#DD6B20"
    else:
        return "🟢 LOW", "#276749"


def combined_estimate(predicted_class, confidence, env_score):
    """
    A blended 'if this photo is from the selected city' note.
    60% image confidence contribution + 40% env risk.
    Returned as a 0-100 number plus a plain-English sentence.
    """
    img_contrib = confidence * 0.6 if predicted_class != "normal" else (100 - confidence) * 0.15
    env_contrib = env_score  * 0.4
    blended = round(min(img_contrib + env_contrib, 100), 1)

    if blended >= 70:
        verdict = "very high concern"
    elif blended >= 45:
        verdict = "moderate-to-high concern"
    elif blended >= 25:
        verdict = "low-to-moderate concern"
    else:
        verdict = "low concern"

    return blended, verdict


def get_weather_recommendations(weather, alerts, env_score):
    """
    Recommendations based purely on environmental conditions —
    no assumption about what disaster type is in the image.
    """
    recs = []
    label, _ = env_risk_label(env_score)

    if env_score >= 60:
        recs.append("🔴 **High environmental risk** — notify local disaster management authorities as a precaution.")
    elif env_score >= 30:
        recs.append("🟠 **Moderate environmental risk** — increase monitoring frequency for this area.")
    else:
        recs.append("🟢 **Low environmental risk** — conditions appear stable. Routine surveillance is sufficient.")

    if weather:
        try:
            humidity  = weather["main"]["humidity"]
            wind      = weather["wind"]["speed"]
            condition = weather["weather"][0]["main"].lower()
            temp      = weather["main"]["temp"]
        except (KeyError, IndexError):
            humidity = wind = temp = None
            condition = ""

        if temp is not None and temp > 38:
            recs.append("🌡️ Extreme heat detected — risk of heat stress and dry-condition fire ignition is elevated.")
        if humidity is not None and humidity < 30:
            recs.append("💧 Very low humidity — vegetation and structures are more susceptible to fire spread.")
        if wind is not None and wind > 10:
            recs.append("💨 Strong winds — could accelerate fire spread or worsen smoke dispersion.")
        if "thunderstorm" in condition or "storm" in condition:
            recs.append("⛈️ Active thunderstorm — elevated flood and landslide risk. Avoid low-lying and hilly areas.")
        elif "rain" in condition or "drizzle" in condition:
            recs.append("🌧️ Rainfall detected — monitor drainage channels; slope areas may be at higher landslide risk.")

    if alerts:
        close = [a for a in alerts if a.get("distance_km", 9999) < 200 and (a.get("magnitude") or 0) >= 5.0]
        if close:
            recs.append(f"🌐 **Major earthquake within 200 km** (M{close[0]['magnitude']}, {close[0]['distance_km']} km away) — check for structural damage and aftershock risk.")

    recs.append("📋 Always cross-verify AI predictions with on-ground reports before dispatching resources.")
    return recs


# ---------------------------------------------------------
# CACHED API CALLS
# ---------------------------------------------------------
@st.cache_data(ttl=600)
def cached_weather(city):
    return get_weather(city)

@st.cache_data(ttl=600)
def cached_alerts():
    return get_disaster_alerts()


# ---------------------------------------------------------
# PDF REPORT
# ---------------------------------------------------------
def generate_pdf_report(results, counts, total, disaster_pct, risk_label):
    buffer = io.BytesIO()
    doc    = SimpleDocTemplate(
        buffer, pagesize=A4,
        rightMargin=2*cm, leftMargin=2*cm,
        topMargin=2*cm, bottomMargin=2*cm,
        title="AI Disaster Intelligence Report",
        author="AI Disaster Intelligence System"
    )
    styles = getSampleStyleSheet()
    W      = A4[0] - 4*cm

    title_sty = ParagraphStyle("T",  parent=styles["Title"],  fontSize=20,
                                textColor=colors.HexColor("#1a202c"), spaceAfter=3)
    sub_sty   = ParagraphStyle("S",  parent=styles["Normal"], fontSize=9,
                                textColor=colors.HexColor("#718096"))
    sec_sty   = ParagraphStyle("Sec",parent=styles["Normal"], fontSize=8,
                                fontName="Helvetica-Bold",
                                textColor=colors.HexColor("#718096"),
                                spaceBefore=12, spaceAfter=5)
    body_sty  = ParagraphStyle("B",  parent=styles["Normal"], fontSize=10,
                                textColor=colors.HexColor("#2d3748"), leading=15, spaceAfter=5)
    sm_sty    = ParagraphStyle("Sm", parent=styles["Normal"], fontSize=9,
                                textColor=colors.HexColor("#2d3748"), leading=13)
    foot_sty  = ParagraphStyle("F",  parent=styles["Normal"], fontSize=7.5,
                                textColor=colors.HexColor("#a0aec0"), alignment=TA_CENTER)

    now      = datetime.now().strftime("%d %B %Y  ·  %H:%M")
    avg_conf = sum(r["confidence"] for r in results) / len(results)
    disaster_classes = [c for c in CLASS_NAMES if c != "normal" and counts[c] > 0]
    d_count  = total - counts["normal"]

    story = []

    # Header
    hdr = Table([[
        Paragraph("🚨  AI Disaster Intelligence System", title_sty),
        Paragraph(f"Generated: {now}", sub_sty)
    ]], colWidths=[W*0.65, W*0.35])
    hdr.setStyle(TableStyle([
        ("VALIGN",    (0,0),(-1,-1),"BOTTOM"),
        ("ALIGN",     (1,0),(1,0), "RIGHT"),
        ("LINEBELOW", (0,0),(-1,0), 2, colors.HexColor("#667EEA")),
        ("BOTTOMPADDING",(0,0),(-1,0),10),
    ]))
    story += [hdr, Spacer(1, 0.35*cm)]

    # Meta
    meta = Table([
        ["Model",               "MobileNetV2 (Transfer Learning)"],
        ["Classes",             "Fire · Flood · Landslide · Normal · Smoke"],
        ["Validation Accuracy", "~89%"],
        ["Images Analyzed",     str(total)],
        ["Avg Model Confidence",f"{avg_conf:.1f}%"],
    ], colWidths=[W*0.3, W*0.7])
    meta.setStyle(TableStyle([
        ("FONTNAME",(0,0),(0,-1),"Helvetica-Bold"),
        ("FONTSIZE",(0,0),(-1,-1),8.5),
        ("TEXTCOLOR",(0,0),(0,-1),colors.HexColor("#4a5568")),
        ("TEXTCOLOR",(1,0),(1,-1),colors.HexColor("#2d3748")),
        ("ROWBACKGROUNDS",(0,0),(-1,-1),[colors.HexColor("#f7fafc"),colors.HexColor("#edf2f7")]),
        ("GRID",(0,0),(-1,-1),0.25,colors.HexColor("#e2e8f0")),
        ("LEFTPADDING",(0,0),(-1,-1),8),
        ("TOPPADDING",(0,0),(-1,-1),4),
        ("BOTTOMPADDING",(0,0),(-1,-1),4),
    ]))
    story += [meta, Spacer(1, 0.4*cm)]

    # Risk banner
    if disaster_pct >= 50:
        rb, rf, rbo = colors.HexColor("#FED7D7"), colors.HexColor("#C53030"), colors.HexColor("#FC8181")
    elif disaster_pct >= 20:
        rb, rf, rbo = colors.HexColor("#FEEBC8"), colors.HexColor("#C05621"), colors.HexColor("#F6AD55")
    else:
        rb, rf, rbo = colors.HexColor("#C6F6D5"), colors.HexColor("#276749"), colors.HexColor("#68D391")

    risk_clean = risk_label.replace("🔴 ","").replace("🟠 ","").replace("🟢 ","")
    banner = Table([[
        Paragraph(f"<b>AREA RISK ASSESSMENT: {risk_clean}</b>",
                  ParagraphStyle("RB", parent=styles["Normal"], fontSize=12,
                                 textColor=rf, fontName="Helvetica-Bold")),
        Paragraph(f"{disaster_pct:.0f}% disaster frames  ({d_count} of {total})",
                  ParagraphStyle("RS", parent=styles["Normal"], fontSize=9.5, textColor=rf,
                                 alignment=TA_RIGHT)),
    ]], colWidths=[W*0.6, W*0.4])
    banner.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,-1),rb),
        ("LINEABOVE", (0,0),(-1,0),2,rbo),("LINEBELOW",(0,0),(-1,0),2,rbo),
        ("LEFTPADDING",(0,0),(-1,-1),14),("RIGHTPADDING",(0,0),(-1,-1),14),
        ("TOPPADDING",(0,0),(-1,-1),10),("BOTTOMPADDING",(0,0),(-1,-1),10),
        ("VALIGN",(0,0),(-1,-1),"MIDDLE"),("ALIGN",(1,0),(1,0),"RIGHT"),
    ]))
    story += [banner, Spacer(1, 0.4*cm)]

    # Executive Summary
    story.append(Paragraph("EXECUTIVE SUMMARY", sec_sty))
    action_txt = (
        "Immediate multi-agency emergency response is recommended." if disaster_pct >= 50
        else "Targeted monitoring and precautionary measures are recommended." if disaster_pct >= 20
        else "Routine surveillance is sufficient at this time."
    )
    dis_types = ", ".join(c.capitalize() for c in disaster_classes) if disaster_classes else "None"
    story.append(Paragraph(
        f"Analysis of <b>{total} images</b> identified active disaster conditions in "
        f"<b>{d_count} frames ({disaster_pct:.0f}% of total)</b>. "
        f"Detected disaster types: <b>{dis_types}</b>. "
        f"Average model confidence: <b>{avg_conf:.1f}%</b>. {action_txt}", body_sty))

    # Metrics
    story.append(Spacer(1, 0.25*cm))
    mt = Table([
        ["Images Analyzed","Disaster Frames","Safe Frames","Disaster Rate"],
        [str(total), str(d_count), str(counts["normal"]), f"{disaster_pct:.0f}%"],
    ], colWidths=[W/4]*4)
    mt.setStyle(TableStyle([
        ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),("FONTSIZE",(0,0),(-1,0),8),
        ("TEXTCOLOR",(0,0),(-1,0),colors.HexColor("#718096")),
        ("ALIGN",(0,0),(-1,-1),"CENTER"),
        ("FONTSIZE",(0,1),(-1,1),17),("FONTNAME",(0,1),(-1,1),"Helvetica-Bold"),
        ("TEXTCOLOR",(0,1),(0,1),colors.HexColor("#2d3748")),
        ("TEXTCOLOR",(1,1),(1,1),colors.HexColor("#C53030")),
        ("TEXTCOLOR",(2,1),(2,1),colors.HexColor("#276749")),
        ("TEXTCOLOR",(3,1),(3,1),rf),
        ("GRID",(0,0),(-1,-1),0.5,colors.HexColor("#e2e8f0")),
        ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#EDF2F7")),
        ("TOPPADDING",(0,0),(-1,-1),7),("BOTTOMPADDING",(0,0),(-1,-1),7),
    ]))
    story += [mt, Spacer(1, 0.4*cm)]

    # Class Distribution
    story.append(Paragraph("CLASS DISTRIBUTION", sec_sty))
    dh = [Paragraph(f"<b>{h}</b>", sm_sty) for h in ["Class","Count","% of Total","Avg Confidence"]]
    drows = [dh]
    for cls in CLASS_NAMES:
        if counts[cls] == 0:
            continue
        cr = [r for r in results if r["prediction"] == cls]
        avg_c = sum(r["confidence"] for r in cr) / len(cr)
        drows.append([
            Paragraph(f"<b>{cls.capitalize()}</b>",
                      ParagraphStyle("cn", parent=sm_sty, textColor=RL_COLORS[cls],
                                     fontName="Helvetica-Bold")),
            Paragraph(str(counts[cls]), sm_sty),
            Paragraph(f"{counts[cls]/total*100:.1f}%", sm_sty),
            Paragraph(f"{avg_c:.1f}%", sm_sty),
        ])
    dt = Table(drows, colWidths=[W*0.25, W*0.2, W*0.25, W*0.3])
    dt.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#EDF2F7")),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white, colors.HexColor("#F7FAFC")]),
        ("GRID",(0,0),(-1,-1),0.25,colors.HexColor("#e2e8f0")),
        ("LEFTPADDING",(0,0),(-1,-1),8),("RIGHTPADDING",(0,0),(-1,-1),8),
        ("TOPPADDING",(0,0),(-1,-1),5),("BOTTOMPADDING",(0,0),(-1,-1),5),
    ]))
    story += [dt, Spacer(1, 0.4*cm)]

    # Key Findings
    if disaster_classes:
        story.append(Paragraph("KEY FINDINGS", sec_sty))
        fmsg = {
            "fire":      "Active fire presence across multiple frames. Structural and thermal signatures are highly distinct. Immediate suppression required.",
            "flood":     "Water accumulation and inundation detected. Low-lying routes likely compromised. Monitor drainage channels.",
            "landslide": "Slope instability and debris displacement detected. Collapse risk is elevated. Restrict hillside access.",
            "smoke":     "Smoke dispersion pattern identified, likely correlated with fire zones. Ensure ventilation in enclosed spaces.",
        }
        frows = []
        for cls in disaster_classes:
            cr    = [r for r in results if r["prediction"] == cls]
            avg_c = sum(r["confidence"] for r in cr) / len(cr)
            frows.append([
                Paragraph(f"<b>{cls.upper()}</b><br/>{counts[cls]} frame(s) · {avg_c:.1f}% avg",
                          ParagraphStyle("ft", parent=sm_sty, textColor=RL_COLORS[cls],
                                         fontName="Helvetica-Bold")),
                Paragraph(fmsg.get(cls, f"{cls.capitalize()} detected."), sm_sty),
            ])
        ft = Table(frows, colWidths=[W*0.28, W*0.72])
        ft.setStyle(TableStyle([
            ("ROWBACKGROUNDS",(0,0),(-1,-1),[colors.HexColor("#F7FAFC"), colors.white]),
            ("GRID",(0,0),(-1,-1),0.25,colors.HexColor("#e2e8f0")),
            ("LEFTPADDING",(0,0),(-1,-1),9),("RIGHTPADDING",(0,0),(-1,-1),9),
            ("TOPPADDING",(0,0),(-1,-1),7),("BOTTOMPADDING",(0,0),(-1,-1),7),
            ("VALIGN",(0,0),(-1,-1),"TOP"),
        ]))
        story += [ft, Spacer(1, 0.4*cm)]

    # Recommended Actions
    story.append(Paragraph("RECOMMENDED ACTIONS", sec_sty))
    amap = {
        "fire":      "Dispatch fire response teams immediately. Evacuate all civilians within the affected radius.",
        "flood":     "Issue flood advisory. Restrict access to low-lying roads. Deploy water rescue if needed.",
        "landslide": "Close hillside routes. Evacuate vulnerable zones. Deploy geotechnical survey team.",
        "smoke":     "Ventilate enclosed structures. Investigate smoke source. Monitor air quality index.",
    }
    arows = []
    for i, cls in enumerate(disaster_classes):
        arows.append([
            Paragraph(f"<b>{i+1}</b>", ParagraphStyle("n", parent=sm_sty,
                                                        textColor=RL_COLORS[cls],
                                                        fontName="Helvetica-Bold")),
            Paragraph(f"<b>[{cls.upper()}]</b>  {amap.get(cls,'')}", sm_sty),
        ])
    arows.append([
        Paragraph(f"<b>{len(arows)+1}</b>", sm_sty),
        Paragraph("<b>[GENERAL]</b>  Share this report with local emergency management agencies. "
                  "Perform manual verification of all flagged frames before large-scale deployment.", sm_sty),
    ])
    at = Table(arows, colWidths=[W*0.06, W*0.94])
    at.setStyle(TableStyle([
        ("VALIGN",(0,0),(-1,-1),"TOP"),
        ("TOPPADDING",(0,0),(-1,-1),6),("BOTTOMPADDING",(0,0),(-1,-1),6),
        ("LINEBELOW",(0,0),(-1,-2),0.25,colors.HexColor("#e2e8f0")),
    ]))
    story += [at, Spacer(1, 0.4*cm)]

    # Per-Image Results
    story.append(Paragraph("PER-IMAGE RESULTS", sec_sty))
    rh = [Paragraph(f"<b>{h}</b>", sm_sty)
          for h in ["#","Filename","Prediction","Confidence","Status"]]
    rrows = [rh]
    for i, r in enumerate(results):
        cls   = r["prediction"]
        s_col = colors.HexColor("#276749") if cls == "normal" else colors.HexColor("#C53030")
        s_txt = "SAFE" if cls == "normal" else "DISASTER"
        rrows.append([
            Paragraph(str(i+1), sm_sty),
            Paragraph(r["filename"][:38] + ("…" if len(r["filename"]) > 38 else ""), sm_sty),
            Paragraph(f"<b>{cls.capitalize()}</b>",
                      ParagraphStyle("pc", parent=sm_sty, textColor=RL_COLORS[cls],
                                     fontName="Helvetica-Bold")),
            Paragraph(f"{r['confidence']:.1f}%", sm_sty),
            Paragraph(f"<b>{s_txt}</b>",
                      ParagraphStyle("st", parent=sm_sty, textColor=s_col,
                                     fontName="Helvetica-Bold")),
        ])
    rt = Table(rrows, colWidths=[W*0.05, W*0.36, W*0.19, W*0.2, W*0.2])
    rt.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#EDF2F7")),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white, colors.HexColor("#F7FAFC")]),
        ("GRID",(0,0),(-1,-1),0.25,colors.HexColor("#e2e8f0")),
        ("LEFTPADDING",(0,0),(-1,-1),6),("RIGHTPADDING",(0,0),(-1,-1),6),
        ("TOPPADDING",(0,0),(-1,-1),5),("BOTTOMPADDING",(0,0),(-1,-1),5),
    ]))
    story += [rt, Spacer(1, 0.4*cm)]

    # Footer
    story += [
        Spacer(1, 0.8*cm),
        HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#e2e8f0")),
        Spacer(1, 0.2*cm),
        Paragraph(
            f"AI Disaster Intelligence System  ·  MobileNetV2 Transfer Learning  ·  "
            f"Report generated {now}  ·  For official emergency use only.",
            foot_sty)
    ]

    doc.build(story)
    buffer.seek(0)
    return buffer.read()


# ---------------------------------------------------------
# SESSION STATE
# ---------------------------------------------------------
for key, val in [("history", []), ("current_image", None),
                 ("last_hash", None), ("last_prediction", None)]:
    if key not in st.session_state:
        st.session_state[key] = val


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### 🛰️ Project Info")
    st.markdown(
        "**AI-Powered Disaster Intelligence System**\n\n"
        "CNN-based system that classifies disaster imagery "
        "and generates area-level risk assessments using "
        "transfer learning (MobileNetV2)."
    )
    st.markdown("---")
    st.markdown("### 📊 Model Details")
    st.markdown("**Architecture:** MobileNetV2")
    st.markdown("**Classes:** Fire · Flood · Landslide · Normal · Smoke")
    st.markdown("**Validation Accuracy:** ~89%")
    st.markdown("**Input Size:** 128 × 128")
    st.markdown("---")
    st.markdown("### 🕘 Session History")
    if st.session_state.history:
        for cls, conf in reversed(st.session_state.history[-6:]):
            st.markdown(f"{CLASS_INFO[cls]['emoji']} **{cls.capitalize()}** — `{conf:.1f}%`")
    else:
        st.caption("No predictions yet.")


# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------
st.markdown("""
    <div class="header-banner">
        <div style="font-size:2rem">🚨</div>
        <div>
            <h1>AI-Powered Disaster Intelligence System</h1>
            <p>Computer vision · MobileNetV2 transfer learning · Grad-CAM explainability · Area risk assessment</p>
        </div>
        <div class="status-pill">⬤ Model Online</div>
    </div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# TABS
# ---------------------------------------------------------
tab1, tab2 = st.tabs(["🔍  Single Image Analysis", "📊  Area Risk Dashboard"])


# ===========================================================
# TAB 1 — SINGLE IMAGE ANALYSIS
# ===========================================================
with tab1:

    # ── Sample images ──
    st.markdown('<div class="section-label">Quick test — sample images</div>', unsafe_allow_html=True)
    sample_cols = st.columns(5)
    for i, cls in enumerate(CLASS_NAMES):
        sample_path = os.path.join(SAMPLE_DIR, f"{cls}.jpg")
        with sample_cols[i]:
            with st.container(border=True):
                if os.path.exists(sample_path):
                    st.image(sample_path, use_container_width=True)
                    st.markdown(
                        f'<p class="sample-caption">{CLASS_INFO[cls]["emoji"]} {cls.capitalize()}</p>',
                        unsafe_allow_html=True)
                    if st.button("Test this", key=f"sample_{cls}", use_container_width=True):
                        st.session_state.current_image = Image.open(sample_path).convert("RGB")
                else:
                    st.caption(f"({cls} sample missing)")

    st.markdown("---")
    uploaded_file = st.file_uploader("Or upload your own image…",
                                     type=["jpg","jpeg","png"], key="single_upload")
    if uploaded_file is not None:
        st.session_state.current_image = Image.open(uploaded_file).convert("RGB")

    image = st.session_state.current_image

    # ── Prediction results ──
    if image is not None:
        img_hash = hashlib.md5(image.tobytes()).hexdigest()
        if img_hash != st.session_state.last_hash:
            predicted_class, confidence, all_probs, class_idx = predict_image(image)
            st.session_state.last_hash       = img_hash
            st.session_state.last_prediction = (predicted_class, confidence, all_probs, class_idx)
            st.session_state.history.append((predicted_class, confidence))
        else:
            predicted_class, confidence, all_probs, class_idx = st.session_state.last_prediction

        info = CLASS_INFO[predicted_class]

        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Input Image", width=380)
        with col2:
            with st.spinner("Generating Grad-CAM…"):
                overlay = generate_gradcam_overlay(image, class_idx)
            st.image(overlay, caption="Grad-CAM — where the model focused", width=380)

        # Image disaster confidence alert (unchanged)
        st.markdown(
            f'<div class="alert-box" style="background:{info["hex_dark"]}33;'
            f'border-color:{info["color"]}66;color:{info["color"]}">'
            f'<span style="font-size:1.5rem">{info["emoji"]}</span>'
            f'<div><div style="font-size:1rem;font-weight:700">'
            f'{predicted_class.upper()} &nbsp;·&nbsp; {confidence:.1f}% confidence</div>'
            f'<div class="alert-msg">{info["message"]}</div></div></div>',
            unsafe_allow_html=True)

        st.markdown('<div class="section-label">Class probability distribution</div>',
                    unsafe_allow_html=True)
        st.pyplot(plot_probabilities(all_probs))

    else:
        st.info("👆 Upload an image or click **Test this** on a sample above to begin analysis.")
        # Still show environmental risk even without an image
        predicted_class = None
        confidence      = None

    # ══════════════════════════════════════════════════════
    # AREA ENVIRONMENTAL RISK  (independent of image)
    # ══════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown("### 🌍 Area Environmental Risk")
    st.caption(
        "This section assesses **real-time environmental conditions** for any location you enter — "
        "completely independent of the image above. It uses live weather data and recent seismic "
        "activity to give you an area-level baseline risk."
    )

    city    = st.text_input("Enter a city / location name", value="Moradabad", key="city_input")
    weather = cached_weather(city) if city.strip() else None

    if weather:
        all_alerts   = cached_alerts()
        lat, lon     = weather["coord"]["lat"], weather["coord"]["lon"]
        nearby_alerts = filter_nearby_alerts(all_alerts, lat, lon, radius_km=500)
        env_score    = calculate_env_risk(weather, nearby_alerts)
        env_label, env_color = env_risk_label(env_score)

        # Weather metric cards
        temp        = weather["main"]["temp"]
        feels_like  = weather["main"]["feels_like"]
        humidity    = weather["main"]["humidity"]
        wind_speed  = weather["wind"]["speed"]
        condition   = weather["weather"][0]["main"]
        description = weather["weather"][0]["description"].title()

        wc1, wc2, wc3, wc4 = st.columns(4)
        with wc1:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-label">Temperature</div>
                <div class="metric-val">{temp:.1f}°C</div>
                <div class="metric-sub">Feels like {feels_like:.1f}°C</div>
            </div>""", unsafe_allow_html=True)
        with wc2:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-label">Humidity</div>
                <div class="metric-val">{humidity}%</div>
                <div class="metric-sub">{description}</div>
            </div>""", unsafe_allow_html=True)
        with wc3:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-label">Wind Speed</div>
                <div class="metric-val">{wind_speed} m/s</div>
                <div class="metric-sub">{condition}</div>
            </div>""", unsafe_allow_html=True)
        with wc4:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-label">Env. Risk Score</div>
                <div class="metric-val" style="color:{env_color}">{env_score}/100</div>
                <div class="metric-sub">weather + seismic only</div>
            </div>""", unsafe_allow_html=True)

        # Gauge
        st.markdown('<div class="section-label">Environmental Risk Meter</div>',
                    unsafe_allow_html=True)
        gcol, _ = st.columns([1, 1])
        with gcol:
            st.pyplot(plot_env_gauge(env_score))

        # Seismic alerts
        st.markdown(
            '<div class="section-label">Live Seismic Activity — within 500 km (USGS, last 24h, M 4.5+)</div>',
            unsafe_allow_html=True)
        if nearby_alerts:
            for a in nearby_alerts[:5]:
                st.markdown(
                    f"🌐 **M{a['magnitude']}** — {a['place']} · `{a['distance_km']} km away`")
        else:
            st.caption("No significant seismic activity reported near this location in the last 24 hours.")

        # Recommendations
        st.markdown('<div class="section-label">Recommendations for this area</div>',
                    unsafe_allow_html=True)
        for rec in get_weather_recommendations(weather, nearby_alerts, env_score):
            st.markdown(f"- {rec}")

        # ── Combined note (only if image has been analysed) ──
        if image is not None and predicted_class is not None:
            blended, verdict = combined_estimate(predicted_class, confidence, env_score)
            st.markdown(
                f'<div class="combined-note">'
                f'<strong>ℹ️ Combined estimate (informational only)</strong><br>'
                f'<em>If the image above is from <strong>{city}</strong></em>, '
                f'then combining the model\'s {predicted_class} detection '
                f'({confidence:.1f}% confidence) with the current environmental '
                f'risk for that area ({env_score}/100) gives a blended estimate of '
                f'<strong>{blended}/100</strong> — indicating <strong>{verdict}</strong>.<br>'
                f'<span style="opacity:0.6;font-size:11px">'
                f'This is an approximate cross-reference, not a precise measurement. '
                f'Always verify with ground truth before taking action.</span>'
                f'</div>',
                unsafe_allow_html=True)

    elif city.strip():
        st.warning(
            "⚠️ Could not fetch weather data. Check the city name, your internet connection, "
            "or the API key in `weather_service.py`.")


# ===========================================================
# TAB 2 — AREA RISK DASHBOARD (BATCH)
# ===========================================================
with tab2:
    st.markdown("#### 📡 Batch Area-Risk Assessment")
    st.write(
        "Upload a batch of images (drone footage, CCTV frames, satellite imagery) "
        "to get an automated risk summary and downloadable disaster report.")

    batch_files = st.file_uploader("Upload multiple images",
                                   type=["jpg","jpeg","png"],
                                   accept_multiple_files=True,
                                   key="batch_upload")

    if batch_files:
        results  = []
        progress = st.progress(0, text="Analyzing images…")
        for i, f in enumerate(batch_files):
            img        = Image.open(f).convert("RGB")
            pred_class, conf, _, _ = predict_image(img)
            results.append({"filename": f.name, "image": img,
                             "prediction": pred_class, "confidence": conf})
            progress.progress((i+1)/len(batch_files),
                              text=f"Analyzing images… ({i+1}/{len(batch_files)})")
        progress.empty()

        counts = {cls: 0 for cls in CLASS_NAMES}
        for r in results:
            counts[r["prediction"]] += 1

        total         = len(results)
        disaster_count = total - counts["normal"]
        disaster_pct  = (disaster_count / total) * 100

        if disaster_pct >= 50:
            risk_label, risk_color = "🔴 HIGH ALERT ZONE", "#E53E3E"
        elif disaster_pct >= 20:
            risk_label, risk_color = "🟠 MODERATE RISK ZONE", "#DD6B20"
        else:
            risk_label, risk_color = "🟢 LOW RISK / SAFE ZONE", "#276749"

        # Metric cards
        st.markdown(f"""<div class="metric-row">
            <div class="metric-card">
                <div class="metric-label">Images Analyzed</div>
                <div class="metric-val">{total}</div>
                <div class="metric-sub">batch scan</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Disaster Frames</div>
                <div class="metric-val" style="color:#E53E3E">{disaster_count}</div>
                <div class="metric-sub">require action</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Safe Frames</div>
                <div class="metric-val" style="color:#276749">{counts['normal']}</div>
                <div class="metric-sub">no disaster detected</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Disaster Rate</div>
                <div class="metric-val" style="color:{risk_color}">{disaster_pct:.0f}%</div>
                <div class="metric-sub">of total frames</div>
            </div>
        </div>""", unsafe_allow_html=True)

        risk_bg = "#FFF5F5" if disaster_pct >= 50 else "#FFFAF0" if disaster_pct >= 20 else "#F0FFF4"
        st.markdown(
            f'<div class="alert-box" style="background:{risk_bg};'
            f'border-color:{risk_color}88;color:{risk_color};font-size:1.1rem">'
            f'<span style="font-size:1.6rem">'
            f'{"🔴" if disaster_pct>=50 else "🟠" if disaster_pct>=20 else "🟢"}</span>'
            f'<div><div>{risk_label}</div>'
            f'<div class="alert-msg">{disaster_count} of {total} images show disaster signs</div>'
            f'</div></div>',
            unsafe_allow_html=True)

        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown("##### Class Distribution")
            st.pyplot(plot_class_distribution(counts))
        with col2:
            st.markdown("##### Per-Class Breakdown")
            for cls in CLASS_NAMES:
                if counts[cls] > 0:
                    inf = CLASS_INFO[cls]
                    st.markdown(
                        f"{inf['emoji']} **{cls.capitalize()}** — {counts[cls]} image(s) "
                        f"&nbsp; `{counts[cls]/total*100:.0f}%`")

        st.markdown("##### Image Grid")
        grid_cols = st.columns(4)
        for i, r in enumerate(results):
            with grid_cols[i % 4]:
                st.image(r["image"], use_container_width=True)
                inf = CLASS_INFO[r["prediction"]]
                st.markdown(
                    f'<p class="sample-caption">{inf["emoji"]} '
                    f'{r["prediction"].capitalize()} ({r["confidence"]:.0f}%)</p>',
                    unsafe_allow_html=True)

        # PDF Download
        st.markdown("---")
        st.markdown("### 📥 Download Disaster Report")
        col_dl, col_info = st.columns([1, 2])
        with col_dl:
            with st.spinner("Preparing PDF…"):
                pdf_bytes = generate_pdf_report(
                    results, counts, total, disaster_pct, risk_label)
            fname = f"disaster_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            st.download_button(
                label="⬇️  Download PDF Report",
                data=pdf_bytes,
                file_name=fname,
                mime="application/pdf",
                use_container_width=True,
                type="primary"
            )
        with col_info:
            st.caption(
                "The PDF includes: executive summary · risk banner · metric table · "
                "class distribution · key findings · recommended actions · "
                "full per-image results table · model confidence notes.")
    else:
        st.info("👆 Upload multiple images to generate an area-risk report.")