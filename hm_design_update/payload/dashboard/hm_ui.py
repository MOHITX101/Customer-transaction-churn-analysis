"""Shared presentation only. No model loading or prediction logic."""
from pathlib import Path
from html import escape
import base64
import mimetypes

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
LOGO = ROOT / "images/H&M-Logo.svg.webp"
BRAND_PANEL = ROOT / "images/images.jpg"
CSS_FILE = Path(__file__).with_name("hm_theme.css")
RED = "#C7002C"
INK = "#171717"
PALETTE = [RED, INK, "#A8794F", "#77716C", "#E6C9CE", "#BCB9B3"]


@st.cache_data(max_entries=8)
def _image_uri(path, modified):
    file = Path(path)
    mime = mimetypes.guess_type(file.name)[0] or "image/webp"
    return f"data:{mime};base64," + base64.b64encode(file.read_bytes()).decode("ascii")


def image_uri(file):
    return _image_uri(str(file), file.stat().st_mtime_ns)


def apply_design():
    if CSS_FILE.is_file():
        st.html("<style>" + CSS_FILE.read_text(encoding="utf-8") + "</style>")
    if LOGO.is_file():
        st.logo(str(LOGO), size="large", icon_image=str(LOGO))

    pio.templates["retail_editorial"] = go.layout.Template(
        layout=go.Layout(
            font=dict(family="Arial, sans-serif", color=INK, size=13),
            title=dict(font=dict(size=18, color=INK)),
            colorway=PALETTE,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=12, r=18, t=46, b=34),
            hoverlabel=dict(bgcolor="white", font_color=INK),
            xaxis=dict(showgrid=False, zeroline=False, automargin=True),
            yaxis=dict(gridcolor="#ECE9E5", zeroline=False, automargin=True),
            legend=dict(orientation="h", y=-0.2, x=0, title_text=""),
        )
    )
    pio.templates.default = "plotly_white+retail_editorial"


def render_header(page_title):
    st.html(f"""
    <div class="hm-topline">
      <div><span class="hm-wordmark">RETAIL INTELLIGENCE</span>
        <span class="hm-breadcrumb">/ {escape(page_title)}</span></div>
      <span class="hm-project-tag">INDEPENDENT ACADEMIC PROJECT</span>
    </div>
    """)


def render_hero():
    if BRAND_PANEL.is_file():
        panel = f'<img src="{image_uri(BRAND_PANEL)}" alt="H&amp;M logo on a red background" />'
    else:
        panel = '<div class="hm-brand-fallback">H&amp;M<br><small>DATASET PROJECT</small></div>'
    st.html(f"""
    <section class="hm-hero">
      <div class="hm-hero-copy">
        <span class="hm-eyebrow">THE CUSTOMER &amp; COMMERCE EDIT</span>
        <h1>Retail intelligence.<br><em>In focus.</em></h1>
        <p>Understand the purchase. Anticipate the next one.<br>
        Explore sales, customer behaviour and predictive insights.</p>
        <div class="hm-hero-chips"><span>HISTORICAL DATA · 2018–2020</span>
          <span>ANALYTICS + MACHINE LEARNING</span></div>
      </div>
      <div class="hm-hero-art">{panel}
        <span class="hm-art-caption">H&amp;M DATASET / INDEPENDENT STUDY</span>
      </div>
    </section>
    """)


def section_label(number, label):
    st.html(f'<div class="hm-section-label"><span>{escape(number)}</span>{escape(label)}</div>')


def render_footer():
    st.html("""
    <footer class="hm-footer">
      <strong>RETAIL INTELLIGENCE</strong>
      <span>Independent academic project using the H&amp;M dataset.<br>
      Not an official H&amp;M service. Historical predictions; spending in dataset units.</span>
    </footer>
    """)
