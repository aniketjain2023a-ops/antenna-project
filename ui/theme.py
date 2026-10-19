"""MetaOpt-AI design system: tokens, one small CSS layer, and reusable UI components.

Most styling lives in the native Streamlit theme (``.streamlit/config.toml``).
This module adds only what the theme cannot express, following the Apple
Design skill (``~/.claude/skills/apple-design``):

* Response (§1)      - instant press feedback on buttons (scale on :active, 100 ms ease-out).
* Materials (§12)    - the top navigation is a translucent layer (backdrop blur) that
                       content scrolls under, instead of an opaque strip with a hard divider.
* Accessibility (§14)- reduced-motion, reduced-transparency and increased-contrast fallbacks.
* Typography (§15)   - system font, size-specific tracking (tight headings, slightly open
                       small labels), inverse leading, tabular numbers for metrics.
* Foundations (§16)  - wayfinding (context bar on every page), four kinds of feedback
                       (status, completion, warning, error), simplicity (advanced options
                       one level deeper), specific labels.

Keep ML/database logic out of this file.
"""

from __future__ import annotations

import streamlit as st

import config

# Mirrors .streamlit/config.toml - used by CSS below and by Plotly figures.
TOKENS = {
    "canvas": "#F5F5F3",
    "surface": "#FFFFFF",
    "text": "#1C1C1E",
    "text_secondary": "#6B6B70",
    "border": "#E4E4E0",
    "accent": "#0A64D6",
    "radius_card": "0.75rem",
    "radius_control": "0.5rem",
}

PROVENANCE_BADGES = {
    "SYNTHETIC": ("Synthetic demo data", "orange", ":material/science:"),
    "SIMULATION": ("EM simulation data", "blue", ":material/memory:"),
    "MEASUREMENT": ("Measured data", "green", ":material/straighten:"),
    "PUBLIC": ("Public data", "gray", ":material/public:"),
}

_CSS = f"""
<style>
:root {{
  --mo-surface: {TOKENS['surface']};
  --mo-border: {TOKENS['border']};
  --mo-text-2: {TOKENS['text_secondary']};
  --mo-accent: {TOKENS['accent']};
  --mo-radius: {TOKENS['radius_card']};
}}

.st-key-mo-styles {{ display: none; }}

/* Layout: a comfortable reading width on a MacBook, content clears the floating nav */
[data-testid="stMainBlockContainer"] {{ max-width: 1240px; padding-top: 4.75rem; padding-bottom: 4rem; }}

/* Typography (skill §15): tighter tracking + leading as size grows, slightly open small text */
.stApp h1 {{ letter-spacing: -0.022em; line-height: 1.1; }}
.stApp h2 {{ letter-spacing: -0.016em; line-height: 1.2; }}
.stApp h3, .stApp h4 {{ letter-spacing: -0.01em; line-height: 1.25; }}
[data-testid="stCaptionContainer"], [data-testid="stMetricLabel"] {{ letter-spacing: 0.01em; }}
[data-testid="stMetricLabel"] p {{ color: var(--mo-text-2); font-size: 0.82rem; }}
[data-testid="stMetricValue"] {{ font-variant-numeric: tabular-nums; letter-spacing: -0.015em; }}

/* Cards (theme.card): white surfaces on the warm canvas, thin border, no heavy shadow.
   Targeted through Streamlit's documented per-key class (.st-key-<key>). */
[class*="st-key-card-"] {{ background: var(--mo-surface); border-color: var(--mo-border) !important;
  border-radius: var(--mo-radius) !important; padding: 1.1rem 1.25rem 1.2rem !important; }}
[data-testid="stExpander"] details {{ background: var(--mo-surface); }}

/* Selected chips in multiselects: tinted, not saturated (accent reserved for primary actions) */
[data-testid="stMultiSelectTagsContainer"] > span > span {{ background-color: #EAF2FC !important; color: #0A55B5 !important; }}
[data-testid="stMultiSelectTagsContainer"] > span > span svg {{ fill: #0A55B5 !important; color: #0A55B5 !important; }}

/* Response (skill §1): feedback on press, instantly */
.stButton > button, .stDownloadButton > button, [data-testid="stFormSubmitButton"] > button {{
  transition: transform 100ms ease-out, background-color 150ms ease-out, border-color 150ms ease-out;
  font-weight: 550;
}}
.stButton > button:active, .stDownloadButton > button:active,
[data-testid="stFormSubmitButton"] > button:active {{ transform: scale(0.97); }}
.stButton > button:focus-visible, .stDownloadButton > button:focus-visible,
[data-testid="stFormSubmitButton"] > button:focus-visible {{
  outline: 2px solid var(--mo-accent); outline-offset: 2px;
}}

/* Materials (skill §12): translucent navigation layer, content scrolls underneath */
header[data-testid="stHeader"] {{
  background: rgba(245, 245, 243, 0.72);
  -webkit-backdrop-filter: saturate(180%) blur(20px);
  backdrop-filter: saturate(180%) blur(20px);
}}

/* Accessibility (skill §14) */
@media (prefers-reduced-motion: reduce) {{
  .stButton > button, .stDownloadButton > button, [data-testid="stFormSubmitButton"] > button {{ transition: none; }}
  .stButton > button:active, .stDownloadButton > button:active,
  [data-testid="stFormSubmitButton"] > button:active {{ transform: none; }}
}}
@media (prefers-reduced-transparency: reduce) {{
  header[data-testid="stHeader"] {{ background: {TOKENS['canvas']}; backdrop-filter: none; -webkit-backdrop-filter: none; }}
}}
@media (prefers-contrast: more) {{
  :root {{ --mo-border: #8E8E93; --mo-text-2: #3A3A3C; }}
  [class*="st-key-card-"] {{ border-color: #8E8E93 !important; }}
}}
</style>
"""


def inject_styles() -> None:
    """Apply the CSS layer on every script run.

    Rendered through markdown (static CSS only, no scripts) inside a keyed container whose
    wrapper is hidden by the same stylesheet, so it takes no layout space. ``st.html`` sends
    style-only content to a transient container that can be cleared between reruns.
    """
    with st.container(key="mo-styles"):
        st.markdown(_CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------- components
def page_header(title: str, description: str) -> None:
    st.caption(f"**{config.PROJECT_NAME}** · Antenna Design Explorer")
    st.title(title)
    st.caption(description)


def provenance_badge(ds: dict | None) -> None:
    if ds is None:
        st.badge("No dataset", color="gray", icon=":material/block:")
        return
    label, color, icon = PROVENANCE_BADGES.get(ds["dataset_type"], (ds["dataset_type"], "gray", None))
    st.badge(label, color=color, icon=icon)


def context_bar(ds: dict | None, n_models: int | None = None, where=None) -> None:
    """Wayfinding: which dataset, what kind of data, how many samples, are models ready.

    Pass ``where`` (a placeholder created near the top of the page) to draw the bar after the
    page's actions have run, so it reflects models trained during this run.
    """
    with (where or st.container()).container(horizontal=True, gap="small", vertical_alignment="center"):
        if ds is None:
            provenance_badge(None)
            return
        st.badge(ds["name"], color="gray", icon=":material/table:")
        provenance_badge(ds)
        st.badge(f"{ds['sample_count']} samples", color="gray")
        if n_models is not None:
            if n_models:
                st.badge(f"{n_models} active model{'s' if n_models != 1 else ''}", color="green",
                         icon=":material/check_circle:")
            else:
                st.badge("No trained models", color="gray", icon=":material/radio_button_unchecked:")


def synthetic_notice(ds: dict | None) -> None:
    """One concise warning sentence; the full explanation is one level deeper."""
    if ds is None or ds["dataset_type"] != "SYNTHETIC":
        return
    st.warning("**Synthetic demonstration data.** Results show that the workflow runs; they are not "
               "EM-simulation or measurement results and say nothing about real antenna accuracy.",
               icon=":material/science:")


def card(name: str):
    """A bordered white panel. ``name`` must be unique on the page and stable across reruns."""
    return st.container(border=True, key=f"card-{name}")


def empty_state(title: str, body: str, icon: str = ":material/info:") -> None:
    with card("empty-" + "".join(ch for ch in title.lower() if ch.isalnum())[:24]):
        st.markdown(f"##### {icon} {title}")
        st.caption(body)


def section(title: str, caption: str | None = None) -> None:
    st.subheader(title)
    if caption:
        st.caption(caption)


def chart(fig, caption: str | None = None, key: str | None = None) -> None:
    """Render a Plotly figure with the project template (Streamlit theme disabled so styling is consistent)."""
    if fig is None:
        st.caption("Not enough data to draw this chart.")
        return
    st.plotly_chart(fig, theme=None, width="stretch", key=key, config={"displaylogo": False})
    if caption:
        st.caption(caption)


def unit(column: str) -> str:
    return config.unit_of(column)


def fmt_value(value: float | None, column: str, digits: int = 2) -> str:
    if value is None:
        return "-"
    u = unit(column)
    return f"{value:.{digits}f} {u}".strip()
