"""MetaOpt-AI | Antenna Design Explorer - Streamlit entry point.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

import config  # noqa: E402
from ui.common import get_repo  # noqa: E402
from ui.navigation import pages  # noqa: E402
from ui.theme import inject_styles  # noqa: E402

st.set_page_config(page_title=config.APP_TITLE, page_icon=":material/settings_input_antenna:", layout="wide")
inject_styles()

try:
    get_repo()  # creates the database + synthetic dataset on first start
except Exception as exc:  # pragma: no cover - shown in the UI
    st.error(f"MetaOpt-AI could not start: {exc}. Check that the project folder is writable.")
    st.stop()

# Exactly three sections, shown as a top navigation bar that always marks the current one.
st.navigation(list(pages().values()), position="top").run()
