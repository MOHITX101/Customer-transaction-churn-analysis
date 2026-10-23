from pathlib import Path
import sys

import streamlit as st

DASHBOARD_DIR = Path(__file__).resolve().parent
if str(DASHBOARD_DIR) not in sys.path:
    sys.path.insert(0, str(DASHBOARD_DIR))

from hm_ui import apply_design, render_footer, render_header

st.set_page_config(
    page_title="Retail Intelligence | H&M Dataset",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)
apply_design()

PAGE_GROUPS = {
    "DASHBOARD": [
        ("overview.py", "Overview", "dashboard"),
        ("pages/1_Sales_Analysis.py", "Sales analysis", "bar_chart"),
        ("pages/2_Customer_Analysis.py", "Customer analysis", "group"),
    ],
    "PREDICTIONS": [
        ("pages/3_Customer_Predictions.py", "Customer predictions", "person_search"),
        ("pages/4_Product_Recommendations.py", "Product recommendations", "shopping_bag"),
        ("pages/5_Weekly_Forecasting.py", "Weekly forecasting", "trending_up"),
    ],
    "EXPLAIN & PLAN": [
        ("pages/6_Model_Explanations.py", "Model explanations", "insights"),
        ("pages/7_Campaign_Simulator.py", "Campaign simulator", "campaign"),
    ],
}

missing = [
    path for group in PAGE_GROUPS.values()
    for path, _, _ in group if not (DASHBOARD_DIR / path).is_file()
]
if missing:
    st.error("Some application pages are missing.")
    st.code("\n".join(missing), language=None)
    st.stop()

navigation = st.navigation(
    {
        group_name: [
            st.Page(
                path, title=title, icon=f":material/{icon}:",
                default=path == "overview.py",
            )
            for path, title, icon in pages
        ]
        for group_name, pages in PAGE_GROUPS.items()
    },
    position="sidebar",
    expanded=True,
)

with st.sidebar:
    st.html("""
    <div class="hm-sidebar-note">
      <strong>RETAIL INTELLIGENCE</strong>
      <span>Independent academic project</span>
      <span>H&amp;M historical dataset · 2018–2020</span>
    </div>
    """)

render_header(navigation.title)
navigation.run()
render_footer()
