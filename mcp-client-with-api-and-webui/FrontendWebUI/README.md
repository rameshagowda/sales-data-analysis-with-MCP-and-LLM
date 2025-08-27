uv init
uv venv
source .venv/bin/activate

uv add streamlit requests pandas plotly python-dotenv watchdog

uv run streamlit run chatui.py
