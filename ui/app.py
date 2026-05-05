import sys
from pathlib import Path

# Streamlit adds the script's directory (ui/) to sys.path, which shadows the
# top-level package facts_extraction.
# Remove ui/ and ensure the project root comes first instead.
_ui_path = str(Path(__file__).parent)
if _ui_path in sys.path:
    sys.path.remove(_ui_path)
_project_root = str(Path(__file__).parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

import streamlit as st
from api.mistral_api import MistralAPI
from ui import facts_extraction

st.set_page_config(page_title="ChaTP - Plongée", layout="wide")
st.title("ChaTP - Assistant plongée")


@st.cache_resource
def load_api():
    return MistralAPI()


api = load_api()

facts_extraction.render(api)
