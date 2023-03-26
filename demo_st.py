"""Simple Web Demo that calls the model via API."""

from datetime import datetime
from textwrap import wrap as _wrap

import requests
import streamlit as st

from alpaca import InferenceRequest


def wrap(text: str, width: int = 80) -> str:
    lines = []
    for line in text.split("\n"):
        lines.extend(_wrap(line, width=width))
    return "\n".join(lines)


st.set_page_config("Alpaca Demo", page_icon="🦙")


st.title("Alpaca Demo")

alpaca_api = st.text_input("alpaca_api_url", "http://localhost:8080")
input_text = st.text_input("input_text", "Are alpacas afraid of snakes?")
use_simple_endpoint = st.checkbox("use-simple-endpoint", value=True)
endpoint = "run_simple" if use_simple_endpoint else "run"

t_start = datetime.now()
with st.spinner("Running model..."):
    request = InferenceRequest(input_text=input_text)
    response = requests.post(url=f"{alpaca_api}/{endpoint}", json=request.dict())
t_end = datetime.now()
duration = t_end - t_start
if response.ok:
    output = response.json()
    st.text(wrap(output.pop("output")))
    with st.expander("details"):
        st.write(output)
else:
    st.error(f"{response.status_code}: {response.content}")
st.write(f"Execution took: {duration.total_seconds()} seconds.")
