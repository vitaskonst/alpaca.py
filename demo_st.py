import os
from datetime import datetime
from textwrap import wrap as _wrap

import streamlit as st

from alpaca import Alpaca, InferenceRequest


def wrap(text: str, width: int = 80) -> str:
    lines = []
    for line in text.split("\n"):
        lines.extend(_wrap(line, width=width))
    return "\n".join(lines)


st.set_page_config("Alpaca Demo", page_icon="🦙")


@st.cache_resource()
def load_model() -> Alpaca:
    alpaca = Alpaca(os.environ["ALPACA_CLI_PATH"], os.environ["ALPACA_MODEL_PATH"])
    return alpaca


alpaca = load_model()

st.title("Alpaca Demo")

input_text = st.text_input("input_text", "Are alpacas afraid of snakes?")

t_start = datetime.now()
with st.spinner("Running model..."):
    output = alpaca.run_simple(InferenceRequest(input_text=input_text))
t_end = datetime.now()
duration = t_end - t_start

st.text(wrap(output.pop("output")))

with st.expander("details"):
    output.update(alpaca.system_info)
    st.write(output)

st.write(f"Execution took: {duration.total_seconds()} seconds.")
