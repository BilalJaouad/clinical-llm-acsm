"""
app.py
------
Responsible ONLY for:
    - Streamlit interface
    - chat display
    - history (session-only, capped at MAX_HISTORY messages)
    - calling the safety filter, the RAG helper, and the LLM

See cahier des charges sections 1-4, 7, 11, 14.
"""

import streamlit as st
from dotenv import load_dotenv

import safety
import rag
from llm import query_biomistral, LLMError
from prompts import build_prompt

load_dotenv()

# Keep the history small on purpose to limit token usage (section 3 / 7).
MAX_HISTORY = 6

st.set_page_config(page_title="CardioAssist", page_icon="🫀", layout="centered")

st.title("🫀 CardioAssist")
st.caption("General cardiovascular health information — academic prototype")

st.warning(
    "⚠️ **Medical disclaimer**: CardioAssist is an academic prototype, "
    "**not a medical device**. It cannot diagnose conditions, prescribe "
    "medication, or replace a healthcare professional. In an emergency, "
    "contact your local emergency services immediately.",
    icon="⚠️",
)

with st.sidebar:
    st.header("About")
    st.write(
        "CardioAssist answers general questions about cardiovascular "
        "diseases, hypertension, diet, physical activity, risk factors, "
        "medical exams, terminology, and prevention."
    )
    st.divider()
    use_rag = st.checkbox(
        "Use reference documents (RAG)",
        value=True,
        help="When enabled, the assistant looks up a few short passages "
        "from a small local reference corpus before answering.",
    )
    st.divider()
    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()
    st.caption(f"History is limited to the last {MAX_HISTORY} messages to save tokens.")

if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Render existing conversation -------------------------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- Handle new input ---------------------------------------------------------------
user_input = st.chat_input("Ask a general cardiovascular health question...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        # 1) Safety check runs BEFORE any call to the LLM (section 4).
        result = safety.check_message(user_input)

        if result["status"] != "ok":
            reply = result["message"]
            st.markdown(reply)
        else:
            with st.spinner("Thinking..."):
                # 2) Optional minimal RAG (section 9, Phase 2).
                passages = []
                if use_rag:
                    hits = rag.get_relevant_passages(user_input, top_k=2)
                    passages = [p for _, p in hits]

                # 3) Build a short history string from the last few turns only.
                recent = st.session_state.messages[-MAX_HISTORY:]
                history_lines = []
                for m in recent[:-1]:  # exclude the message we're answering now
                    role = "User" if m["role"] == "user" else "Assistant"
                    history_lines.append(f"{role}: {m['content']}")
                history_text = "\n".join(history_lines)

                prompt = build_prompt(history_text, user_input, context_passages=passages)

                # 4) Call the LLM, with essential error handling (section 14).
                try:
                    reply = query_biomistral(prompt)
                except LLMError as exc:
                    reply = f"⚠️ {exc}"

            st.markdown(reply)

            if use_rag and passages:
                with st.expander("Sources used"):
                    for source, passage in hits:
                        st.caption(f"**{source}**")
                        st.write(passage)

    st.session_state.messages.append({"role": "assistant", "content": reply})

    # Keep only the last MAX_HISTORY messages in session state (section 3/7).
    if len(st.session_state.messages) > MAX_HISTORY:
        st.session_state.messages = st.session_state.messages[-MAX_HISTORY:]
