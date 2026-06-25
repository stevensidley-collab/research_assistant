"""
Streamlit chat UI for the Research Assistant.

Reuses the tool definitions and agentic loop from research_assistant.py —
no logic is duplicated here. API keys stay server-side: research_assistant.py
loads them via python-dotenv from .env, and Streamlit never sees them directly.
"""

import os

import streamlit as st

from research_assistant import run_turn

# research_assistant.py already calls load_dotenv() on import, so APP_PASSWORD
# (if set in .env) is available here too.
APP_PASSWORD = os.environ.get("APP_PASSWORD")
MAX_MESSAGES_PER_SESSION = 20

st.set_page_config(page_title="Research Assistant", page_icon="🔎")
st.title("🔎 Research Assistant")

# --- Password gate -----------------------------------------------------
# Only enforced if APP_PASSWORD is set in .env; without it the app is open,
# which is fine for local use but not recommended for a public deployment.
if APP_PASSWORD:
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if not st.session_state.authenticated:
        entered = st.text_input("Enter password to continue", type="password")
        if entered:
            if entered == APP_PASSWORD:
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("Incorrect password.")
        st.stop()

# Conversation history lives in session_state so it persists across reruns.
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render prior turns. Tool-result messages (role="user" with list content) are
# internal plumbing for the agentic loop, not something the user typed — skip them.
for message in st.session_state.messages:
    if message["role"] == "user" and isinstance(message["content"], list):
        continue
    if message["role"] == "assistant" and isinstance(message["content"], list):
        text = "".join(b.text for b in message["content"] if hasattr(b, "text"))
        if not text:
            continue
        with st.chat_message("assistant"):
            st.markdown(text)
        continue
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Rate limit ----------------------------------------------------------
# Caps API spend per browser session. Does not protect against many distinct
# sessions (e.g. a shared link going viral) — pair with usage alerts/caps in
# the Anthropic and Tavily dashboards for that.
if "turn_count" not in st.session_state:
    st.session_state.turn_count = 0

if st.session_state.turn_count >= MAX_MESSAGES_PER_SESSION:
    st.warning(
        f"You've reached the {MAX_MESSAGES_PER_SESSION}-message limit for this "
        "session. Refresh the page to start a new session."
    )
    st.stop()

if user_input := st.chat_input("Ask a research question..."):
    with st.chat_message("user"):
        st.markdown(user_input)

    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.turn_count += 1

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            reply = run_turn(st.session_state.messages)
        st.markdown(reply)
