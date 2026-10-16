"""
Gradio chat front-end for the forecasting assistant (Track B, step 4 -
stretch goal). Wraps the existing LangChain agent (agent_forecast.py) - no
new agent logic here, just a real UI on top of what's already built and
verified.

The agent itself is built once, at module load, and shared across every
visitor - but each browser session gets its own unique conversation
thread_id (via gr.State, generated once per session), so multiple people
using the public deployment at once never see each other's conversation
history.

The page opens in Gradio's normal look; a small button switches an optional warm look on and off
(the colours, fonts, header and footer live in theme.py).
"""

from __future__ import annotations

import os
import uuid

import gradio as gr

from src.assistant.agent_forecast import _extract_text, build_agent, invoke_agent
from src.assistant.query_forecast import MODEL_INTRO_MESSAGE
from src.assistant.theme import CSS, HEAD, HEADER_HTML, INTRO_HTML, PRIVACY_HTML, RESTORE_JS, TOGGLE_JS

agent = build_agent()


def chat_fn(message: str, history: list, thread_id: str) -> str:
    result = invoke_agent(agent, message, thread_id)
    return _extract_text(result["messages"][-1].content)


with gr.Blocks(title="Hospitality Forecasting Assistant") as demo:
    with gr.Row(elem_id="style-bar"):
        style_toggle = gr.Button("Switch page style", size="sm", scale=0, min_width=170, elem_id="style-toggle")
    gr.HTML(HEADER_HTML, elem_classes="warm-only")
    with gr.Column(elem_id="main"):
        gr.HTML(INTRO_HTML, elem_classes="warm-only")
        with gr.Column(elem_id="chat-card"):
            gr.ChatInterface(
                fn=chat_fn,
                additional_inputs=[gr.State(lambda: str(uuid.uuid4()))],
                # A real, static greeting (not LLM-generated) so every new conversation
                # opens with the trained model's actual evaluated accuracy - see
                # MODEL_INTRO_MESSAGE's own comment for why this is a fixed string
                # shared with the system prompt rather than something asked of the LLM.
                chatbot=gr.Chatbot(value=[{"role": "assistant", "content": MODEL_INTRO_MESSAGE}]),
                title="Hospitality Forecasting Assistant",
                description=(
                    "Ask about the sales forecast in plain language - e.g. \"what's the forecast for next Saturday?\", "
                    "\"is anything special about this weekend?\", \"what if it rains?\", or \"how accurate was last "
                    "week's forecast?\". Answers are grounded in a real, live forecasting API - never invented."
                    "\n\n**Privacy:** please don't type personal data. Your messages are sent to Google's Gemini API to "
                    "write the answers, and this app keeps them only in the running chat's memory. The hosting service "
                    "(Google Cloud Run) keeps standard request logs, which can include IP addresses. See \"Data and "
                    "privacy\" in the project README."
                ),
                examples=[
                    ["What's the forecast for next Saturday?"],
                    ["What's the forecast for next week, and is anything special about those days?"],
                    ["What if it rains this weekend?"],
                    ["How accurate was our forecast for last month?"],
                ],
            )
    gr.HTML(PRIVACY_HTML, elem_classes="warm-only")
    # Both run in the browser: the button flips the look, the load event restores the visitor's last choice.
    style_toggle.click(fn=None, js=TOGGLE_JS)
    demo.load(fn=None, js=RESTORE_JS)

if __name__ == "__main__":
    # 0.0.0.0 works fine for local use too (still reachable via localhost),
    # and is required in a container - Cloud Run injects the real PORT to
    # listen on via this env var, defaulting to 7860 for local runs.
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)), css=CSS, head=HEAD)
