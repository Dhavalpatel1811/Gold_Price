# -*- coding: utf-8 -*-
"""
Gradio web UI for the investment advisor chatbot.

Run from the rag_chatbot/ folder:
    python app.py
Then open http://localhost:7860
"""
import sys

import gradio as gr

from chatbot import get_response

CUSTOM_CSS = """
#chatbot {
    border-radius: 12px;
    border: 1px solid #e0e0e0;
}
#chatbot .message.user {
    background-color: #007bff !important;
    color: white !important;
    border-radius: 18px !important;
    padding: 10px 15px !important;
    margin: 5px 0 !important;
}
#chatbot .message.bot {
    background-color: #007bff !important;
    color: #202124 !important;
    border-radius: 18px !important;
    padding: 10px 15px !important;
    margin: 5px 0 !important;
}
#input_box {
    border-radius: 24px;
    border: 1px solid #dadce0;
    padding: 10px 20px;
}
.button-row {
    margin-top: 10px;
}
footer {
    display: none !important;
}
#exit_btn {
    background-color: #dc3545 !important;
}
"""


def chat_interface(message, history):
    if not message.strip():
        return history, ""
    response = get_response(message)
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": response})
    return history, ""


def exit_app():
    print("\nShutting down chatbot...")
    sys.exit(0)


with gr.Blocks() as demo:
    gr.Markdown(
        """
        # Investment Advisor Chatbot
        ### Powered by RAG + Ollama LLaMA3 | The Intelligent Investor + Live Predictions
        """
    )

    chatbot = gr.Chatbot(
        label="Chat", height=500, elem_id="chatbot"
    )

    with gr.Row():
        msg_input = gr.Textbox(
            placeholder="Ask me anything about investing, predictions, or financial theory...",
            show_label=False, scale=9, elem_id="input_box",
        )
        send_btn = gr.Button("Send", variant="primary", scale=1)

    with gr.Row(elem_classes="button-row"):
        clear_btn = gr.Button("Clear Chat", variant="secondary", size="sm")
        exit_btn = gr.Button("Exit", variant="stop", size="sm", elem_id="exit_btn")

    gr.Markdown(
        """
        **Tips:**
        - Ask theory questions: *"What is value investing?"*
        - Get predictions: *"What's the gold price forecast?"*
        - Investment advice: *"Should I invest in gold?"*
        """
    )

    send_btn.click(fn=chat_interface, inputs=[msg_input, chatbot], outputs=[chatbot, msg_input])
    msg_input.submit(fn=chat_interface, inputs=[msg_input, chatbot], outputs=[chatbot, msg_input])
    clear_btn.click(fn=lambda: ([], None), outputs=[chatbot, msg_input])
    exit_btn.click(fn=exit_app, inputs=None, outputs=None)

if __name__ == "__main__":
    print("Launching Investment Advisor Chatbot...")
    print("Open in browser: http://localhost:7860")
    demo.launch(css=CUSTOM_CSS, theme=gr.themes.Soft())
