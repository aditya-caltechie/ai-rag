"""
RAG Chatbot Web Interface

Gradio-based UI for interacting with the RAG system.
Shows both the conversation and retrieved context side-by-side.

To switch between implementations, change RAG_MODE in config.py or set RAG_MODE environment variable.
"""

import gradio as gr
from dotenv import load_dotenv

load_dotenv(override=True)

# Import configuration to determine which implementation to use
from config import RAG_MODE

# Dynamically import the answer function based on RAG_MODE
if RAG_MODE == "basic":
    from implementation.answer import answer_question
    print("✓ Using Basic RAG (implementation/)")
elif RAG_MODE == "pro":
    from pro_implementation.answer import answer_question
    print("✓ Using Advanced RAG (pro_implementation/)")
else:
    raise ValueError(f"Invalid RAG_MODE: {RAG_MODE}")


def format_context(context):
    """Format retrieved context documents as HTML with sources"""
    result = "<h2 style='color: #ff7800;'>Relevant Context</h2>\n\n"
    for doc in context:
        result += f"<span style='color: #ff7800;'>Source: {doc.metadata['source']}</span>\n\n"
        result += doc.page_content + "\n\n"
    return result


def chat(history):
    """
    Process user message with RAG pipeline.
    
    Flow:
    1. Extract latest user message
    2. Get answer and context from RAG
    3. Append assistant response to history
    4. Return updated chat and formatted context
    """
    last_message = history[-1]["content"]
    prior = history[:-1]
    answer, context = answer_question(last_message, prior)
    history.append({"role": "assistant", "content": answer})
    return history, format_context(context)


def main():
    """
    Launch Gradio web interface for RAG chatbot.
    
    UI Layout:
    - Left: Chatbot conversation window
    - Right: Retrieved context display
    - Two-column responsive design
    """
    def put_message_in_chatbot(message, history):
        """Add user message to chat history and clear input"""
        return "", history + [{"role": "user", "content": message}]

    # Custom theme with Inter font
    theme = gr.themes.Soft(font=["Inter", "system-ui", "sans-serif"])

    with gr.Blocks(title="Insurellm Expert Assistant", theme=theme) as ui:
        gr.Markdown("# 🏢 Insurellm Expert Assistant\nAsk me anything about Insurellm!")

        with gr.Row():
            # Left column: Chat interface
            with gr.Column(scale=1):
                chatbot = gr.Chatbot(
                    label="💬 Conversation", height=600, type="messages", show_copy_button=True
                )
                message = gr.Textbox(
                    label="Your Question",
                    placeholder="Ask anything about Insurellm...",
                    show_label=False,
                )

            # Right column: Retrieved context display
            with gr.Column(scale=1):
                context_markdown = gr.Markdown(
                    label="📚 Retrieved Context",
                    value="*Retrieved context will appear here*",
                    container=True,
                    height=600,
                )

        # Event chain: Submit message → Add to chat → Get RAG response → Update UI
        message.submit(
            put_message_in_chatbot, inputs=[message, chatbot], outputs=[message, chatbot]
        ).then(chat, inputs=chatbot, outputs=[chatbot, context_markdown])

    ui.launch(inbrowser=True)


if __name__ == "__main__":
    main()

