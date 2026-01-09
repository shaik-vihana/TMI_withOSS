import streamlit as st
import os
import tempfile
from datetime import datetime
from langchain_core.callbacks import BaseCallbackHandler
from pdf_qa_engine import PDFQAEngine
from model_config import MODEL_CONFIG

st.set_page_config(page_title="Chat with PDF", layout="wide")

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text="", message_context=None):
        self.container = container
        self.text = initial_text
        self.token_count = 0
        self.message_context = message_context

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.token_count += 1
        # Add cursor effect for smoother appearance
        self.container.markdown(self.text + "▌")
        if self.message_context is not None:
            self.message_context["content"] = self.text

    def on_llm_end(self, _response, **_kwargs) -> None:
        # Remove cursor when done
        self.container.markdown(self.text)
        if self.message_context is not None:
            self.message_context["content"] = self.text

def main():
    st.title("📄 Chat with PDF")
    
    # Custom CSS for chat styling
    st.markdown("""
    <style>
    /* Style for inline metadata */
    .stCaption {
        display: inline-block;
        font-family: monospace;
        color: #666;
        background-color: rgba(0,0,0,0.05);
        padding: 2px 6px;
        border-radius: 4px;
    }
    /* Page button styling to match caption font */
    .stButton button {
        font-family: monospace;
        font-size: 0.8rem;
        padding: 2px 8px;
    }
    /* Stop button styling - Next to input field */
    div[data-testid="stChatInput"] {
        position: relative;
    }
    div:has(span#stop-btn-anchor) {
        position: fixed !important;
        bottom: 58px !important;
        right: 75px !important;
        z-index: 99999 !important;
    }
    div:has(span#stop-btn-anchor) button {
        width: 36px !important;
        height: 36px !important;
        border-radius: 4px !important;
        background-color: #ff4b4b !important;
        color: white !important;
        border: none !important;
        padding: 0 !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2) !important;
        transition: background-color 0.2s;
        font-size: 18px !important;
    }
    div:has(span#stop-btn-anchor) button:hover {
        background-color: #ff3333 !important;
    }
    div:has(span#stop-btn-anchor) button:active {
        background-color: #cc0000 !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "engine" not in st.session_state:
        st.session_state.engine = PDFQAEngine(
            model_name=MODEL_CONFIG["model_name"],
            base_url=MODEL_CONFIG["base_url"]
        )

    if "is_generating" not in st.session_state:
        st.session_state.is_generating = False
    
    # Sidebar for file upload
    with st.sidebar:
        st.header("Upload Document")
        uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
        
        if uploaded_file:
            if "current_file" not in st.session_state or st.session_state.current_file != uploaded_file.name:
                with st.spinner("Processing PDF..."):
                    # Save to temp file
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_path = tmp_file.name
                    
                    try:
                        st.session_state.engine.ingest_pdf(tmp_path)
                        st.session_state.current_file = uploaded_file.name
                        st.session_state.messages = [] # Reset chat on new file
                        st.success("PDF Processed Successfully!")
                    except Exception as e:
                        st.error(f"Error processing PDF: {e}")
                    finally:
                        os.unlink(tmp_path)
        
        st.markdown("---")
        st.markdown("### Model Config")
        st.info(f"Model: {MODEL_CONFIG['model_name']}")
        st.info(f"Backend: Ollama")

    # Chat Interface
    for i, message in enumerate(st.session_state.messages):
        if message["role"] == "user":
            _, col_msg = st.columns([2, 10])
            with col_msg:
                with st.chat_message("user", avatar="👤"):
                    st.markdown(message["content"])
                    if "timestamp" in message: st.caption(f"🕒 {message['timestamp']}")
        else:
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(message["content"])
                # Inline Metadata
                meta_text = ""
                if "timestamp" in message: meta_text += f"🕒 {message['timestamp']}"
                if "response_time" in message and message["response_time"] > 0: meta_text += f" | ⏱️ {message['response_time']:.2f}s"
                
                # Layout: Metadata Text | Pages: [1] [2]
                if "pages" in message and message["pages"]:
                    # Fixed ratio layout: Text(4) | Btn(0.5)... | Spacer(5)
                    cols = st.columns([4] + [0.5] * len(message["pages"]) + [5])
                    with cols[0]: 
                        st.caption(meta_text + " | Pages:")
                    for idx, pg in enumerate(message["pages"]):
                        if cols[idx+1].button(str(pg), key=f"msg_{i}_{idx}"):
                            st.session_state.pdf_page = pg
                            st.session_state.show_right = True
                            st.rerun()
                elif meta_text:
                    st.caption(meta_text)

    if prompt := st.chat_input("Ask a question about your PDF..."):
        st.session_state.is_generating = True
        current_time = datetime.now().strftime("%H:%M:%S")
        st.session_state.messages.append({"role": "user", "content": prompt, "timestamp": current_time})

        # Pre-append assistant message to state to prevent clearing on Stop
        st.session_state.messages.append({
            "role": "assistant",
            "content": "",
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "response_time": 0,
            "pages": []
        })
        current_msg_index = len(st.session_state.messages) - 1

        _, col_msg = st.columns([2, 10])
        with col_msg:
            with st.chat_message("user", avatar="👤"):
                st.markdown(prompt)
                st.caption(f"🕒 {current_time}")

        with st.chat_message("assistant", avatar="🤖"):
            # Stop button container
            stop_container = st.container()
            with stop_container:
                st.markdown('<span id="stop-btn-anchor"></span>', unsafe_allow_html=True)
                stop_btn = st.button("⏹", key="stop_gen", help="Stop generation")

            try:
                message_placeholder = st.empty()
                stream_handler = StreamHandler(message_placeholder, message_context=st.session_state.messages[current_msg_index])

                # Check if stop was clicked
                if stop_btn:
                    st.session_state.is_generating = False
                    st.stop()

                # Show thinking indicator only for initial retrieval
                with st.spinner("Retrieving relevant context..."):
                    response = st.session_state.engine.answer_question(prompt, callbacks=[stream_handler])

                # Parse response
                answer_text = response["result"]
                time_taken = response["response_time"]
                source_docs = response["source_documents"]
                pages = sorted(list(set([doc.metadata.get("page", 0) + 1 for doc in source_docs]))) if source_docs else []

                # Ensure final text is displayed (handles greetings/non-streaming cases)
                message_placeholder.markdown(answer_text)

                # Update final metadata in session state
                st.session_state.messages[current_msg_index]["content"] = answer_text
                st.session_state.messages[current_msg_index]["response_time"] = time_taken
                st.session_state.messages[current_msg_index]["pages"] = pages

                # Mark generation complete and clear stop button
                st.session_state.is_generating = False
                stop_container.empty()

            except Exception as e:
                st.session_state.is_generating = False
                stop_container.empty()
                st.error(f"Error generating response: {e}")

if __name__ == "__main__":
    main()
