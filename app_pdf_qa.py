import streamlit as st
import os
import tempfile
from datetime import datetime
from langchain_core.callbacks import BaseCallbackHandler
from pdf_qa_engine import PDFQAEngine
from model_config import MODEL_CONFIG

st.set_page_config(page_title="Chat with PDF", layout="wide")

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text
        self.token_count = 0

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.token_count += 1
        # Add cursor effect for smoother appearance
        self.container.markdown(self.text + "▌")

    def on_llm_end(self, _response, **_kwargs) -> None:
        # Remove cursor when done
        self.container.markdown(self.text)

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
    /* Stop button styling - Floating square button */
    div:has(> span#stop-btn-anchor) + div button {
        position: fixed;
        bottom: 28px;
        right: 3rem;
        z-index: 10000;
        width: 2.5rem !important;
        height: 2.5rem !important;
        border-radius: 4px !important;
        background-color: #ff4b4b;
        color: white;
        border: none;
        padding: 0;
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.2);
    }
    div:has(> span#stop-btn-anchor) + div button:hover {
        background-color: #ff3333;
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
                meta = []
                if "timestamp" in message: meta.append(f"🕒 {message['timestamp']}")
                if "response_time" in message and message["response_time"] > 0: meta.append(f"⏱️ {message['response_time']:.2f}s")
                if meta: st.caption(" | ".join(meta))
                
                if "pages" in message and message["pages"]:
                    cols = st.columns(len(message["pages"]) + 10)
                    for idx, pg in enumerate(message["pages"]):
                        if cols[idx].button(str(pg), key=f"msg_{i}_{idx}"):
                            st.session_state.pdf_page = pg
                            st.session_state.show_right = True
                            st.rerun()

    if prompt := st.chat_input("Ask a question about your PDF..."):
        current_time = datetime.now().strftime("%H:%M:%S")
        st.session_state.messages.append({"role": "user", "content": prompt, "timestamp": current_time})
        
        _, col_msg = st.columns([2, 10])
        with col_msg:
            with st.chat_message("user", avatar="👤"):
                st.markdown(prompt)
                st.caption(f"🕒 {current_time}")

        with st.chat_message("assistant", avatar="🤖"):
            try:
                message_placeholder = st.empty()
                stream_handler = StreamHandler(message_placeholder)
                
                # Placeholder for Stop button
                stop_placeholder = st.empty()
                with stop_placeholder:
                    st.markdown('<span id="stop-btn-anchor"></span>', unsafe_allow_html=True)
                    st.button("⏹", key="stop_gen", help="Stop generation")

                # Show thinking indicator only for initial retrieval
                with st.spinner("Retrieving relevant context..."):
                    response = st.session_state.engine.answer_question(prompt, callbacks=[stream_handler])
                
                # Clear stop button after response is complete
                stop_placeholder.empty()

                # Parse response
                answer_text = response["result"]
                time_taken = response["response_time"]
                source_docs = response["source_documents"]
                pages = sorted(list(set([doc.metadata.get("page", 0) + 1 for doc in source_docs]))) if source_docs else []

                # Ensure final text is displayed (handles greetings/non-streaming cases)
                message_placeholder.markdown(answer_text)

                current_time = datetime.now().strftime("%H:%M:%S")
                
                # Inline Metadata
                meta = [f"🕒 {current_time}"]
                if time_taken > 0: meta.append(f"⏱️ {time_taken:.2f}s")
                st.caption(" | ".join(meta))
                
                if pages:
                    cols = st.columns(len(pages) + 10)
                    for idx, pg in enumerate(pages):
                        if cols[idx].button(str(pg), key=f"curr_{idx}"):
                            st.session_state.pdf_page = pg
                            st.session_state.show_right = True
                            st.rerun()

                st.session_state.messages.append({"role": "assistant", "content": answer_text, "response_time": time_taken, "pages": pages, "timestamp": current_time})
                st.rerun()
            except Exception as e:
                st.error(f"Error generating response: {e}")

if __name__ == "__main__":
    main()
