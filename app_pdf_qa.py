import streamlit as st
import os
import tempfile
from pdf_qa_engine import PDFQAEngine
from model_config import MODEL_CONFIG

st.set_page_config(page_title="Chat with PDF", layout="wide")

def main():
    st.title("📄 Chat with PDF using Mistral 7B")
    
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
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "response_time" in message and message["response_time"] > 0:
                st.caption(f"⏱️ {message['response_time']:.2f}s")
            if "pages" in message and message["pages"]:
                st.caption(f"📄 Pages: {', '.join(map(str, message['pages']))}")

    if prompt := st.chat_input("Ask a question about your PDF..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.engine.answer_question(prompt)
                    
                    # Parse response
                    answer_text = response["result"]
                    time_taken = response["response_time"]
                    source_docs = response["source_documents"]
                    pages = sorted(list(set([doc.metadata.get("page", 0) + 1 for doc in source_docs]))) if source_docs else []

                    st.markdown(answer_text)
                    
                    # Display metadata
                    if time_taken > 0:
                        st.caption(f"⏱️ {time_taken:.2f}s")
                    if pages:
                        st.caption(f"📄 Pages: {', '.join(map(str, pages))}")

                    st.session_state.messages.append({"role": "assistant", "content": answer_text, "response_time": time_taken, "pages": pages})
                except Exception as e:
                    st.error(f"Error generating response: {e}")

if __name__ == "__main__":
    main()
