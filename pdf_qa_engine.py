from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import logging
import time

logger = logging.getLogger(__name__)

class PDFQAEngine:
    def __init__(self, model_name="mistral", base_url="http://localhost:11434"):
        self.llm = ChatOllama(model=model_name, base_url=base_url, streaming=True)
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vector_store = None
        self.qa_chain = None

    def ingest_pdf(self, pdf_file_path):
        logger.info(f"Ingesting PDF: {pdf_file_path}")
        loader = PyPDFLoader(pdf_file_path)
        documents = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = text_splitter.split_documents(documents)
        
        self.vector_store = FAISS.from_documents(chunks, self.embeddings)
        
        retriever = self.vector_store.as_retriever(search_kwargs={"k": 4})
        
        prompt_template = """Use the following pieces of context to answer the question at the end. 
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        
        Context: {context}
        
        Question: {question}
        Answer:"""
        
        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )
        
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": PROMPT},
            return_source_documents=True
        )
        logger.info("PDF Ingested and QA Chain created.")

    def _is_greeting(self, text):
        greetings = ["hi", "hello", "hey", "greetings", "good morning", "good afternoon", "good evening"]
        text = text.lower().strip(" .!,")
        return text in greetings

    def answer_question(self, question, callbacks=None):
        start_time = time.time()
        
        # Quick greeting check
        if self._is_greeting(question):
            return {
                "result": "Hello! How can I help you with your document today?",
                "response_time": 0.0,
                "source_documents": []
            }

        if not self.qa_chain:
            return {
                "result": "Please upload and process a PDF document first.",
                "response_time": 0.0,
                "source_documents": []
            }
        
        response = self.qa_chain.invoke({"query": question}, config={"callbacks": callbacks})
        end_time = time.time()
        
        return {
            "result": response["result"],
            "response_time": end_time - start_time,
            "source_documents": response.get("source_documents", [])
        }
