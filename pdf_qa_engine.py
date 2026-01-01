"""
PDF QA Engine with 20B LLM (CPU+GPU Offload)
Uses llama-cpp-python for efficient inference with GGUF models
Supports multi-page answers with exact page references and confidence scores
"""

import logging
import json
import math
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import chromadb
from chromadb.config import Settings
import numpy as np

# Import transformers for GPT-2
try:
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("transformers not installed. Install with: pip install transformers torch")

# Import configuration
from model_config import (
    MODEL_CONFIG,
    INFERENCE_CONFIG,
    RETRIEVAL_CONFIG,
    get_model_path,
    validate_model_exists
)

logger = logging.getLogger(__name__)


class PDFQAEngine:
    """
    QA Engine using 20B parameter LLM with CPU+GPU offloading.
    Features:
    - Multi-page answer support with exact page references
    - Confidence scoring (0-100%)
    - Response time optimization (5-15 seconds target)
    - Memory efficient with GGUF quantization
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        n_gpu_layers: Optional[int] = None,
        n_ctx: int = 4096,
        chroma_persist_dir: str = "chroma_db"
    ):
        """
        Initialize PDF QA Engine.

        Args:
            model_path: Path to GGUF model file (if None, uses config)
            n_gpu_layers: Number of layers to offload to GPU (if None, uses config)
            n_ctx: Context window size
            chroma_persist_dir: Directory for ChromaDB persistence
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers is not installed.\n"
                "Install with: pip install transformers torch"
            )

        # Model configuration
        self.model_path = model_path or get_model_path()
        self.n_gpu_layers = n_gpu_layers if n_gpu_layers is not None else MODEL_CONFIG["n_gpu_layers"]
        self.n_ctx = n_ctx

        logger.info(f"Initializing PDF QA Engine")
        logger.info(f"Model: {self.model_path}")
        logger.info(f"GPU Layers: {self.n_gpu_layers}")
        logger.info(f"Context Window: {self.n_ctx}")

        # Validate model exists
        try:
            validate_model_exists()
        except FileNotFoundError as e:
            logger.error(str(e))
            raise

        # Load LLM with transformers
        logger.info("Loading GPT-2 model (this may take 30-60 seconds)...")
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_path)
        self.model = GPT2LMHeadModel.from_pretrained(self.model_path)
        
        # Set pad token
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Move to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        logger.info(f"GPT-2 model loaded successfully on {self.device}!")

        # Initialize ChromaDB for document storage
        self.chroma_client = chromadb.PersistentClient(
            path=chroma_persist_dir,
            settings=Settings(anonymized_telemetry=False)
        )

        # Initialize embedding model for semantic search
        if RETRIEVAL_CONFIG["use_semantic_search"]:
            try:
                from sentence_transformers import SentenceTransformer
                self.embedding_model = SentenceTransformer(RETRIEVAL_CONFIG["embedding_model"])
                logger.info(f"Loaded embedding model: {RETRIEVAL_CONFIG['embedding_model']}")
            except Exception as e:
                logger.warning(f"Failed to load embeddings: {e}. Using keyword search only.")
                self.embedding_model = None
        else:
            self.embedding_model = None

    def create_collection(
        self,
        session_id: str,
        page_images: List[str],
        page_texts: List[Dict[str, Any]],
        metadata: Dict[str, Any]
    ) -> bool:
        """
        Create ChromaDB collection with page data.

        Args:
            session_id: Unique session identifier
            page_images: List of page image paths
            page_texts: List of page text data with format: [{'page': 1, 'text': '...'}, ...]
            metadata: PDF metadata

        Returns:
            True if successful
        """
        try:
            collection_name = f"pdf_{session_id}"

            # Delete existing collection if it exists
            try:
                self.chroma_client.delete_collection(collection_name)
            except:
                pass

            collection = self.chroma_client.create_collection(
                name=collection_name,
                metadata={"session_id": session_id, **metadata}
            )

            # Add pages to collection
            documents = []
            metadatas = []
            ids = []

            for i, (img_path, text_data) in enumerate(zip(page_images, page_texts)):
                page_num = i + 1
                doc_text = text_data.get('text', '')

                documents.append(doc_text if doc_text else f"Page {page_num}")
                metadatas.append({
                    'page': page_num,
                    'image_path': img_path,
                    'has_text': len(doc_text) > 0,
                    'text_length': len(doc_text)
                })
                ids.append(f"page_{page_num}")

            # Add to ChromaDB
            collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )

            logger.info(f"Created collection with {len(documents)} pages")
            return True

        except Exception as e:
            logger.error(f"Error creating collection: {str(e)}")
            return False

    def _retrieve_pages(
        self,
        query: str,
        session_id: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant pages using semantic search.

        Args:
            query: Search query
            session_id: Session identifier
            top_k: Number of results

        Returns:
            List of page results with scores
        """
        try:
            collection_name = f"pdf_{session_id}"
            collection = self.chroma_client.get_collection(collection_name)

            results = collection.query(
                query_texts=[query],
                n_results=top_k
            )

            pages = []
            for i in range(len(results['ids'][0])):
                metadata = results['metadatas'][0][i]
                distance = results['distances'][0][i]

                # Convert distance to similarity score (0-1)
                similarity = 1.0 / (1.0 + distance)

                pages.append({
                    'page': metadata['page'],
                    'image_path': metadata['image_path'],
                    'score': similarity,
                    'text': results['documents'][0][i],
                    'distance': distance
                })

            return pages

        except Exception as e:
            logger.error(f"Error retrieving pages: {str(e)}")
            return []

    def _calculate_confidence(
        self,
        answer: str,
        context: str,
        query: str,
        page_scores: List[float]
    ) -> float:
        """
        Calculate confidence score (0-100%) for the answer.

        Considers:
        - Retrieval scores (how relevant the pages are)
        - Answer length (too short = low confidence)
        - Context overlap (answer uses context)

        Args:
            answer: Generated answer
            context: Source context
            query: Original query
            page_scores: Similarity scores from retrieval

        Returns:
            Confidence score (0-100)
        """
        try:
            # Component 1: Retrieval confidence (40% weight)
            # Higher page scores = better retrieval
            avg_retrieval_score = np.mean(page_scores) if page_scores else 0.5
            retrieval_confidence = min(avg_retrieval_score * 100, 100)

            # Component 2: Answer quality (40% weight)
            # Check answer length (too short = uncertain)
            answer_length = len(answer.split())
            if answer_length < 10:
                length_score = 50
            elif answer_length < 30:
                length_score = 70
            else:
                length_score = 90

            # Component 3: Context usage (20% weight)
            # Check if answer uses context (simple overlap check)
            answer_words = set(answer.lower().split())
            context_words = set(context.lower().split())
            overlap = len(answer_words & context_words)
            context_score = min(overlap * 2, 100)  # Scale based on overlap

            # Weighted combination
            confidence = (
                retrieval_confidence * 0.4 +
                length_score * 0.4 +
                context_score * 0.2
            )

            return round(min(confidence, 100), 1)

        except Exception as e:
            logger.error(f"Error calculating confidence: {e}")
            return 50.0  # Default to medium confidence

    def _extract_page_references(
        self,
        answer: str,
        relevant_pages: List[Dict[str, Any]]
    ) -> str:
        """
        Extract and format page references from answer.

        Formats like:
        - "Page 5"
        - "Pages 5, 12-14, 18"

        Args:
            answer: Generated answer
            relevant_pages: List of pages used

        Returns:
            Formatted page reference string
        """
        try:
            # Extract page numbers
            page_numbers = sorted(set([p['page'] for p in relevant_pages]))

            if not page_numbers:
                return "N/A"

            # Group consecutive pages into ranges
            ranges = []
            start = page_numbers[0]
            end = page_numbers[0]

            for i in range(1, len(page_numbers)):
                if page_numbers[i] == end + 1:
                    end = page_numbers[i]
                else:
                    # Add previous range
                    if start == end:
                        ranges.append(f"{start}")
                    else:
                        ranges.append(f"{start}-{end}")
                    start = page_numbers[i]
                    end = page_numbers[i]

            # Add last range
            if start == end:
                ranges.append(f"{start}")
            else:
                ranges.append(f"{start}-{end}")

            # Format output
            if len(ranges) == 1:
                return f"Page {ranges[0]}"
            else:
                return f"Pages {', '.join(ranges)}"

        except Exception as e:
            logger.error(f"Error extracting page references: {e}")
            return "N/A"

    def answer_question(
        self,
        question: str,
        session_id: str,
        top_k: int = 5,
        use_text_context: bool = True,
        return_images: bool = False,
        conversation_history: List[Dict] = None
    ) -> Dict[str, Any]:
        """
        Answer question with multi-page support and confidence scoring.

        Args:
            question: User's question
            session_id: Session identifier
            top_k: Number of pages to retrieve
            use_text_context: Include text context
            return_images: Return extracted images
            conversation_history: Previous Q&A for context

        Returns:
            Dict with:
                - answer: Generated answer
                - page_references: Formatted page numbers (e.g., "Pages 5, 12-14")
                - confidence: Confidence score (0-100%)
                - response_time: Generation time
                - pages_used: List of page numbers
                - images: List of image paths (if return_images=True)
        """
        try:
            import time
            start_time = time.time()

            # Retrieve relevant pages
            relevant_pages = self._retrieve_pages(question, session_id, top_k)

            if not relevant_pages:
                return {
                    'answer': "I couldn't find relevant information in the document.",
                    'page_references': "N/A",
                    'confidence': 0.0,
                    'response_time': time.time() - start_time,
                    'pages_used': [],
                    'images': []
                }

            # Build context from multiple pages
            context = ""
            pages_used = []
            page_scores = []

            for page_data in relevant_pages[:top_k]:
                if page_data['text']:
                    context += f"\n--- Page {page_data['page']} ---\n"
                    context += page_data['text'] + "\n"
                    pages_used.append(page_data['page'])
                    page_scores.append(page_data['score'])

            # Build conversation context if provided
            history_text = ""
            if conversation_history and len(conversation_history) > 0:
                history_text = "\nPrevious conversation:\n"
                for i, exchange in enumerate(conversation_history[-3:], 1):
                    history_text += f"Q{i}: {exchange['question']}\n"
                    history_text += f"A{i}: {exchange['answer'][:150]}...\n\n"

            # Build prompt for LLM
            prompt = self._build_prompt(question, context, history_text)

            # Generate answer with LLM
            logger.info(f"Generating answer for: {question[:50]}...")
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            outputs = self.model.generate(
                inputs['input_ids'],
                max_length=inputs['input_ids'].shape[1] + MODEL_CONFIG["max_tokens"],
                temperature=MODEL_CONFIG["temperature"],
                top_p=MODEL_CONFIG["top_p"],
                top_k=MODEL_CONFIG["top_k"],
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
            answer = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()

            # Calculate confidence score
            confidence = self._calculate_confidence(
                answer, context, question, page_scores
            )

            # Format page references
            page_references = self._extract_page_references(answer, relevant_pages)

            # Collect images if requested
            images = []
            if return_images:
                images = self._collect_page_images(session_id, pages_used)

            response_time = time.time() - start_time

            logger.info(f"Answer generated in {response_time:.2f}s (confidence: {confidence}%)")

            return {
                'answer': answer,
                'page_references': page_references,
                'confidence': confidence,
                'response_time': response_time,
                'pages_used': pages_used,
                'images': images
            }

        except Exception as e:
            logger.error(f"Error answering question: {str(e)}", exc_info=True)
            return {
                'answer': f"Error generating answer: {str(e)}",
                'page_references': "N/A",
                'confidence': 0.0,
                'response_time': 0.0,
                'pages_used': [],
                'images': []
            }

    def _build_prompt(self, question: str, context: str, history: str = "") -> str:
        """
        Build prompt for LLM.

        Args:
            question: User question
            context: Document context
            history: Conversation history

        Returns:
            Formatted prompt
        """
        prompt = f"""You are a helpful AI assistant that answers questions about documents accurately and comprehensively.

Instructions:
- Provide detailed, complete answers based on the context
- If the answer spans multiple pages, cite all relevant page numbers
- Include specific details and examples from the text
- If information is not in the context, say so clearly
- Be thorough but focused on answering the question

{history}
Context from document:
{context}

Question: {question}

Answer (be detailed and comprehensive):"""

        return prompt

    def _collect_page_images(self, session_id: str, pages: List[int]) -> List[str]:
        """
        Collect image paths for given pages.

        Args:
            session_id: Session ID
            pages: List of page numbers

        Returns:
            List of image URLs
        """
        images = []
        try:
            from pathlib import Path

            for page_num in pages:
                # Page images
                page_img = f"/data/{session_id}/page_{page_num:04d}.png"
                images.append(page_img)

                # Embedded images
                embedded_dir = Path("data") / session_id / "embedded_images"
                if embedded_dir.exists():
                    import glob
                    pattern = str(embedded_dir / f"page_{page_num:04d}_img_*.*")
                    for img_path in glob.glob(pattern):
                        img_name = Path(img_path).name
                        images.append(f"/data/{session_id}/embedded_images/{img_name}")

        except Exception as e:
            logger.error(f"Error collecting images: {e}")

        return images

    def cleanup_session(self, session_id: str):
        """Clean up session data."""
        try:
            collection_name = f"pdf_{session_id}"
            self.chroma_client.delete_collection(collection_name)
            logger.info(f"Cleaned up session {session_id}")
        except Exception as e:
            logger.warning(f"Error cleaning up session: {str(e)}")

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            'model_path': self.model_path,
            'n_gpu_layers': self.n_gpu_layers,
            'n_ctx': self.n_ctx,
            'n_threads': MODEL_CONFIG['n_threads'],
            'model_type': MODEL_CONFIG['model_type']
        }


if __name__ == "__main__":
    # Test the engine
    logging.basicConfig(level=logging.INFO)

    try:
        engine = PDFQAEngine()
        print("PDF QA Engine initialized successfully!")
        print(engine.get_model_info())
    except Exception as e:
        print(f"Failed to initialize: {e}")
