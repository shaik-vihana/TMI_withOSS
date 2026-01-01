"""
PDF Q&A Application with 20B LLM (CPU+GPU Offload)
Flask web application for document question-answering
Features: Multi-page answers, confidence scores, exact page references
"""

from flask import Flask, render_template, request, jsonify, session, send_file
from werkzeug.utils import secure_filename
import os
import uuid
import logging
from pathlib import Path
import time
from datetime import datetime

from pdf_processor import PDFProcessor
from pdf_qa_engine import PDFQAEngine

# Global in-memory log storage (backup for quick access)
performance_logs = []

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.urandom(24)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'pdf'}

# Create necessary directories
for directory in ['uploads', 'data', 'logs', 'processed_pdfs', 'chroma_db', 'models']:
    os.makedirs(directory, exist_ok=True)

# Initialize PDF QA Engine
try:
    qa_engine = PDFQAEngine(
        chroma_persist_dir='chroma_db'
    )
    logger.info("="*80)
    logger.info("PDF QA Engine initialized successfully")
    logger.info(f"Model Info: {qa_engine.get_model_info()}")
    logger.info("="*80)
except Exception as e:
    logger.error(f"Failed to initialize PDF QA Engine: {str(e)}")
    logger.error("Please ensure:")
    logger.error("1. transformers and torch are installed")
    logger.error("2. Model directory exists at the path specified in model_config.py")
    logger.error("3. Model files (config.json, pytorch_model.bin, etc.) are present")
    qa_engine = None


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def log_performance(session_id, question, answer, response_time, page_refs, confidence=0.0):
    """Log performance metrics to file with confidence and page references."""
    log_file = Path('logs') / 'qa_performance.txt'
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Create log entry
    log_entry = (
        f"[{timestamp}] "
        f"Session: {session_id} | "
        f"Question: {question[:50]}... | "
        f"Response Time: {response_time:.2f}s | "
        f"Pages: {page_refs} | "
        f"Confidence: {confidence:.1f}% | "
        f"Answer Length: {len(answer)} chars\n"
    )

    # Append to log file
    try:
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(log_entry)
    except Exception as e:
        logger.error(f"Failed to write to log file: {e}")

    # Also keep in-memory for quick access (last 100 entries)
    global performance_logs
    memory_entry = {
        'timestamp': timestamp,
        'session_id': session_id,
        'question': question,
        'response_time': response_time,
        'page_references': page_refs,
        'confidence': confidence,
        'answer_length': len(answer)
    }
    performance_logs.append(memory_entry)
    if len(performance_logs) > 100:
        performance_logs.pop(0)


@app.route('/')
def index():
    return render_template('upload.html')


@app.route('/qa')
def qa_page():
    if 'session_id' not in session:
        return render_template('upload.html')

    metadata = session.get('metadata', {})
    return render_template('qa.html', metadata=metadata)


@app.route('/upload', methods=['POST'])
def upload_pdf():
    """Upload and process PDF."""
    try:
        if not qa_engine:
            return jsonify({'error': 'PDF QA Engine not initialized. Please check server logs.'}), 500

        # Check if file is present
        if 'pdf_file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['pdf_file']

        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Please upload a PDF file.'}), 400

        # Generate unique session ID
        session_id = str(uuid.uuid4())
        session['session_id'] = session_id

        # Save the PDF file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"{session_id}_{filename}")
        file.save(filepath)

        logger.info(f"PDF uploaded: {filepath}")
        logger.info(f"File size: {os.path.getsize(filepath) / (1024*1024):.2f} MB")

        # Process PDF
        output_dir = os.path.join('processed_pdfs', session_id)
        os.makedirs(output_dir, exist_ok=True)

        # Get processing options
        dpi = int(request.form.get('dpi', 150))
        extract_images = request.form.get('extract_images', 'true').lower() == 'true'

        logger.info(f"Processing PDF with DPI={dpi}, extract_images={extract_images}")

        processor = PDFProcessor(
            dpi=dpi,
            extract_images=extract_images,
            extract_text=True,
            batch_size=10
        )

        # Process PDF
        start_time = time.time()
        result = processor.process_pdf(filepath, output_dir)
        processing_time = time.time() - start_time

        if not result:
            os.remove(filepath)
            return jsonify({'error': 'Failed to process PDF. Please check the file.'}), 400

        logger.info(f"PDF processed in {processing_time:.2f} seconds")

        # Create ChromaDB collection
        logger.info("Creating vector index...")
        index_start = time.time()

        success = qa_engine.create_collection(
            session_id=session_id,
            page_images=result['page_images'],
            page_texts=result['page_text'],
            metadata=result['metadata']
        )

        index_time = time.time() - index_start

        if not success:
            return jsonify({'error': 'Failed to create search index'}), 500

        logger.info(f"Index created in {index_time:.2f} seconds")

        # Save metadata to session
        metadata = {
            'filename': filename,
            'total_pages': result['metadata']['total_pages'],
            'num_page_images': len(result['page_images']),
            'num_embedded_images': len(result['embedded_images']),
            'processing_time': round(processing_time, 2),
            'index_time': round(index_time, 2),
            'session_id': session_id
        }
        session['metadata'] = metadata
        session['conversation_history'] = []  # Initialize empty conversation history

        return jsonify({
            'success': True,
            'message': f'PDF processed successfully! {metadata["total_pages"]} pages indexed.',
            'metadata': metadata
        }), 200

    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}", exc_info=True)
        return jsonify({'error': f'Error processing PDF: {str(e)}'}), 500


@app.route('/ask', methods=['POST'])
def ask_question():
    """Answer question using 20B LLM."""
    try:
        if not qa_engine:
            return jsonify({'error': 'PDF QA Engine not available'}), 500

        data = request.get_json()

        if not data or 'question' not in data:
            return jsonify({'error': 'No question provided'}), 400

        question = data['question'].strip()

        if not question:
            return jsonify({'error': 'Question cannot be empty'}), 400

        if len(question) > 1000:
            return jsonify({'error': 'Question is too long. Maximum 1000 characters.'}), 400

        session_id = session.get('session_id')

        if not session_id:
            return jsonify({'error': 'No PDF uploaded. Please upload a PDF first.'}), 400

        # Options
        top_k = int(data.get('top_k', 5))

        # Get conversation history from session (limit to last 5)
        conversation_history = session.get('conversation_history', [])[-5:]

        logger.info(f"Question: {question[:100]}... (top_k={top_k}, history_len={len(conversation_history)})")

        # Track response time
        start_time = time.time()

        # Generate answer with LLM
        result = qa_engine.answer_question(
            question=question,
            session_id=session_id,
            top_k=top_k,
            use_text_context=True,
            return_images=True,
            conversation_history=conversation_history
        )

        response_time = time.time() - start_time

        # Extract results
        answer = result.get('answer', '')
        page_references = result.get('page_references', 'N/A')
        confidence = result.get('confidence', 0.0)
        pages_used = result.get('pages_used', [])
        images = result.get('images', [])

        if not answer:
            return jsonify({'error': 'Could not generate an answer. Please try rephrasing your question.'}), 500

        # Log performance
        log_performance(session_id, question, answer, response_time, page_references, confidence)

        # Add to conversation history
        current_timestamp = datetime.now().strftime('%H:%M:%S')
        if 'conversation_history' not in session:
            session['conversation_history'] = []

        session['conversation_history'].append({
            'question': question,
            'answer': answer,
            'pages': pages_used,
            'timestamp': current_timestamp
        })

        # Keep only last 5 exchanges
        session['conversation_history'] = session['conversation_history'][-5:]

        logger.info(f"Answer generated in {response_time:.2f}s (confidence: {confidence}%)")

        return jsonify({
            'success': True,
            'answer': answer,
            'question': question,
            'response_time': round(response_time, 3),
            'page_references': page_references,
            'confidence': confidence,
            'images': images,
            'pages_used': pages_used
        }), 200

    except Exception as e:
        logger.error(f"Error answering question: {str(e)}", exc_info=True)
        return jsonify({'error': f'Error generating answer: {str(e)}'}), 500


@app.route('/reset', methods=['POST'])
def reset_session():
    """Reset session and cleanup."""
    try:
        session_id = session.get('session_id')

        if session_id:
            # Clean up ChromaDB collection
            if qa_engine:
                qa_engine.cleanup_session(session_id)

            # Clean up files
            upload_dir = Path(app.config['UPLOAD_FOLDER'])
            for file in upload_dir.glob(f"{session_id}_*"):
                file.unlink()

            # Clean up processed PDFs
            processed_dir = Path('processed_pdfs') / session_id
            if processed_dir.exists():
                import shutil
                shutil.rmtree(processed_dir)

        session.clear()

        return jsonify({'success': True, 'message': 'Session reset successfully'}), 200

    except Exception as e:
        logger.error(f"Error resetting session: {str(e)}")
        return jsonify({'error': f'Error resetting session: {str(e)}'}), 500


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    health_status = {
        'status': 'healthy' if qa_engine else 'degraded',
        'qa_engine': qa_engine is not None
    }

    if qa_engine:
        health_status['model_info'] = qa_engine.get_model_info()

    return jsonify(health_status), 200


@app.route('/analytics', methods=['GET'])
def analytics():
    """Get analytics data."""
    try:
        log_file = Path('logs') / 'qa_performance.txt'

        if not log_file.exists():
            return jsonify({
                'total_questions': 0,
                'avg_response_time': 0,
                'avg_confidence': 0,
                'recent_logs': []
            }), 200

        with open(log_file, 'r', encoding='utf-8') as f:
            logs = f.readlines()

        # Parse logs
        total_questions = len(logs)
        response_times = []
        confidences = []
        recent_logs = []

        for line in logs[-50:]:  # Last 50 entries
            try:
                # Extract response time
                if 'Response Time:' in line:
                    time_str = line.split('Response Time:')[1].split('s')[0].strip()
                    response_times.append(float(time_str))

                # Extract confidence
                if 'Confidence:' in line:
                    conf_str = line.split('Confidence:')[1].split('%')[0].strip()
                    confidences.append(float(conf_str))

                recent_logs.append(line.strip())
            except:
                continue

        avg_response_time = sum(response_times) / len(response_times) if response_times else 0
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0

        return jsonify({
            'total_questions': total_questions,
            'avg_response_time': round(avg_response_time, 2),
            'avg_confidence': round(avg_confidence, 1),
            'recent_logs': recent_logs[::-1]  # Newest first
        }), 200

    except Exception as e:
        logger.error(f"Error getting analytics: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/view-log', methods=['GET'])
def view_log():
    """View performance log in browser."""
    try:
        log_file = Path('logs') / 'qa_performance.txt'

        if not log_file.exists():
            return "<h1>No logs found</h1><p>Upload a PDF and ask questions to generate logs.</p>", 200

        with open(log_file, 'r', encoding='utf-8') as f:
            log_content = f.read()

        # Parse log entries
        log_lines = log_content.strip().split('\n') if log_content.strip() else []
        log_lines.reverse()  # Show newest first

        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Performance Analytics</title>
            <style>
                body {{ font-family: Arial; margin: 20px; background: #f5f5f5; }}
                .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; }}
                h1 {{ color: #333; }}
                .log-entry {{ padding: 10px; margin: 10px 0; background: #f9f9f9; border-left: 4px solid #4CAF50; }}
                .stats {{ display: flex; gap: 20px; margin: 20px 0; }}
                .stat {{ flex: 1; padding: 20px; background: #e8f5e9; border-radius: 8px; text-align: center; }}
                .stat-value {{ font-size: 2em; font-weight: bold; color: #2e7d32; }}
                .stat-label {{ color: #666; margin-top: 5px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>📊 Performance Analytics</h1>
                <div class="stats">
                    <div class="stat">
                        <div class="stat-value">{len(log_lines)}</div>
                        <div class="stat-label">Total Questions</div>
                    </div>
                    <div class="stat">
                        <div class="stat-value">{(len(log_content) / 1024):.1f}KB</div>
                        <div class="stat-label">Log File Size</div>
                    </div>
                </div>
                <h2>Recent Activity (Newest First)</h2>
                <div>
        """

        for line in log_lines[:100]:  # Show last 100
            if line.strip():
                html_content += f'<div class="log-entry">{line}</div>\n'

        html_content += """
                </div>
                <p style="text-align: center; margin-top: 30px;">
                    <a href="/" style="padding: 10px 20px; background: #4CAF50; color: white; text-decoration: none; border-radius: 5px;">Back to App</a>
                </p>
            </div>
        </body>
        </html>
        """

        return html_content

    except Exception as e:
        logger.error(f"Error reading log file: {e}")
        return f"<h1>Error reading logs: {e}</h1>", 500


@app.route('/data/<session_id>/<path:filename>')
def serve_file(session_id, filename):
    """Serve page images and embedded images."""
    try:
        # Try processed_pdfs directory first (page images)
        file_path = Path('processed_pdfs') / session_id / filename

        if file_path.exists():
            return send_file(file_path, mimetype='image/png')

        # Try data directory (embedded images)
        file_path = Path('data') / session_id / filename

        if file_path.exists():
            return send_file(file_path, mimetype='image/png')

        return jsonify({'error': 'File not found'}), 404

    except Exception as e:
        logger.error(f"Error serving file: {str(e)}")
        return jsonify({'error': f'Error serving file: {str(e)}'}), 500


if __name__ == '__main__':
    if not qa_engine:
        logger.error("="*80)
        logger.error("STARTUP FAILED")
        logger.error("="*80)
        logger.error("PDF QA Engine could not be initialized.")
        logger.error("")
        logger.error("Please follow these steps:")
        logger.error("1. Ensure transformers and torch are installed:")
        logger.error("   pip install transformers torch")
        logger.error("2. Download the GPT-2 model using the startup script")
        logger.error("3. Ensure model files are in the correct directory")
        logger.error("4. Run this app again")
        logger.error("="*80)
        exit(1)

    logger.info("="*80)
    logger.info("PDF Q&A SYSTEM WITH GPT-2")
    logger.info("="*80)
    logger.info(f"Server running at: http://localhost:5000")
    logger.info(f"Analytics: http://localhost:5000/view-log")
    logger.info(f"Health check: http://localhost:5000/health")
    logger.info("="*80)

    app.run(host='0.0.0.0', port=5000, debug=False)
