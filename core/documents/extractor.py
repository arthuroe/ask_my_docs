import fitz
import markdown
import logging
import re

logger = logging.getLogger(__name__)


def extract_text_from_pdf(file_path):
    """Extract text content from PDF files."""
    text = ""
    try:
        doc = fitz.open(file_path)
        for page in doc:
            text += page.get_text()
        return text
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {e}")
        raise Exception(
            status_code=500, detail=f"Error processing PDF: {str(e)}")


def extract_text_from_markdown(file_path):
    """Convert Markdown to plain text."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            md_content = f.read()
        html = markdown.markdown(md_content)
        # Simple HTML to text conversion - in production use a more robust method
        text = html.replace('<p>', '\n\n').replace(
            '</p>', '').replace('<br>', '\n')
        text = text.replace('<h1>', '\n\n# ').replace('</h1>', '\n')
        text = text.replace('<h2>', '\n\n## ').replace('</h2>', '\n')
        text = text.replace('<h3>', '\n\n### ').replace('</h3>', '\n')
        # Remove other HTML tags
        text = re.sub('<[^<]+?>', '', text)
        return text
    except Exception as e:
        logger.error(f"Error processing Markdown: {e}")
        raise Exception(
            status_code=500, detail=f"Error processing Markdown: {str(e)}")


def extract_text_from_file(file_path, content_type):
    """Extract text based on file type."""
    if content_type == "application/pdf":
        return extract_text_from_pdf(file_path)
    elif content_type == "text/markdown":
        return extract_text_from_markdown(file_path)
    elif content_type == "text/plain":
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    else:
        raise Exception(
            status_code=400, detail=f"Unsupported file type: {content_type}")
