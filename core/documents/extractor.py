import fitz
import markdown
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)


class DocumentProcessingError(Exception):
    """Custom exception for document processing errors."""

    pass


def extract_text_from_pdf(file_path: str) -> str:
    """
    Extract text content from PDF files.

    Args:
        file_path: Path to the PDF file

    Returns:
        Extracted text content

    Raises:
        DocumentProcessingError: If PDF processing fails
    """
    if not Path(file_path).exists():
        raise DocumentProcessingError(f"PDF file not found: {file_path}")

    text = ""
    doc = None
    try:
        doc = fitz.open(file_path)

        if doc.page_count == 0:
            raise DocumentProcessingError("PDF contains no pages")

        for page_num in range(doc.page_count):
            try:
                page = doc[page_num]
                page_text = page.get_text()
                text += page_text + "\n"
            except Exception as e:
                logger.warning(f"Error extracting text from page {page_num + 1}: {e}")
                continue

        if not text.strip():
            logger.warning(f"No text content extracted from PDF: {file_path}")

        return text

    except fitz.FileDataError as e:
        logger.error(f"Invalid PDF file {file_path}: {e}")
        raise DocumentProcessingError(f"Invalid or corrupted PDF file: {str(e)}")
    except Exception as e:
        logger.error(f"Error extracting text from PDF {file_path}: {e}")
        raise DocumentProcessingError(f"Error processing PDF: {str(e)}")
    finally:
        if doc:
            doc.close()


def extract_text_from_markdown(file_path: str) -> str:
    """
    Convert Markdown to plain text.

    Args:
        file_path: Path to the Markdown file

    Returns:
        Extracted text content

    Raises:
        DocumentProcessingError: If Markdown processing fails
    """
    if not Path(file_path).exists():
        raise DocumentProcessingError(f"Markdown file not found: {file_path}")

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            md_content = f.read()

        if not md_content.strip():
            logger.warning(f"Markdown file is empty: {file_path}")
            return ""

        # Convert markdown to HTML
        html = markdown.markdown(md_content)

        # Convert HTML to text with basic formatting preservation
        text = html.replace("<p>", "\n\n").replace("</p>", "")
        text = text.replace("<br>", "\n").replace("<br/>", "\n")
        text = text.replace("<h1>", "\n\n# ").replace("</h1>", "\n")
        text = text.replace("<h2>", "\n\n## ").replace("</h2>", "\n")
        text = text.replace("<h3>", "\n\n### ").replace("</h3>", "\n")
        text = text.replace("<h4>", "\n\n#### ").replace("</h4>", "\n")
        text = text.replace("<h5>", "\n\n##### ").replace("</h5>", "\n")
        text = text.replace("<h6>", "\n\n###### ").replace("</h6>", "\n")

        # Handle lists
        text = text.replace("<li>", "• ").replace("</li>", "\n")
        text = text.replace("<ul>", "\n").replace("</ul>", "\n")
        text = text.replace("<ol>", "\n").replace("</ol>", "\n")

        # Remove remaining HTML tags
        text = re.sub(r"<[^<]+?>", "", text)

        # Clean up extra whitespace
        text = re.sub(r"\n\s*\n", "\n\n", text)
        text = text.strip()

        return text

    except UnicodeDecodeError as e:
        logger.error(f"Encoding error reading Markdown file {file_path}: {e}")
        raise DocumentProcessingError(
            f"Unable to read Markdown file (encoding issue): {str(e)}"
        )
    except Exception as e:
        logger.error(f"Error processing Markdown file {file_path}: {e}")
        raise DocumentProcessingError(f"Error processing Markdown: {str(e)}")


def extract_text_from_plain_text(file_path: str) -> str:
    """
    Extract text from plain text files.

    Args:
        file_path: Path to the text file

    Returns:
        File content

    Raises:
        DocumentProcessingError: If text file processing fails
    """
    if not Path(file_path).exists():
        raise DocumentProcessingError(f"Text file not found: {file_path}")

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        if not content.strip():
            logger.warning(f"Text file is empty: {file_path}")

        return content

    except UnicodeDecodeError:
        # Try with different encodings
        for encoding in ["latin-1", "cp1252", "iso-8859-1"]:
            try:
                with open(file_path, "r", encoding=encoding) as f:
                    content = f.read()
                    logger.info(
                        f"Successfully read {file_path} with {encoding} encoding"
                    )
                    return content
            except UnicodeDecodeError:
                continue

        logger.error(f"Unable to decode text file {file_path} with any encoding")
        raise DocumentProcessingError("Unable to read text file (encoding issue)")
    except Exception as e:
        logger.error(f"Error reading text file {file_path}: {e}")
        raise DocumentProcessingError(f"Error reading text file: {str(e)}")


def extract_text_from_file(file_path: str, content_type: str) -> str:
    """
    Extract text based on file type.

    Args:
        file_path: Path to the file
        content_type: MIME type of the file

    Returns:
        Extracted text content

    Raises:
        DocumentProcessingError: If file processing fails
        ValueError: If content type is not supported
    """
    if not file_path or not content_type:
        raise ValueError("File path and content type are required")

    content_type = content_type.lower()

    try:
        if content_type == "application/pdf":
            return extract_text_from_pdf(file_path)
        elif content_type == "text/markdown":
            return extract_text_from_markdown(file_path)
        elif content_type == "text/plain":
            return extract_text_from_plain_text(file_path)
        else:
            raise ValueError(f"Unsupported file type: {content_type}")

    except DocumentProcessingError:
        # Re-raise document processing errors as-is
        raise
    except Exception as e:
        logger.error(f"Unexpected error processing file {file_path}: {e}")
        raise DocumentProcessingError(f"Unexpected error processing file: {str(e)}")


def validate_extracted_text(text: str, min_length: int = 10) -> bool:
    """
    Validate that extracted text meets minimum quality requirements.

    Args:
        text: Extracted text content
        min_length: Minimum text length to consider valid

    Returns:
        True if text is valid, False otherwise
    """
    if not text or not isinstance(text, str):
        return False

    # Check minimum length
    if len(text.strip()) < min_length:
        return False

    # Check that it's not just whitespace or special characters
    text_cleaned = re.sub(r"[\s\n\r\t\f\v]+", " ", text).strip()
    if len(text_cleaned) < min_length:
        return False

    return True
