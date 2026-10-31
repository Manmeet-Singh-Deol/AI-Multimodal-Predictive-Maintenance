"""Document processing module for industrial maintenance manuals.

Extracts text, detects section headings, chunks content intelligently
respecting sentence boundaries, and extracts embedded schematics/images
from PDF documents using PyMuPDF.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger

from config.settings import get_settings

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None


@dataclass
class Document:
    """Represents a text chunk extracted from a document with associated metadata.

    Attributes:
        content: The text content of the chunk.
        metadata: Dictionary containing chunk metadata (source, page, chunk_id, section_title, etc.).
    """

    content: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Ensure standard metadata fields exist with sensible defaults."""
        self.metadata.setdefault("source", "unknown")
        self.metadata.setdefault("page", 1)
        self.metadata.setdefault("chunk_id", 0)
        self.metadata.setdefault("section_title", "General")


class DocumentProcessor:
    """Parses and chunks maintenance manual PDFs for RAG ingestion.

    Extracts text page-by-page, detects structural section headers,
    applies sentence-boundary-aware chunking with overlap, and extracts
    embedded diagram and component images.
    """

    # Common industrial manual section keywords
    _SECTION_PATTERNS = [
        r"^(?:SECTION|CHAPTER|PART|MODULE)\s+[0-9IVX]+[:\s\.-]*",
        r"^(?:SAFETY|DANGER|WARNING|CAUTION|NOTICE|LOTO)[:\s\.-]*",
        r"^(?:SPECIFICATIONS?|TECHNICAL\s+DATA|PARAMETERS?)[:\s\.-]*",
        r"^(?:TROUBLESHOOTING|FAULT\s+DIAGNOSIS|ERROR\s+CODES?)[:\s\.-]*",
        r"^(?:MAINTENANCE|INSPECTION|LUBRICATION|SERVICING)[:\s\.-]*",
        r"^(?:DISASSEMBLY|ASSEMBLY|INSTALLATION|REMOVAL)[:\s\.-]*",
        r"^(?:OPERATION|START-UP|SHUTDOWN|EMERGENCY\s+STOP)[:\s\.-]*",
        r"^[0-9]+(?:\.[0-9]+)+\s+",  # e.g., 2.1.3 Bearing Inspection
    ]

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> None:
        """Initialize DocumentProcessor with chunking configuration.

        Args:
            chunk_size: Maximum characters per chunk. Defaults to settings.chunk_size.
            chunk_overlap: Character overlap between consecutive chunks.
                Defaults to settings.chunk_overlap.
        """
        settings = get_settings()
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap

        if fitz is None:
            logger.warning(
                "PyMuPDF (fitz) is not installed. PDF parsing will fail until installed."
            )

    def is_section_header(self, text: str, is_bold: bool = False) -> bool:
        """Determine whether a line of text represents a section header.

        Checks font properties (boldness), uppercase conventions, length heuristics,
        and known maintenance manual terminology.

        Args:
            text: Stripped single line of candidate text.
            is_bold: Whether the text span is styled as bold in the PDF font flags.

        Returns:
            True if the line is classified as a section header, False otherwise.
        """
        line = text.strip()
        if not line or len(line) < 3 or len(line) > 100:
            return False

        # Exclude lines that end in standard sentence punctuation
        if line.endswith((".", ",", ";", "...", "—", "-")):
            return False

        # Explicit regex match on industrial section patterns
        for pattern in self._SECTION_PATTERNS:
            if re.search(pattern, line, re.IGNORECASE):
                return True

        # Bold styling combined with short length
        if is_bold and len(line) <= 80:
            return True

        # ALL CAPS line with multiple words (e.g., 'BEARING ALIGNMENT PROCEDURES')
        if line.isupper() and any(c.isalpha() for c in line) and len(line.split()) >= 1:
            return True

        # Numbered headings like '1.2 Visual Inspection' or '3. Maintenance Steps'
        if re.match(r"^\d+(\.\d+)*\s+[A-Z]", line):
            return True

        return False

    def parse_pdf(self, pdf_path: str | Path) -> list[dict[str, Any]]:
        """Extract text and section titles page by page from a PDF file using PyMuPDF.

        Args:
            pdf_path: Path to the target PDF file.

        Returns:
            List of dictionaries per page, each containing:
                - page: 1-indexed page number
                - text: Full text of the page
                - sections: List of detected section dictionaries with title and text

        Raises:
            FileNotFoundError: If the PDF path does not exist.
            RuntimeError: If PyMuPDF is not installed or parsing fails.
        """
        path = Path(pdf_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"PDF file not found: {path}")

        if fitz is None:
            raise RuntimeError("PyMuPDF (fitz) is required to parse PDF files.")

        logger.info(f"Parsing PDF document: {path.name}")
        pages_data: list[dict[str, Any]] = []

        try:
            doc = fitz.open(path)
        except Exception as e:
            logger.error(f"Failed to open PDF file {path}: {e}")
            raise RuntimeError(f"Error opening PDF {path}: {e}") from e

        current_section = "General"

        try:
            for page_index in range(len(doc)):
                page_num = page_index + 1
                page = doc[page_index]

                page_text_blocks: list[str] = []
                page_sections: list[dict[str, Any]] = []

                # Extract text using dict mode to inspect font flags & spans
                try:
                    page_dict = page.get_text("dict")
                    blocks = page_dict.get("blocks", [])
                except Exception as parse_err:
                    logger.warning(
                        f"Error reading block dict on page {page_num} of {path.name}: {parse_err}. "
                        f"Falling back to basic text extraction."
                    )
                    plain_text = page.get_text("text")
                    pages_data.append(
                        {
                            "page": page_num,
                            "text": plain_text,
                            "sections": [{"title": current_section, "text": plain_text}],
                        }
                    )
                    continue

                current_section_lines: list[str] = []

                for block in blocks:
                    if block.get("type") != 0:  # Type 0 is text block
                        continue

                    for line in block.get("lines", []):
                        spans = line.get("spans", [])
                        line_text = "".join(span.get("text", "") for span in spans).strip()
                        if not line_text:
                            continue

                        # Determine if any span in this line is bold
                        # flags & 2 != 0 means bold in PyMuPDF font flags
                        is_bold = any(
                            (span.get("flags", 0) & 2 != 0)
                            or ("bold" in span.get("font", "").lower())
                            or ("black" in span.get("font", "").lower())
                            for span in spans
                        )

                        if self.is_section_header(line_text, is_bold=is_bold):
                            # Save previous section if it has text
                            if current_section_lines:
                                section_content = "\n".join(current_section_lines).strip()
                                if section_content:
                                    page_sections.append(
                                        {"title": current_section, "text": section_content}
                                    )
                                current_section_lines = []

                            current_section = line_text
                            logger.debug(
                                f"Detected section on page {page_num}: '{current_section}'"
                            )
                        else:
                            current_section_lines.append(line_text)

                        page_text_blocks.append(line_text)

                if current_section_lines:
                    section_content = "\n".join(current_section_lines).strip()
                    if section_content:
                        page_sections.append(
                            {"title": current_section, "text": section_content}
                        )

                full_page_text = "\n".join(page_text_blocks).strip()
                pages_data.append(
                    {
                        "page": page_num,
                        "text": full_page_text,
                        "sections": page_sections
                        if page_sections
                        else [{"title": current_section, "text": full_page_text}],
                    }
                )

            logger.info(
                f"Successfully extracted {len(pages_data)} pages from {path.name}"
            )
            return pages_data

        finally:
            doc.close()

    def chunk_text(
        self,
        text: str,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> list[str]:
        """Split text into manageable chunks respecting sentence boundaries.

        Maintains semantic coherence by grouping complete sentences together up
        to chunk_size, while applying sliding-window character overlap.

        Args:
            text: Input text string to chunk.
            chunk_size: Target maximum characters per chunk.
            chunk_overlap: Overlap length in characters between consecutive chunks.

        Returns:
            List of chunk strings.
        """
        max_size = chunk_size or self.chunk_size
        overlap = chunk_overlap or self.chunk_overlap

        cleaned = text.strip()
        if not cleaned:
            return []

        if len(cleaned) <= max_size:
            return [cleaned]

        # Sentence segmentation regex: splits on punctuation followed by whitespace
        sentence_endings = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"\'\(\[])|\n\n+")
        raw_sentences = sentence_endings.split(cleaned)
        sentences: list[str] = [s.strip() for s in raw_sentences if s.strip()]

        if not sentences:
            sentences = [cleaned]

        chunks: list[str] = []
        current_chunk_sentences: list[str] = []
        current_length = 0

        for sentence in sentences:
            sentence_len = len(sentence)

            # If a single sentence exceeds chunk_size, hard-split it by words
            if sentence_len > max_size:
                if current_chunk_sentences:
                    chunk_str = " ".join(current_chunk_sentences).strip()
                    if chunk_str:
                        chunks.append(chunk_str)
                    current_chunk_sentences = []
                    current_length = 0

                words = sentence.split()
                sub_chunk: list[str] = []
                sub_len = 0
                for w in words:
                    if sub_len + len(w) + 1 > max_size and sub_chunk:
                        chunks.append(" ".join(sub_chunk).strip())
                        # Retain trailing words for overlap
                        overlap_words = []
                        overlap_len = 0
                        for rev_w in reversed(sub_chunk):
                            if overlap_len + len(rev_w) + 1 <= overlap:
                                overlap_words.insert(0, rev_w)
                                overlap_len += len(rev_w) + 1
                            else:
                                break
                        sub_chunk = list(overlap_words)
                        sub_len = sum(len(x) + 1 for x in sub_chunk)
                    sub_chunk.append(w)
                    sub_len += len(w) + 1

                if sub_chunk:
                    current_chunk_sentences.append(" ".join(sub_chunk))
                    current_length = len(current_chunk_sentences[-1])
                continue

            # Standard accumulation
            if current_length + sentence_len + 1 > max_size and current_chunk_sentences:
                chunk_str = " ".join(current_chunk_sentences).strip()
                if chunk_str:
                    chunks.append(chunk_str)

                # Collect trailing sentences for overlap
                overlap_sentences: list[str] = []
                accumulated_overlap = 0
                for rev_s in reversed(current_chunk_sentences):
                    if accumulated_overlap + len(rev_s) + 1 <= overlap:
                        overlap_sentences.insert(0, rev_s)
                        accumulated_overlap += len(rev_s) + 1
                    else:
                        break

                current_chunk_sentences = list(overlap_sentences)
                current_length = sum(len(s) + 1 for s in current_chunk_sentences)

            current_chunk_sentences.append(sentence)
            current_length += sentence_len + 1

        if current_chunk_sentences:
            final_chunk = " ".join(current_chunk_sentences).strip()
            if final_chunk and (not chunks or final_chunk != chunks[-1]):
                chunks.append(final_chunk)

        return chunks

    def extract_images(
        self,
        pdf_path: str | Path,
        output_dir: str | Path | None = None,
    ) -> list[Path]:
        """Extract embedded images and diagrams from a maintenance manual PDF.

        Args:
            pdf_path: Path to the PDF file.
            output_dir: Directory where extracted images will be stored.
                If None, saves to 'data/extracted_images/<pdf_stem>'.

        Returns:
            List of Path objects for all saved image files.
        """
        path = Path(pdf_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"PDF file not found: {path}")

        if fitz is None:
            raise RuntimeError("PyMuPDF (fitz) is required to extract images.")

        if output_dir is None:
            settings = get_settings()
            target_dir = Path(settings.data_raw_dir).parent / "extracted_images" / path.stem
        else:
            target_dir = Path(output_dir)

        target_dir.mkdir(parents=True, exist_ok=True)
        saved_images: list[Path] = []

        try:
            doc = fitz.open(path)
        except Exception as e:
            logger.error(f"Cannot open PDF {path} for image extraction: {e}")
            return []

        try:
            image_counter = 0
            for page_idx in range(len(doc)):
                page_num = page_idx + 1
                page = doc[page_idx]

                try:
                    images_on_page = page.get_images(full=True)
                except Exception as img_err:
                    logger.warning(
                        f"Failed to retrieve images from page {page_num} of {path.name}: {img_err}"
                    )
                    continue

                for img_idx, img_info in enumerate(images_on_page):
                    xref = img_info[0]
                    try:
                        base_image = doc.extract_image(xref)
                        if not base_image:
                            continue

                        image_bytes = base_image.get("image")
                        image_ext = base_image.get("ext", "png")

                        if not image_bytes or len(image_bytes) < 1024:
                            # Skip tiny icons, bullets, or decorative lines (<1KB)
                            continue

                        image_counter += 1
                        filename = f"{path.stem}_p{page_num}_img{img_idx + 1}.{image_ext}"
                        out_path = target_dir / filename
                        out_path.write_bytes(image_bytes)
                        saved_images.append(out_path)
                        logger.debug(f"Saved extracted image: {out_path.name}")
                    except Exception as extract_err:
                        logger.warning(
                            f"Could not extract image xref {xref} on page {page_num}: {extract_err}"
                        )
                        continue

            logger.info(
                f"Extracted {len(saved_images)} images from {path.name} into {target_dir}"
            )
            return saved_images

        finally:
            doc.close()

    def process_pdf(self, pdf_path: str | Path) -> list[Document]:
        """Process a single PDF into a list of chunked Document objects.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            List of Document objects ready for embedding and indexing.
        """
        path = Path(pdf_path).resolve()
        pages = self.parse_pdf(path)
        documents: list[Document] = []
        global_chunk_idx = 0

        for page_data in pages:
            page_num = page_data["page"]
            sections = page_data.get("sections", [])

            for section in sections:
                title = section.get("title", "General")
                text = section.get("text", "")
                if not text.strip():
                    continue

                chunks = self.chunk_text(text)
                for chunk in chunks:
                    if not chunk.strip():
                        continue
                    global_chunk_idx += 1
                    doc = Document(
                        content=chunk,
                        metadata={
                            "source": path.name,
                            "source_path": str(path),
                            "page": page_num,
                            "chunk_id": global_chunk_idx,
                            "section_title": title,
                        },
                    )
                    documents.append(doc)

        logger.info(
            f"Processed '{path.name}': generated {len(documents)} document chunks across {len(pages)} pages."
        )
        return documents

    def process_directory(self, dir_path: str | Path) -> list[Document]:
        """Process all PDF files found in a directory.

        Args:
            dir_path: Directory path containing PDF manuals.

        Returns:
            Combined list of Document objects from all processed PDFs.
        """
        target = Path(dir_path).resolve()
        if not target.exists() or not target.is_dir():
            logger.warning(f"Manuals directory does not exist or is not a directory: {target}")
            return []

        pdf_files = sorted(
            [p for p in target.glob("**/*") if p.is_file() and p.suffix.lower() == ".pdf"]
        )

        if not pdf_files:
            logger.warning(f"No PDF files found in directory: {target}")
            return []

        logger.info(f"Found {len(pdf_files)} PDF manuals in {target}")
        all_docs: list[Document] = []

        for pdf_path in pdf_files:
            try:
                docs = self.process_pdf(pdf_path)
                all_docs.extend(docs)
            except Exception as e:
                logger.error(f"Failed to process manual {pdf_path}: {e}")

        logger.info(
            f"Directory processing complete. Total chunks generated: {len(all_docs)}."
        )
        return all_docs


def main() -> None:
    """CLI entry point for processing and indexing maintenance manuals."""
    parser = argparse.ArgumentParser(
        description="Process industrial maintenance manual PDFs into chunked documents for RAG."
    )
    parser.add_argument(
        "--index",
        "-i",
        type=str,
        default=None,
        help="Path to directory containing PDF manuals to process and index.",
    )
    parser.add_argument(
        "--extract-images",
        action="store_true",
        help="Extract embedded images and diagrams from PDFs.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for extracted images.",
    )

    args = parser.parse_args()
    settings = get_settings()

    target_dir = args.index or settings.data_manuals_dir
    processor = DocumentProcessor()

    logger.info(f"Target manual directory: {target_dir}")
    documents = processor.process_directory(target_dir)

    print("\n" + "=" * 60)
    print("DOCUMENT PROCESSING REPORT")
    print("=" * 60)
    print(f"Target Directory: {target_dir}")
    print(f"Total Chunks Generated: {len(documents)}")

    if documents:
        sources = {doc.metadata.get("source") for doc in documents}
        sections = {doc.metadata.get("section_title") for doc in documents}
        print(f"Unique Manuals: {len(sources)} ({', '.join(str(s) for s in sources)})")
        print(f"Sample Sections Detected: {list(sections)[:5]}")
        print("-" * 60)
        print("First chunk preview:")
        print(f"Source: {documents[0].metadata['source']}, Page: {documents[0].metadata['page']}")
        print(f"Section: {documents[0].metadata['section_title']}")
        print(f"Content:\n{documents[0].content[:250]}...")
    print("=" * 60 + "\n")

    if args.extract_images:
        path = Path(target_dir)
        pdf_files = [p for p in path.glob("**/*.pdf") if p.is_file()]
        total_images = 0
        for pdf in pdf_files:
            imgs = processor.extract_images(pdf, output_dir=args.output_dir)
            total_images += len(imgs)
        print(f"Total images extracted: {total_images}")


if __name__ == "__main__":
    main()
