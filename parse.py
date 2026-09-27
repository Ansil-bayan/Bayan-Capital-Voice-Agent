import io
import re
import json
import uuid
import logging
from datetime import datetime, timezone

import requests
import pdfplumber
from pypdf import PdfReader
import pandas as pd
from unstructured.partition.html import partition_html
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import spacy

# Fast Near-Deduplication Engine
from datasketch import MinHash, MinHashLSH

# ---------------------------------------------------------------------
# LOGGING & SPACY INITIALIZATION
# ---------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    logging.warning("Downloading spaCy model 'en_core_web_sm'...")
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")


# ---------------------------------------------------------------------
# RESILIENT HTTP SESSION SETUP
# ---------------------------------------------------------------------
def get_resilient_session():
    session = requests.Session()
    retries = Retry(
        total=3,
        backoff_factor=1.5,
        status_forcelist=[429, 500, 502, 503, 504],
        raise_on_status=False
    )
    session.mount("http://", HTTPAdapter(max_retries=retries))
    session.mount("https://", HTTPAdapter(max_retries=retries))
    return session

session = get_resilient_session()


# =====================================================================
# ULTRA-FAST NEAR-DEDUPLICATION ENGINE (MINHASH + LSH)
# =====================================================================

def create_minhash(text: str, num_perm: int = 128, shingle_size: int = 3) -> MinHash:
    """Creates a MinHash object from 3-word shingles of the input text."""
    m = MinHash(num_perm=num_perm)
    words = text.lower().split()
    
    # Create word n-grams (shingles)
    if len(words) < shingle_size:
        m.update(text.lower().encode('utf-8'))
    else:
        for i in range(len(words) - shingle_size + 1):
            shingle = " ".join(words[i:i + shingle_size])
            m.update(shingle.encode('utf-8'))
            
    return m


def deduplicate_records_minhash(records: list[dict], threshold: float = 0.85) -> tuple[list[dict], int]:
    """
    Identifies and removes near-duplicate text chunks using MinHash LSH.
    Threshold ~0.85 Jaccard Similarity corresponds to high text overlap.
    """
    if not records:
        return records, 0

    logging.info(f"[MinHash LSH Deduplication] Scannning {len(records)} records (Threshold Jaccard >= {threshold})...")
    
    lsh = MinHashLSH(threshold=threshold, num_perm=128)
    unique_records = []
    failed_records = []
    duplicates_removed_count = 0

    for idx, record in enumerate(records):
        # Pass failed extractions directly without indexing
        if record.get("status") == "EXTRACTION_FAILED":
            failed_records.append(record)
            continue

        content = record.get("content", "")
        minhash = create_minhash(content, num_perm=128, shingle_size=3)

        # Query LSH index for existing near-duplicates
        result = lsh.query(minhash)

        if not result:
            # Unique content: insert into LSH index and store record
            record_key = f"rec_{idx}_{record['record_id']}"
            lsh.insert(record_key, minhash)
            unique_records.append(record)
        else:
            duplicates_removed_count += 1
            logging.info(f"  [Duplicate Removed] Chunk '{record['record_id']}' matched near-duplicate key '{result[0]}'. Skipped.")

    final_knowledge_base = unique_records + failed_records
    return final_knowledge_base, duplicates_removed_count


# =====================================================================
# PII IDENTIFICATION & REDACTION ENGINE
# =====================================================================

class PIIRedactor:
    REGEX_PATTERNS = {
        "[REDACTED SSN]": r'\b\d{3}-\d{2}-\d{4}\b',
        "[REDACTED CREDIT CARD]": r'\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|6(?:011|5[0-9]{2})[0-9]{12})\b',
        "[REDACTED EMAIL]": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        "[REDACTED PHONE]": r'\b(?:\+?1[-. ]?)?\(?\d{3}\)?[-. ]?\d{3}[-. ]?\d{4}\b',
        "[REDACTED EIN]": r'\b\d{2}-\d{7}\b'
    }

    @classmethod
    def sanitize_text(cls, text: str) -> tuple[str, bool, list[str]]:
        if not text:
            return text, False, []

        pii_detected = False
        redacted_types = []
        cleaned_text = text

        for replacement_label, pattern in cls.REGEX_PATTERNS.items():
            if re.search(pattern, cleaned_text):
                pii_detected = True
                redacted_types.append(replacement_label)
                cleaned_text = re.sub(pattern, replacement_label, cleaned_text)

        doc = nlp(cleaned_text)
        entities_to_redact = sorted(doc.ents, key=lambda e: e.start_char, reverse=True)

        for ent in entities_to_redact:
            if ent.label_ == "PERSON" and len(ent.text.strip()) > 2:
                pii_detected = True
                redacted_types.append("[REDACTED NAME]")
                start, end = ent.start_char, ent.end_char
                cleaned_text = cleaned_text[:start] + "[REDACTED NAME]" + cleaned_text[end:]

            elif ent.label_ in ["GPE", "LOC", "FAC"] and any(char.isdigit() for char in ent.text):
                pii_detected = True
                redacted_types.append("[REDACTED ADDRESS]")
                start, end = ent.start_char, ent.end_char
                cleaned_text = cleaned_text[:start] + "[REDACTED ADDRESS]" + cleaned_text[end:]

        return cleaned_text, pii_detected, list(set(redacted_types))


# =====================================================================
# DATA VALIDATION AUDIT
# =====================================================================

def validate_content(content: str) -> tuple[bool, list[str]]:
    errors = []
    if len(content.strip()) < 40:
        errors.append("MISSING_MANDATORY_TEXT: Character count under threshold (<40 chars).")
        
    non_ascii_ratio = len(re.findall(r'[^\x00-\x7F]', content)) / max(len(content), 1)
    if non_ascii_ratio > 0.30:
        errors.append("BROKEN_FORMATTING: High non-ASCII noise ratio detected.")
        
    return len(errors) == 0, errors


# =====================================================================
# 1. WEB EXTRACTION (UNSTRUCTURED)
# =====================================================================

def extract_website_content(url: str, category: str) -> dict:
    logging.info(f"[Web Scraper] Processing: {url}")
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    parent_doc_id = f"doc_{uuid.uuid4().hex[:6]}"

    try:
        response = session.get(url, headers=headers, timeout=15)
        if response.status_code != 200:
            logging.error(f"Failed HTTP check for {url}: {response.status_code}")
            return {
                "record_id": f"kb_err_{uuid.uuid4().hex[:8]}",
                "parent_doc_id": parent_doc_id,
                "title": "Broken URL Error",
                "content": f"HTTP status code {response.status_code} returned when fetching page.",
                "category": category,
                "source": url,
                "source_type": "website",
                "status": "EXTRACTION_FAILED",
                "extraction_errors": [f"HTTP_{response.status_code}"]
            }

        elements = partition_html(text=response.text)
        cleaned_blocks = []
        
        for el in elements:
            text = str(el).strip()
            if len(text) > 25 and not re.search(r"cookie|privacy policy|terms of service|copyright", text, re.I):
                cleaned_blocks.append(text)

        full_content = "\n\n".join(cleaned_blocks)
        sanitized_content, has_pii, pii_types = PIIRedactor.sanitize_text(full_content)
        is_valid, validation_errors = validate_content(sanitized_content)

        return {
            "record_id": f"kb_web_{uuid.uuid4().hex[:8]}",
            "parent_doc_id": parent_doc_id,
            "title": str(elements[0].text) if elements else "Web Content",
            "content": sanitized_content,
            "category": category,
            "source": url,
            "source_type": "website",
            "version": "1.0",
            "pii_flag": has_pii,
            "redacted_pii_types": pii_types,
            "status": "PROCESSED" if is_valid else "FLAGGED_FOR_REVIEW",
            "extraction_errors": validation_errors,
            "created_at": datetime.now(timezone.utc).isoformat()
        }

    except Exception as e:
        logging.error(f"Error extracting website {url}: {e}")
        return {
            "record_id": f"kb_err_{uuid.uuid4().hex[:8]}",
            "parent_doc_id": parent_doc_id,
            "title": "Web Scrape Exception",
            "content": f"Runtime error: {str(e)}",
            "category": category,
            "source": url,
            "source_type": "website",
            "status": "EXTRACTION_FAILED",
            "extraction_errors": [str(e)]
        }


# =====================================================================
# 2. PDF EXTRACTION (PDFPLUMBER + PYPDF FALLBACK)
# =====================================================================

def parse_pdf_page_with_fallbacks(pdf_bytes: bytes, page_num: int) -> str:
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            page = pdf.pages[page_num - 1]
            page_text = page.extract_text() or ""
            
            tables = page.extract_tables()
            table_str = ""
            if tables:
                for idx, table in enumerate(tables):
                    df = pd.DataFrame(table).fillna("")
                    df = df.replace(r'^\s*$', None, regex=True).dropna(how="all")
                    if not df.empty:
                        table_str += f"\n\n--- Table {idx+1} (Page {page_num}) ---\n"
                        table_str += df.to_markdown(index=False) + "\n"

            combined = f"{page_text}\n{table_str}".strip()
            if len(combined) >= 40:
                return combined
            
            logging.warning(f"pdfplumber yielded low character count on page {page_num}. Triggering pypdf fallback.")
    except Exception as e:
        logging.warning(f"pdfplumber exception on page {page_num}: {e}. Triggering pypdf fallback.")

    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        fallback_text = reader.pages[page_num - 1].extract_text() or ""
        if len(fallback_text.strip()) > 0:
            return fallback_text.strip()
    except Exception as e:
        logging.error(f"pypdf fallback failed on page {page_num}: {e}")

    return "[UNREADABLE PAGE: Content extraction failed across primary and fallback engines]"


def extract_pdf_content(pdf_url: str, category: str) -> list[dict]:
    logging.info(f"[PDF Extractor] Processing: {pdf_url}")
    headers = {"User-Agent": "Mozilla/5.0"}
    parent_doc_id = f"doc_{uuid.uuid4().hex[:6]}"
    records = []

    try:
        response = session.get(pdf_url, headers=headers, timeout=20)
        if response.status_code != 200:
            logging.error(f"Failed HTTP response for PDF: {response.status_code}")
            return [{
                "record_id": f"kb_err_{uuid.uuid4().hex[:8]}",
                "parent_doc_id": parent_doc_id,
                "title": "Broken PDF Link",
                "content": f"HTTP Error {response.status_code}",
                "category": category,
                "source": pdf_url,
                "source_type": "pdf",
                "status": "EXTRACTION_FAILED",
                "extraction_errors": [f"HTTP_{response.status_code}"]
            }]
        pdf_bytes = response.content
    except Exception as e:
        return [{
            "record_id": f"kb_err_{uuid.uuid4().hex[:8]}",
            "parent_doc_id": parent_doc_id,
            "title": "Network Error",
            "content": f"Failed download: {e}",
            "category": category,
            "source": pdf_url,
            "source_type": "pdf",
            "status": "EXTRACTION_FAILED",
            "extraction_errors": [str(e)]
        }]

    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        total_pages = len(reader.pages)
    except Exception as e:
        logging.error(f"Corrupted PDF header at {pdf_url}: {e}")
        return [{
            "record_id": f"kb_err_{uuid.uuid4().hex[:8]}",
            "parent_doc_id": parent_doc_id,
            "title": "Corrupted PDF Error",
            "content": "Document binary header is unparseable or corrupted.",
            "category": category,
            "source": pdf_url,
            "source_type": "pdf",
            "status": "EXTRACTION_FAILED",
            "extraction_errors": ["CORRUPTED_PDF_BINARY"]
        }]

    for page_num in range(1, total_pages + 1):
        raw_content = parse_pdf_page_with_fallbacks(pdf_bytes, page_num)
        sanitized_content, has_pii, pii_types = PIIRedactor.sanitize_text(raw_content)
        is_valid, validation_errors = validate_content(sanitized_content)

        records.append({
            "record_id": f"kb_pdf_{uuid.uuid4().hex[:8]}",
            "parent_doc_id": parent_doc_id,
            "title": f"Page {page_num} - Policy & Underwriting Data",
            "content": sanitized_content,
            "category": category,
            "source": f"{pdf_url}#page={page_num}",
            "source_type": "pdf",
            "version": "1.0",
            "pii_flag": has_pii,
            "redacted_pii_types": pii_types,
            "status": "PROCESSED" if is_valid else "FLAGGED_FOR_REVIEW",
            "extraction_errors": validation_errors,
            "created_at": datetime.now(timezone.utc).isoformat()
        })

    return records


# =====================================================================
# 3. PIPELINE EXECUTION & JSON EXPORT
# =====================================================================

def main():
    raw_knowledge_base = []

    web_sources = [
        {"url": "https://www.sba.gov/loans/7a-loans/#am-i-eligible", "category": "/policy/eligibility"},
        {"url": "https://ramp.com/blog/business-loan-underwriting-process", "category": "/underwriting/guidelines"},
        {"url": "https://newitymarket.com/business-insights/business-loans/sba-7a-eligibility-updates-minimum-requirements-to-apply-for-7a/", "category": "/faq/qualification"}
    ]

    pdf_sources = [
        {"url": "https://seedcorp.com/images/SBA_7a_Lender_TrainingPowerPoint.pdf", "category": "/policy/guaranty_rules"},
        {"url": "https://bankparagon.com/wp-content/uploads/2018/02/7a_SBA-Loan-Application2018.pdf", "category": "/forms/borrower_application"},
        {"url": "https://www.sba7a.loans/downloads/SBA7a-loan-application-checklist.pdf", "category": "/checklist/prequalification"}
    ]

    # 1. Extraction Phase
    for site in web_sources:
        rec = extract_website_content(site["url"], site["category"])
        if rec:
            raw_knowledge_base.append(rec)

    for pdf in pdf_sources:
        recs = extract_pdf_content(pdf["url"], pdf["category"])
        raw_knowledge_base.extend(recs)

    logging.info(f"Extraction Phase Complete. Total raw chunks extracted: {len(raw_knowledge_base)}")

    # 2. Fast MinHash LSH Near-Deduplication Phase (Jaccard Threshold >= 0.85)
    clean_knowledge_base, duplicates_removed = deduplicate_records_minhash(
        raw_knowledge_base, 
        threshold=0.85
    )

    # 3. Save Final Clean Knowledge Base JSON
    output_filename = "business_loans_knowledge_base.json"
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(clean_knowledge_base, f, indent=2, ensure_ascii=False)

    print("\n" + "="*60)
    print("PIPELINE EXECUTION COMPLETE!")
    print(f"Raw Extracted Chunks: {len(raw_knowledge_base)}")
    print(f"Near-Duplicates Removed (MinHash LSH): {duplicates_removed}")
    print(f"Final Deduplicated Records Saved: {len(clean_knowledge_base)}")
    print(f"JSON File Location: {output_filename}")
    print("="*60)

if __name__ == "__main__":
    main()
