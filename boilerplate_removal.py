import json
import re

def clean_boilerplate_from_text(text: str) -> str:
    """
    Filters out non-content noise including headers, footers, cookie banners,
    navigation menus, repeated sidebars, and empty whitespace formatting.
    """
    if not text:
        return ""
    
    # 1. Split text line by line to evaluate structural components
    lines = text.split('\n')
    cleaned_lines = []
    
    # Common boilerplate noise patterns
    boilerplate_patterns = [
        r'^\s*cookie\s*(policy|settings)?\b',
        r'^\s*accept\s*(all\s*)?cookies?\b',
        r'^\s*privacy\s*policy\b',
        r'^\s*terms\s*(of\s*service|of\s*use|and\s*conditions)\b',
        r'^\s*all\s*rights\s*reserved\b',
        r'^\s*copyright\s*©?\b',
        r'^\s*skip\s*to\s*(main\s*)?content\b',
        r'^\s*(home|about\s*us|contact\s*us|login|sign\s*up|menu|navigation)\s*$',
        r'^\s*page\s*\d+\s*(of\s*\d+)?\s*$',  # Repeated PDF page headers/footers
        r'^\s*table\s*of\s*contents\b',
        r'^\s*share\s*on\s*(facebook|twitter|linkedin)\b'
    ]

    for line in lines:
        stripped = line.strip()
        
        # Check if line matches any boilerplate regular expressions
        is_boilerplate = any(re.search(pattern, stripped, re.IGNORECASE) for pattern in boilerplate_patterns)
        
        # Keep only lines that carry core informational value
        if not is_boilerplate and len(stripped) > 0:
            cleaned_lines.append(stripped)

    # 2. Rejoin valid content lines
    cleaned_text = '\n'.join(cleaned_lines)
    
    # 3. Structural whitespace normalization
    cleaned_text = re.sub(r'\n{3,}', '\n\n', cleaned_text)  # Normalize excessive line breaks
    cleaned_text = re.sub(r'[ \t]+', ' ', cleaned_text)     # Standardize inline whitespace
    
    return cleaned_text.strip()


def process_json_knowledge_base(input_json_path: str, output_json_path: str):
    """
    Loads JSON records, applies boilerplate cleaning to the content field,
    and writes out the cleansed records to a new JSON file.
    """
    # Load raw extracted JSON records
    with open(input_json_path, 'r', encoding='utf-8') as f:
        records = json.load(f)

    cleaned_records = []
    for record in records:
        cleaned_record = record.copy()
        
        # Filter content field
        original_content = record.get("content", "")
        cleaned_record["content"] = clean_boilerplate_from_text(original_content)
        
        cleaned_records.append(cleaned_record)

    # Save scrubbed records to new JSON file
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(cleaned_records, f, indent=2, ensure_ascii=False)

    print(f"Boilerplate Removal Complete!")
    print(f"Processed Records: {len(cleaned_records)}")
    print(f"Cleaned Output Saved To: {output_json_path}")


if __name__ == "__main__":
    process_json_knowledge_base(
        input_json_path="business_loans_knowledge_base.json",
        output_json_path="business_loans_knowledge_base_cleaned.json"
    )