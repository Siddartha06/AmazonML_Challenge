"""
ML Challenge 2026 — Multi-View Business Entity Normalization System

This module provides a robust, multi-representation normalization pipeline for
business names, business addresses, and country strings.

Key Principles:
1. Retain original raw input values without destruction.
2. Generate 10 distinct derived representations for business names.
3. Generate 10 distinct derived representations for business addresses.
4. Normalize country strings as open-set (no hardcoded country lists).
5. Handle edge cases: missing values, None, empty strings, Unicode accents, ampersands, numbers.
"""

import re
import unicodedata
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Union, Tuple


# =============================================================================
# Abbreviations & Legal Suffix Mapping Dictionaries
# =============================================================================

# Name abbreviation expansions (word-boundary safe)
NAME_ABBREVIATIONS = {
    r"\bco\b\.?": "company",
    r"\bcorp\b\.?": "corporation",
    r"\bltd\b\.?": "limited",
    r"\bpvt\b\.?": "private",
    r"\binc\b\.?": "incorporated",
    r"\bllc\b\.?": "limited liability company",
    r"\bllp\b\.?": "limited liability partnership",
    r"\bsarl\b\.?": "societe a responsabilite limitee",
    r"\bsa\b\.?": "societe anonyme",
    r"\bgmbh\b\.?": "gesellschaft mit beschraenkter haftung",
    r"\bplc\b\.?": "public limited company",
    r"&": "and",
}

# Standardized legal suffix tokens
LEGAL_SUFFIX_MAP = {
    "private limited": "pvt ltd",
    "pvt limited": "pvt ltd",
    "private ltd": "pvt ltd",
    "pvt ltd": "pvt ltd",
    "limited": "ltd",
    "ltd": "ltd",
    "corporation": "corp",
    "corp": "corp",
    "incorporated": "inc",
    "inc": "inc",
    "company": "co",
    "co": "co",
    "limited liability company": "llc",
    "llc": "llc",
    "l l c": "llc",
    "limited liability partnership": "llp",
    "llp": "llp",
    "l l p": "llp",
    "societe a responsabilite limitee": "sarl",
    "sarl": "sarl",
    "s a r l": "sarl",
    "societe anonyme": "sa",
    "sa": "sa",
    "s a": "sa",
    "gesellschaft mit beschraenkter haftung": "gmbh",
    "gmbh": "gmbh",
    "g m b h": "gmbh",
    "public limited company": "plc",
    "plc": "plc",
    "p l c": "plc",
}

# List of normalized legal suffix terms for removal (sorted longest to shortest)
COMMON_LEGAL_SUFFIX_TERMS = [
    "private limited",
    "pvt limited",
    "private ltd",
    "pvt ltd",
    "limited liability company",
    "limited liability partnership",
    "societe a responsabilite limitee",
    "gesellschaft mit beschraenkter haftung",
    "public limited company",
    "corporation",
    "incorporated",
    "company",
    "limited",
    "private",
    "corp",
    "inc",
    "ltd",
    "pvt",
    "llc",
    "llp",
    "sarl",
    "gmbh",
    "plc",
    "co",
    "sa",
]

# Address abbreviation expansions (word-boundary safe)
ADDRESS_ABBREVIATIONS = {
    r"\bst\b\.?": "street",
    r"\brd\b\.?": "road",
    r"\bave?\b\.?": "avenue",
    r"\bblvd\b\.?": "boulevard",
    r"\bdr\b\.?": "drive",
    r"\bln\b\.?": "lane",
    r"\bpkwy\b\.?": "parkway",
    r"\bplz?\b\.?": "plaza",
    r"\bste\b\.?": "suite",
    r"\bapt\b\.?": "apartment",
    r"\bfl\b\.?": "floor",
    r"\bbldg\b\.?": "building",
    r"\bdept\b\.?": "department",
    r"\bwy\b\.?": "way",
    r"\bhwy\b\.?": "highway",
    r"\bctr\b\.?": "center",
    r"\bsq\b\.?": "square",
    r"\bw\b\.?": "west",
    r"\be\b\.?": "east",
    r"\bn\b\.?": "north",
    r"\bs\b\.?": "south",
    r"&": "and",
}


# =============================================================================
# Helper Utility Normalization Functions
# =============================================================================

def safe_str(text: Optional[str]) -> str:
    """Ensure input is a non-null string, stripping leading/trailing whitespace."""
    if text is None:
        return ""
    return str(text).strip()


def normalize_unicode(text: str) -> str:
    """Decompose Unicode characters (NFKD) and convert to clean ASCII equivalent if Latin, otherwise retain NFKD string."""
    if not text:
        return ""
    nfkd = unicodedata.normalize("NFKD", text)
    ascii_text = nfkd.encode("ASCII", "ignore").decode("utf-8")
    if ascii_text.strip():
        return ascii_text
    # Fallback for non-Latin scripts (e.g. Devanagari, Cyrillic, CJK) to avoid losing original non-ASCII text
    return nfkd


def normalize_whitespace(text: str) -> str:
    """Strip and collapse consecutive whitespace characters into a single space."""
    if not text:
        return ""
    return " ".join(text.split())


def normalize_acronyms(text: str) -> str:
    """Collapse dots in dotted acronyms (e.g., S.A.R.L. -> SARL, P.V.T. -> PVT, Inc. -> Inc)."""
    if not text:
        return ""
    # Collapse dots between single letters: e.g. S.A.R.L. -> SARL, U.S.A. -> USA
    # Match a single letter followed by dot, repeated
    collapsed = re.sub(r"(?<=\b[a-zA-Z])\.(?=[a-zA-Z]\b|\s|$)", "", text)
    return collapsed


def remove_punctuation(text: str, replace_with_space: bool = True) -> str:
    """Remove punctuation marks from string.
    
    If replace_with_space is True, punctuation is replaced by space to avoid concatenating words.
    Ampersands (&) are converted to 'and' prior to punctuation removal.
    Dotted acronyms (e.g. S.A.R.L.) are collapsed to single tokens (SARL).
    """
    if not text:
        return ""
    # Collapse dotted acronyms first
    t = normalize_acronyms(text)
    # Convert ampersand next
    t = re.sub(r"&", " and ", t)
    if replace_with_space:
        t = re.sub(r"[^\w\s]", " ", t)
    else:
        t = re.sub(r"[^\w\s]", "", t)
    return normalize_whitespace(t)


def expand_abbreviations(text: str, abbrev_dict: Dict[str, str]) -> str:
    """Expand abbreviations using regex word boundary patterns."""
    if not text:
        return ""
    t = text
    for pattern, replacement in abbrev_dict.items():
        t = re.sub(pattern, replacement, t, flags=re.IGNORECASE)
    return normalize_whitespace(t)


def make_compact(text: str) -> str:
    """Convert text to lower-case alphanumeric string with zero spaces or punctuation."""
    if not text:
        return ""
    return re.sub(r"[^a-z0-9]", "", text.lower())


# =============================================================================
# Multi-View Name Normalizer
# =============================================================================

def normalize_business_name(raw_name: Optional[str]) -> Dict[str, Union[str, List[str]]]:
    """Generate 10 distinct derived representations for a business name while retaining original value.
    
    Representations:
    1. original_name
    2. name_lower
    3. name_unicode
    4. name_no_punct
    5. name_clean_whitespace
    6. name_abbrev_expanded
    7. name_legal_suffix_norm
    8. name_compact
    9. name_tokens
    10. name_no_legal_suffix
    11. name_sorted_tokens
    """
    original = safe_str(raw_name)
    if not original:
        return {
            "original_name": "",
            "name_lower": "",
            "name_unicode": "",
            "name_no_punct": "",
            "name_clean_whitespace": "",
            "name_abbrev_expanded": "",
            "name_legal_suffix_norm": "",
            "name_compact": "",
            "name_tokens": [],
            "name_no_legal_suffix": "",
            "name_sorted_tokens": "",
        }

    # 1. name_lower
    name_lower = original.lower()

    # 2. unicode-normalized
    name_unicode = normalize_unicode(original)

    # 3. punctuation-normalized
    name_no_punct = remove_punctuation(name_unicode)

    # 4. whitespace-normalized
    name_clean_whitespace = normalize_whitespace(name_unicode)

    # 5. abbreviation-normalized
    name_abbrev_expanded = expand_abbreviations(name_no_punct.lower(), NAME_ABBREVIATIONS)

    # 6. legal-suffix-normalized
    # Standardize legal suffixes to consistent tokens (e.g., 'private limited' -> 'pvt ltd')
    temp_suffix = name_abbrev_expanded
    for full_form, std_token in LEGAL_SUFFIX_MAP.items():
        pattern = r"\b" + re.escape(full_form) + r"\b"
        temp_suffix = re.sub(pattern, std_token, temp_suffix)
    name_legal_suffix_norm = normalize_whitespace(temp_suffix)

    # 7. compact name
    name_compact = make_compact(name_unicode)

    # 8. tokenized name
    name_tokens = name_no_punct.lower().split()

    # 9. name_without_common_legal_suffix
    # Strip common legal terms from end/tokens of the name
    no_legal = name_abbrev_expanded
    for term in COMMON_LEGAL_SUFFIX_TERMS:
        pattern = r"\b" + re.escape(term) + r"\b"
        no_legal = re.sub(pattern, "", no_legal)
    name_no_legal_suffix = normalize_whitespace(no_legal)
    # Fallback: if stripping legal suffix empties the name, retain original lower
    if not name_no_legal_suffix:
        name_no_legal_suffix = name_abbrev_expanded

    # 10. sorted-token representation
    tokens_no_legal = name_no_legal_suffix.split()
    name_sorted_tokens = " ".join(sorted(tokens_no_legal))

    return {
        "original_name": original,
        "name_lower": name_lower,
        "name_unicode": name_unicode,
        "name_no_punct": name_no_punct,
        "name_clean_whitespace": name_clean_whitespace,
        "name_abbrev_expanded": name_abbrev_expanded,
        "name_legal_suffix_norm": name_legal_suffix_norm,
        "name_compact": name_compact,
        "name_tokens": name_tokens,
        "name_no_legal_suffix": name_no_legal_suffix,
        "name_sorted_tokens": name_sorted_tokens,
    }


# =============================================================================
# Multi-View Address Normalizer
# =============================================================================

def extract_postal_code(address_text: str) -> Optional[str]:
    """Extract postal or PIN code string from address text (US 5-digit, IN 6-digit, FR 5-digit)."""
    if not address_text:
        return None
    
    # 6-digit Indian PIN or 5-digit US/FR ZIP pattern
    in_match = re.search(r"\b\d{6}\b", address_text)
    if in_match:
        return in_match.group(0)
    
    us_fr_match = re.search(r"\b\d{5}(?:-\d{4})?\b", address_text)
    if us_fr_match:
        return us_fr_match.group(0)
        
    return None


def extract_region_tokens(address_text: str) -> List[str]:
    """Derive potential city/state/region tokens from text without external databases."""
    if not address_text:
        return []
    
    # Clean text and split by comma or spaces
    cleaned = remove_punctuation(normalize_unicode(address_text)).lower()
    tokens = cleaned.split()
    
    # Non-numeric trailing tokens (cities/states usually appear towards the end of address strings)
    non_numeric = [t for t in tokens if not t.isdigit() and len(t) > 1]
    if len(non_numeric) >= 2:
        return non_numeric[-3:]  # return last 2-3 non-numeric tokens (e.g. ['paris', 'france'] or ['chicago', 'il'])
    return non_numeric


def normalize_business_address(raw_address: Optional[str]) -> Dict[str, Union[str, List[str], Optional[str]]]:
    """Generate 10 distinct derived representations for a business address while retaining original value.
    
    Representations:
    1. original_address
    2. address_lower
    3. address_unicode
    4. address_no_punct
    5. address_clean_whitespace
    6. address_abbrev_expanded
    7. address_compact
    8. address_tokens
    9. address_numeric_tokens
    10. address_postal_code
    11. address_region_tokens
    """
    original = safe_str(raw_address)
    if not original:
        return {
            "original_address": "",
            "address_lower": "",
            "address_unicode": "",
            "address_no_punct": "",
            "address_clean_whitespace": "",
            "address_abbrev_expanded": "",
            "address_compact": "",
            "address_tokens": [],
            "address_numeric_tokens": [],
            "address_postal_code": None,
            "address_region_tokens": [],
        }

    # 1. lower-case
    address_lower = original.lower()

    # 2. unicode-normalized
    address_unicode = normalize_unicode(original)

    # 3. punctuation-normalized
    address_no_punct = remove_punctuation(address_unicode)

    # 4. whitespace-normalized
    address_clean_whitespace = normalize_whitespace(address_unicode)

    # 5. abbreviation-normalized
    address_abbrev_expanded = expand_abbreviations(address_no_punct.lower(), ADDRESS_ABBREVIATIONS)

    # 6. compact address
    address_compact = make_compact(address_unicode)

    # 7. tokenized address
    address_tokens = address_no_punct.lower().split()

    # 8. numeric-token representation
    address_numeric_tokens = re.findall(r"\b\d+\b", original)

    # 9. postal/PIN-code extraction
    address_postal_code = extract_postal_code(original)

    # 10. region tokens
    address_region_tokens = extract_region_tokens(original)

    return {
        "original_address": original,
        "address_lower": address_lower,
        "address_unicode": address_unicode,
        "address_no_punct": address_no_punct,
        "address_clean_whitespace": address_clean_whitespace,
        "address_abbrev_expanded": address_abbrev_expanded,
        "address_compact": address_compact,
        "address_tokens": address_tokens,
        "address_numeric_tokens": address_numeric_tokens,
        "address_postal_code": address_postal_code,
        "address_region_tokens": address_region_tokens,
    }


# =============================================================================
# Open-Set Country Normalizer
# =============================================================================

def normalize_country(raw_country: Optional[str]) -> Dict[str, str]:
    """Normalize country string as open-set (whitespace/case normalized) without hardcoding allowed countries."""
    original = safe_str(raw_country)
    country_clean = normalize_whitespace(normalize_unicode(original)).upper()
    return {
        "original_country": original,
        "country_clean": country_clean,
    }


# =============================================================================
# Full Record Normalizer Dataclass & Function
# =============================================================================

@dataclass
class NormalizedRecord:
    entity_id: str
    original_name: str
    original_address: str
    original_country: str
    
    # Name representations
    name_lower: str
    name_unicode: str
    name_no_punct: str
    name_clean_whitespace: str
    name_abbrev_expanded: str
    name_legal_suffix_norm: str
    name_compact: str
    name_tokens: List[str]
    name_no_legal_suffix: str
    name_sorted_tokens: str
    
    # Address representations
    address_lower: str
    address_unicode: str
    address_no_punct: str
    address_clean_whitespace: str
    address_abbrev_expanded: str
    address_compact: str
    address_tokens: List[str]
    address_numeric_tokens: List[str]
    address_postal_code: Optional[str]
    address_region_tokens: List[str]
    
    # Country representations
    country_clean: str


def normalize_record(
    entity_id: str,
    name: Optional[str],
    address: Optional[str],
    country: Optional[str]
) -> NormalizedRecord:
    """Normalize a complete business entity record across name, address, and country views."""
    name_views = normalize_business_name(name)
    addr_views = normalize_business_address(address)
    ctry_views = normalize_country(country)

    return NormalizedRecord(
        entity_id=safe_str(entity_id),
        **name_views,
        **addr_views,
        **ctry_views
    )
