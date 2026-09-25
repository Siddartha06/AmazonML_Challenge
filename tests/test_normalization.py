"""
Unit tests for src/normalization.py — Multi-View Business Entity Normalization System
"""

import os
import sys
import unittest

# Ensure src is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.normalization import (
    normalize_business_name,
    normalize_business_address,
    normalize_country,
    normalize_record,
    normalize_unicode,
    remove_punctuation,
    expand_abbreviations,
    NAME_ABBREVIATIONS,
    ADDRESS_ABBREVIATIONS
)


class TestNormalizationSystem(unittest.TestCase):

    def test_unicode_normalization(self):
        """Test Unicode NFKD decomposition and ASCII accent stripping (French/German/Spanish support)."""
        # French business name & address
        fr_name = "Société Générale S.A.R.L."
        norm_fr = normalize_business_name(fr_name)
        self.assertEqual(norm_fr["name_unicode"], "Societe Generale S.A.R.L.")
        self.assertIn("societe", norm_fr["name_tokens"])

        # German name with umlaut
        de_name = "München Bäckerei GmbH"
        norm_de = normalize_business_name(de_name)
        self.assertEqual(norm_de["name_unicode"], "Munchen Backerei GmbH")

    def test_punctuation_handling(self):
        """Test punctuation removal without smashing words together."""
        text = "Amazon.com, Inc. (Seattle-Branch)"
        norm = normalize_business_name(text)
        self.assertEqual(norm["name_no_punct"], "Amazon com Inc Seattle Branch")
        self.assertIn("amazon", norm["name_tokens"])
        self.assertIn("seattle", norm["name_tokens"])

    def test_whitespace_normalization(self):
        """Test stripping and collapsing multi-space/tab/newline whitespace."""
        text = "   Walmart   Supercenter   \n\t Inc.   "
        norm = normalize_business_name(text)
        self.assertEqual(norm["name_clean_whitespace"], "Walmart Supercenter Inc.")

    def test_ampersand_expansion(self):
        """Test ampersand (&) conversion to 'and'."""
        text = "AT&T Corp & Verizon"
        norm = normalize_business_name(text)
        self.assertEqual(norm["name_no_punct"], "AT and T Corp and Verizon")
        self.assertIn("and", norm["name_tokens"])

    def test_name_abbreviations_and_legal_suffixes(self):
        """Test abbreviation expansion and legal suffix normalization (Corp/Co/Ltd/Pvt/Inc/SARL)."""
        # Corp / Corporation
        n1 = normalize_business_name("Acme Corp.")
        self.assertEqual(n1["name_abbrev_expanded"], "acme corporation")
        self.assertEqual(n1["name_no_legal_suffix"], "acme")

        # Pvt Ltd / Private Limited
        n2 = normalize_business_name("Tata Consultancy Services Pvt. Ltd.")
        self.assertIn("private", n2["name_abbrev_expanded"])
        self.assertIn("limited", n2["name_abbrev_expanded"])
        self.assertEqual(n2["name_no_legal_suffix"], "tata consultancy services")

        # Co / Company
        n3 = normalize_business_name("The Boeing Co.")
        self.assertEqual(n3["name_abbrev_expanded"], "the boeing company")
        self.assertEqual(n3["name_no_legal_suffix"], "the boeing")

        # French SARL
        n4 = normalize_business_name("Bistro Parisien S.A.R.L.")
        self.assertIn("sarl", n4["name_legal_suffix_norm"])
        self.assertEqual(n4["name_no_legal_suffix"], "bistro parisien")

    def test_no_blind_word_removal(self):
        """Test that meaningful words containing abbreviation letters (e.g. 'Costco', 'Include', 'State') are preserved."""
        n1 = normalize_business_name("Costco Wholesale Corp")
        self.assertEqual(n1["name_no_legal_suffix"], "costco wholesale")
        
        n2 = normalize_business_name("Inclusive Logistics Ltd")
        self.assertIn("inclusive", n2["name_tokens"])
        self.assertEqual(n2["name_no_legal_suffix"], "inclusive logistics")

    def test_address_abbreviations_and_numerics(self):
        """Test address abbreviation expansion, numeric extraction, and postal code extraction."""
        addr = "123 N. Main St., Suite 400, Chicago, IL 60601"
        norm = normalize_business_address(addr)

        self.assertEqual(norm["address_abbrev_expanded"], "123 north main street suite 400 chicago il 60601")
        self.assertEqual(norm["address_numeric_tokens"], ["123", "400", "60601"])
        self.assertEqual(norm["address_postal_code"], "60601")
        self.assertIn("chicago", norm["address_region_tokens"])

        # Indian 6-digit PIN address
        addr_in = "Plot 45, MG Road, Koramangala, Bengaluru 560034"
        norm_in = normalize_business_address(addr_in)
        self.assertEqual(norm_in["address_postal_code"], "560034")

        # French 5-digit postal code address
        addr_fr = "15 Rue de la Paix, 75002 Paris"
        norm_fr = normalize_business_address(addr_fr)
        self.assertEqual(norm_fr["address_postal_code"], "75002")

    def test_compact_representations(self):
        """Test compact (alphanumeric-only) representations."""
        n = normalize_business_name("Café & Bar @ 100!")
        self.assertEqual(n["name_compact"], "cafebar100")

        a = normalize_business_address("123-A, 5th Ave.")
        self.assertEqual(a["address_compact"], "123a5thave")

    def test_open_set_country(self):
        """Test open-set country normalization without hardcoding allowed country lists."""
        c1 = normalize_country("  United States  ")
        self.assertEqual(c1["country_clean"], "UNITED STATES")

        c2 = normalize_country("France")
        self.assertEqual(c2["country_clean"], "FRANCE")

        c3 = normalize_country("India")
        self.assertEqual(c3["country_clean"], "INDIA")

        c4 = normalize_country("  Brasil  ")
        self.assertEqual(c4["country_clean"], "BRASIL")

    def test_missing_values_and_empty_strings(self):
        """Test handling of None, empty strings, and whitespace-only strings across all functions."""
        # None inputs
        n_none = normalize_business_name(None)
        self.assertEqual(n_none["original_name"], "")
        self.assertEqual(n_none["name_tokens"], [])

        a_none = normalize_business_address(None)
        self.assertEqual(a_none["original_address"], "")
        self.assertIsNone(a_none["address_postal_code"])

        c_none = normalize_country(None)
        self.assertEqual(c_none["original_country"], "")

        # Full record with missing values
        rec = normalize_record("S2-000001", "Acme Corp", None, None)
        self.assertEqual(rec.entity_id, "S2-000001")
        self.assertEqual(rec.original_name, "Acme Corp")
        self.assertEqual(rec.original_address, "")
        self.assertEqual(rec.original_country, "")


if __name__ == "__main__":
    unittest.main()
