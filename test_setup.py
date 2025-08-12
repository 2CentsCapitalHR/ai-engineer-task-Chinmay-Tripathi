import os, sys, tempfile
from pathlib import Path
from datetime import datetime
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    print("-- Testing Phase 2 imports...")
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.naive_bayes import MultinomialNB
        print("--scikit-learn imported successfully")
        from parser import DocumentParser
        print("--DocumentParser imported successfully")
        from classifier import DocumentClassifier
        print("--DocumentClassifier imported successfully")
        import spacy
        print("--spaCy imported successfully")
        return True
    except ImportError as e:
        print(f"--Import error: {e}")
        return False

def test_kbc():
    print("\n--Testing KnowledgeBase and Compliance Checker components...")
    try:
        from classifier import ADGMKnowledgeBase, ComplianceChecker, RedFlagDetector
        kb = ADGMKnowledgeBase()
        cc = ComplianceChecker()
        rf = RedFlagDetector()
        assert isinstance(cc.required_types, list)
        assert callable(rf.scan)
        print("--Phase 3 classes loaded & working")
        return True
    except Exception as e:
        print(f"!!Phase 3 import failed: {e}")
        return False

def test_parser():
    print("\n--Testing document parser...")
    try:
        from parser import DocumentParser
        parser = DocumentParser()
        print("--DocumentParser initialized successfully")
        print(f"--Supported formats: {parser.supported_formats}")
        print(f"--Document types configured: {len(parser.type_indicators)}")
        print(f"--Section patterns loaded: {len(parser.section_patterns)}")
        test_text = "Article 1: Company Name. The company shall be known as Test Corporation Limited."
        matches = parser._matches_section_pattern(test_text)
        print(f"--Pattern matching works: {matches}")
        return True
    except Exception as e:
        print(f"-- parser test failed: {e}")
        return False

def test_classifier():
    print("\n--Testing document classifier...")
    try:
        from classifier import DocumentClassifier
        classifier = DocumentClassifier()
        print("--DocumentClassifier initialized successfully")
        print(f"--Document types supported: {len(classifier.document_types)}")
        print(f"--ADGM patterns configured: {len(classifier.adgm_patterns)}")
        print(f"--ML classifiers available: {list(classifier.classifiers.keys())}")
        mock_content = {
            'content': {'full_text': 'Articles of Association for Test Company Limited'},
            'structure': {'paragraphs': [], 'sections': []},
            'statistics': {'word_count': 10, 'table_count': 0}
        }
        features = classifier.extract_features(mock_content)
        print(f"--Feature extraction works: {len(features)} features extracted")
        result = classifier.classify_document(mock_content, method='rule_based')
        print(f"--Classification works: {result['predicted_type']} with {result['confidence']:.2f} confidence")
        return True
    except Exception as e:
        print(f"!! classifier test failed: {e}")
        return False

def test_docx_annotator():
    print("--Testing DocxAnnotator...")
    try:
        from classifier import DocxAnnotator
        from docx import Document
        import tempfile, os
        fd, temp_doc = tempfile.mkstemp(suffix=".docx")
        os.close(fd)
        doc = Document()
        doc.add_paragraph("This is a test.")
        doc.add_paragraph("UAE Federal Courts is a wrong phrase.")
        doc.save(temp_doc)
        da = DocxAnnotator()
        review_comments = [{
            "paragraph_index": 1,
            "text": "UAE Federal Courts is a wrong phrase.",
            "comment": "Incorrect jurisdiction reference – Replace with ADGM Courts (Severity: High)"
        }]
        marked_file = da.add_comments(temp_doc, review_comments)
        assert os.path.exists(marked_file)
        print(f"--DocxAnnotator test passed. Marked file: {marked_file}")
        os.remove(temp_doc)
        os.remove(marked_file)
        return True
    except Exception as e:
        print(f"!!DocxAnnotator test failed: {e}")
        return False

def run_phase2_setup_test():
    print("-->ADGM Corporate Agent - Setup Test")
    print("=" * 60)
    all_tests_passed = True
    tests = [
        ("Imports", test_imports),
        ("Parser", test_parser),
        ("Classifier", test_classifier),
        ("KB and Compliance", test_kbc),
        ("Docx Annotator", test_docx_annotator),
    ]
    for test_name, test_func in tests:
        print(f"\n-->>Running {test_name} test...")
        try:
            result = test_func()
            if not result:
                all_tests_passed = False
        except Exception as e:
            print(f"!! {test_name} test failed with exception: {e}")
            all_tests_passed = False
    print("\n" + "=" * 60)
    if all_tests_passed:
        print("Yay!! All tests passed! Your system is ready.")
        print("\n---> Features Available:")
        print(" • Document upload/parse/classify/flag/check/annotate")
        print(" • JSON summary and marked file download")
    else:
        print("!! Some tests failed. Please check the errors above.")
        print("\nTroubleshooting: Ensure your docx library is installed and your src/ directory is set.")

if __name__ == "__main__":
    run_phase2_setup_test()

