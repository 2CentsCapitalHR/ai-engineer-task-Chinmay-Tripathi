import os
import sys
import tempfile
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
        try:
            import spacy
            print("--spaCy imported successfully")
        except ImportError:
            print("--spaCy not available (optional for Phase 2)")
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

def test_ml_dependencies():
    print("\n--Testing ML dependencies...")
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        vectorizer = TfidfVectorizer(max_features=100)
        test_docs = [
            "Articles of Association for company governance",
            "Memorandum of Association for company objects",
            "Board resolution for director appointments",
        ]
        X = vectorizer.fit_transform(test_docs)
        print(f"--TF-IDF vectorization works: {X.shape} feature matrix")
        from sklearn.naive_bayes import MultinomialNB
        nb = MultinomialNB()
        y = ['articles', 'memorandum', 'resolution']
        nb.fit(X, y)
        predictions = nb.predict(X)
        print(f"--Naive Bayes classification works: {predictions}")
        return True
    except Exception as e:
        print(f"!!ML dependencies test failed: {e}")
        return False

def test_gradio_interface():
    print("\n---Testing Gradio interface...")
    try:
        import gradio as gr
        def mock_analyze(files, method):
            return {"test": "data"}, {"summary": "test"}, "Processing complete"
        with gr.Blocks() as demo:
            file_input = gr.File(file_types=[".docx"], file_count="multiple")
            method_dropdown = gr.Dropdown(choices=["rule_based", "ensemble"])
            analyze_button = gr.Button("Analyze")
            results_json = gr.JSON()
            summary_json = gr.JSON()
            log_text = gr.Textbox()
            analyze_button.click(
                fn=mock_analyze,
                inputs=[file_input, method_dropdown],
                outputs=[results_json, summary_json, log_text]
            )
        print("-- Gradio interface creation successful")
        return True
    except Exception as e:
        print(f"!! Gradio interface test failed: {e}")
        return False

def test_file_operations():
    print("\n--Testing file operations...")
    try:
        with tempfile.NamedTemporaryFile(suffix='.docx', delete=False) as tmp:
            tmp_path = tmp.name
            print(f"--Temporary file creation works: {tmp_path}")
        test_path = Path(tmp_path)
        print(f"--Path operations work: {test_path.exists()}")
        os.unlink(tmp_path)
        import json
        test_data = {
            "timestamp": datetime.now().isoformat(),
            "results": [{"type": "test", "confidence": 0.95}],
            "nested": {"analysis": {"score": 85.5}}
        }
        json_str = json.dumps(test_data, indent=2)
        parsed_data = json.loads(json_str)
        print(f"--JSON serialization works: {len(parsed_data)} keys")
        return True
    except Exception as e:
        print(f"!!File operations test failed: {e}")
        return False

def test_phase2_integration():
    print("\n--Testing Phase 2 component integration...")
    try:
        from parser import DocumentParser
        from classifier import DocumentClassifier
        parser = DocumentParser()
        classifier = DocumentClassifier()
        mock_parsed_content = {
            'metadata': {
                'filename': 'test_articles.docx',
                'file_size': 1024,
                'parsing_timestamp': datetime.now().isoformat(),
                'parser_version': '_v2.0'
            },
            'structure': {
                'paragraphs': [
                    {'text': 'Articles of Association', 'is_header': True, 'content_type': 'title_header'},
                    {'text': 'The company shall be governed by these articles.', 'is_header': False, 'content_type': 'body_paragraph'}
                ],
                'sections': [{'title': 'Articles of Association', 'section_type': 'article'}],
                'document_structure': {'has_title': True, 'has_sections': True}
            },
            'statistics': {
                'word_count': 150,
                'paragraph_count': 2,
                'section_count': 1,
                'sentence_count': 8
            },
            'analysis': {
                'text_quality': {'clarity_score': 0.8},
                'legal_elements': {'legal_patterns': {'obligations': {'count': 2}}, 'compliance_indicators': []},
                'readability': {'complexity_score': 45.0},
                'document_complexity': {'overall_complexity': 'medium'}
            },
            'content': {
                'full_text': 'Articles of Association. The company shall be governed by these articles.',
                'key_phrases': ['Articles of Association', 'company governance'],
                'named_entities': [{'text': 'Articles of Association', 'label': 'DOCUMENT'}]
            }
        }
        classification_result = classifier.classify_document(mock_parsed_content, method='rule_based')
        print(f"--Integration test passed: {classification_result['predicted_type']} classified")
        print(f"--Confidence score: {classification_result['confidence']:.2f}")
        print(f"--Method used: {classification_result['method_used']}")
        explanation = classifier.get_classification_explanation(classification_result)
        print(f"--Explanation generated: {len(explanation)} characters")
        return True
    except Exception as e:
        print(f"!!Phase 2 integration test failed: {e}")
        return False

def run_phase2_setup_test():
    print("-->ADGM Corporate Agent - Phase 2 Setup Test")
    print("=" * 60)
    all_tests_passed = True
    tests = [
        ("Imports", test_imports),
        ("Parser", test_parser),
        ("Classifier", test_classifier),
        ("ML Dependencies", test_ml_dependencies),
        ("Gradio Interface", test_gradio_interface),
        ("File Operations", test_file_operations),
        ("Phase 2 Integration", test_phase2_integration),
        ("KB and Compliance", test_kbc)
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
        print("Yay!! All Phase 2 tests passed! Your system is ready.")
        print("\n--> Phase 2 Features Available:")
        print(" • document parsing with structure analysis")
        print(" • ML-based classification")
        print(" • Detailed content analysis and quality scoring")
        print(" • Compliance indicators and issue detection")
        print(" • Comprehensive analytics and reporting")
    else:
        print("!! Some Phase 2 tests failed. Please check the errors above.")
        print("\n# Troubleshooting:")
        print("1. Install Phase 2 dependencies: pip install -r requirements_v2.txt")
        print("2. Ensure you're in the virtual environment")
        print("3. Check that the src/ directory and files are created")
        print("4. For spaCy issues, try: pip install spacy && python -m spacy download en_core_web_sm")
        print("\n** If you need help, refer to the updated documentation.")

if __name__ == "__main__":
    run_phase2_setup_test()

