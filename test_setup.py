import os
import sys
from pathlib import Path

def test_imports():
    print("\nTesting module imports...")
    
    try:
        import gradio as gr
        print("--Gradio imported successfully")
        
        from docx import Document
        print("--python-docx imported successfully")
        
        import pandas as pd
        print("--pandas imported successfully")
        
        import numpy as np
        print("--numpy imported successfully")
        
        from document_parser import DocumentParser
        print("--DocumentParser imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"!!Import error: {e}")
        return False

def test_document_parser():
    print("\n--Testing document parser...")
    
    try:
        from document_parser import DocumentParser
        parser = DocumentParser()
        
        print("--DocumentParser initialized successfully")
        print(f"-->Supported formats: {parser.supported_formats}")
        
        return True
    except Exception as e:
        print(f"!!DocumentParser test failed: {e}")
        return False

def test_environment():
    print("\n--Testing environment setup...")
    
    # Python version
    python_version = sys.version
    print(f">>Python version: {python_version}")
    
    # Directory structure
    required_dirs = ["src", "data", "tests", "docs"]
    for dir_name in required_dirs:
        dir_path = Path(dir_name)
        if dir_path.exists():
            print(f"-->Directory '{dir_name}' exists")
        else:
            print(f"!!!Directory '{dir_name}' not found - you may need to create it")
    
    return True

def test_gradio_interface():
    print("\n-->Testing Gradio interface...")
    
    try:
        import gradio as gr
        
        def dummy_function(text):
            return f"Echo: {text}"
        
        with gr.Blocks() as demo:
            gr.Textbox(label="Test Input")
            gr.Textbox(label="Test Output")
        
        print("--- Gradio interface creation successful")
        return True
    except Exception as e:
        print(f"!!! Gradio interface test failed: {e}")
        return False

def main():
    print("### Setup Test")
    print("=" * 40)
    
    all_tests_passed = True
    
    # Run tests
    tests = [
        ("Module Imports", test_imports),
        ("Document Parser", test_document_parser),
        ("Environment", test_environment),
        ("Gradio Interface", test_gradio_interface)
    ]
    
    for test_name, test_func in tests:
        print(f"\n--> Running {test_name} test...")
        try:
            result = test_func()
            if not result:
                all_tests_passed = False
        except Exception as e:
            print(f"!!! {test_name} test failed with exception: {e}")
            all_tests_passed = False
    
    # Final result
    print("\n" + "=" * 50)
    if all_tests_passed:
        print("--> Yay! All tests passed! The setup is ready.")

    else:
        print("!!! Some tests failed. Please check the errors above.")
        print("\n Troubleshooting:")
        print("1. Make sure you're in the virtual environment: source venv/bin/activate")
        print("2. Install dependencies: pip install -r requirements.txt")
        print("3. Create missing directories: mkdir -p src data tests docs")
    
    print("\n If you need help, refer to the README.md file.")

if __name__ == "__main__":
    main()
