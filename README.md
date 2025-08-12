[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/vgbm4cZ0)

# ADGM Corporate Agent – Smart Legal Document Analyzer

This project is an AI-powered document processing system that parses, classifies, checks compliance, flags risky clauses, and annotates legal documents in `.docx` format — with an interactive Gradio interface.

---

## Overview

The system was built in multiple phases:

**Basic application**
- Implemented `DocumentParser` for extracting:
  - Paragraphs, sections, tables
  - Document metadata and structure
  - Key statistics such as word count, sentence count, averages
  - Basic NLP features: key phrases, named entities, legal elements
- Implemented `DocumentClassifier` with:
  - Rule-based classification using ADGM-specific patterns
  - Placeholder for ML-based classification (TF-IDF + multiple algorithms)
- Created `app.py` for running an interactive Gradio app
- Added `test_setup.py` to validate imports, parser, classifier, and integration.

**Added features**
- Added a Retrieval-Augmented Generation (RAG) knowledge base:
  - `ADGMKnowledgeBase` for building a FAISS vectorstore from reference ADGM documents
  - Semantic retrieval for each uploaded document to find related legal/regulatory references
- Introduced `ComplianceChecker` for validating presence of required incorporation documents
- Added `RedFlagDetector` for pattern-based risk detection
- Integrated these features into `app.py`, keeping existing structure and file/function names
- Introduced `DocxAnnotator` to generate annotated `.docx` files highlighting or marking issues found
- Annotation uses in-text highlighting and inline comments for flagged paragraphs
- Extended UI to allow downloading marked documents after analysis
- Added structured review reports summarising paragraph-level issues
- Organised outputs into clear tabs for:
  - Compliance summary
  - Full JSON analysis
  - Red flags and review table
  - Download of annotated documents
  - Processing log

---

## Features

- Upload multiple `.docx` files for batch processing
- Extraction of content, structure, statistics, and metadata
- Classification into specific ADGM document types
- RAG-driven reference retrieval from ADGM corpus
- Compliance checklist against required document set
- Detection of red flag clauses and risky wording using regex patterns
- Annotation of flagged issues in generated reviewed `.docx` files
- Structured output of results in the Gradio UI
- Downloadable annotated documents for further review
- JSON and table views of analysis results
- Comprehensive test suite in `test_setup.py`

---

## Technology Stack

- **Python 3.12+**
- [python-docx](https://python-docx.readthedocs.io/) for document parsing and annotation
- [spaCy](https://spacy.io/) for optional NLP (key phrases, named entities)
- [scikit-learn](https://scikit-learn.org/) for ML-based classification (TF-IDF, Naive Bayes, SVM, RandomForest)
- [LangChain](https://www.langchain.com/) + [FAISS](https://faiss.ai/) for vector-based RAG
- [Gradio](https://www.gradio.app/) for interactive UI
- [pandas](https://pandas.pydata.org/) and [plotly](https://plotly.com/python/) for tabular and potential chart outputs

---

## Project Structure

.
├── app.py # Main Gradio application
├── src/
├── data/
│ └── adgm_sources/ # Reference ADGM text files for KB (not added here yet)
├── marked_documents/ # Output directory for annotated DOCX files
├── test_setup.py # Test suite for components and integration
├── classifier.py #Classification of document types
├── parser.py # Document parser
└── requirements.txt # Python dependencies

---

## Usage

1. **Install dependencies**:
pip install -r requirements.txt
python -m spacy download en_core_web_sm

2. **Prepare ADGM KB source texts**
Place `.txt` files in `data/adgm_sources/` — these should contain ADGM regulations, sample templates, etc.

3. **Run tests** (to confirm setup):
python test_setup.py

4. **Run the app**:
python app.py

5. **In the UI**:
- Upload one or more `.docx` legal documents
- Choose classification method (rule_based or ensemble)
- Click "Analyze Documents"
- Review results across the tabs:
  - Compliance checklist
  - Full JSON output
  - Red flags table
  - Download marked documents
  - Processing log

---

## Notes

- Only `.docx` files are supported for parsing and annotation.
- Red flag detection is currently regex-based; false positives/negatives are possible.
- Annotated documents are saved in `marked_documents/`.
- The RAG KB is built from `.txt` files in `data/adgm_sources` at startup and cached for subsequent runs.
- If spaCy model is not present, the parser will fallback to regex-based extraction for key phrases/entities.

---

## Next Steps

Possible enhancements after Phase 4:
- ML-based classification training and evaluation on labelled document sets
- Expand red flag patterns and use ML/NLP for clause classification
- Interactive clause-by-clause review in the UI
- Visualization dashboards for document type distribution, compliance rates, and trends
