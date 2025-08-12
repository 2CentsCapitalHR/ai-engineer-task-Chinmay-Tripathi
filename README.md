[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/vgbm4cZ0)

# ADGM Corporate Agent – Smart Legal Document Analyzer

An AI-powered legal document intelligence platform that provides comprehensive analysis, compliance checking, risk detection, and automated annotation for legal documents within the Abu Dhabi Global Market (ADGM) jurisdiction.

---

## Overview

The ADGM Corporate Agent is a sophisticated document processing system designed specifically for legal professionals, compliance officers, and corporate entities operating within the ADGM framework. The system combines advanced natural language processing, machine learning classification, and retrieval-augmented generation (RAG) to provide intelligent analysis of legal documents in DOCX format.

The platform automatically identifies document types, performs compliance verification against ADGM requirements, detects potential legal risks and inconsistencies, and generates annotated versions of documents with expert recommendations and legal citations.

---

## Key Capabilities

### Document Intelligence
- **Advanced Parsing**: Comprehensive extraction of document structure, metadata, paragraphs, sections, tables, and statistical analysis
- **ML Classification**: Intelligent document type identification using ensemble methods combining rule-based and machine learning approaches
- **Content Analysis**: Deep linguistic analysis including readability assessment, legal term density, formality scoring, and complexity evaluation
- **Entity Recognition**: Extraction of key phrases, named entities, legal terms, and compliance references

### Legal Analysis & Compliance
- **ADGM Compliance Verification**: Automated checking against required document sets for various legal processes (incorporation, licensing, regulatory filings)
- **Red Flag Detection**: Pattern-based identification of legal risks, incorrect jurisdictions, missing clauses, and non-compliance issues
- **Legal Citation Integration**: Each detected issue includes relevant ADGM law references and regulatory citations
- **Clause Suggestion Engine**: Automated generation of compliant legal clauses and alternative wording recommendations

### Knowledge Management
- **RAG-Enhanced Analysis**: Integration with ADGM knowledge base for context-aware legal advice and reference retrieval
- **Semantic Search**: Vector-based similarity matching against regulatory documents and legal precedents
- **Dynamic Knowledge Integration**: Continuous integration of ADGM regulations, templates, and legal frameworks

### Document Annotation & Export
- **Intelligent Annotation**: Automated highlighting and commenting of problematic clauses with severity-based color coding
- **Legal Commentary**: Inline insertion of expert recommendations, alternative clauses, and regulatory references
- **Export Capabilities**: Generation of annotated DOCX files with comprehensive review reports and downloadable marked documents

### Analytics & Reporting
- **Executive Dashboards**: Comprehensive analytics including document type distribution, compliance rates, and issue severity analysis
- **Structured Reporting**: JSON-formatted detailed analysis results and executive summary generation
- **Process Tracking**: Complete audit trail with processing logs and confidence scoring

---

## Supported Document Types

The system recognizes and processes various ADGM legal documents including:

- **Corporate Formation**: Articles of Association, Memorandum of Association, Incorporation Applications, Board Resolutions
- **Compliance Documents**: UBO Declarations, Register of Members and Directors, Regulatory Filings, Compliance Policies
- **Commercial Agreements**: Employment Contracts, Service Agreements, Licensing Applications, Commercial Contracts
- **Financial Documents**: Audit Reports, Financial Statements, Shareholder Resolutions
- **Legal Instruments**: Power of Attorney documents, Legal Opinions, Regulatory Submissions

---

## Technology Architecture

### Core Technologies
- **Python 3.12+** - Primary development language
- **Natural Language Processing**: spaCy for advanced linguistic analysis and entity recognition
- **Machine Learning**: scikit-learn with TF-IDF vectorization, Naive Bayes, SVM, and Random Forest classifiers
- **Vector Database**: FAISS for high-performance semantic search and RAG implementation
- **Document Processing**: python-docx for DOCX file parsing, annotation, and generation

### AI & Machine Learning Stack
- **LangChain**: Framework for RAG implementation and knowledge base integration
- **OpenAI Embeddings**: Vector embeddings for semantic document analysis
- **Ensemble Classification**: Multi-algorithm approach combining rule-based and ML methods
- **Pattern Recognition**: Advanced regex and linguistic pattern matching for legal clause detection

### User Interface & Visualization
- **Gradio**: Professional web interface with multi-tab analytics dashboard
- **pandas**: Data manipulation and structured reporting
- **Interactive Components**: File upload, real-time processing, and downloadable results

---

## Project Structure

```text
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
```
---

## Installation & Setup

### Prerequisites
- Python 3.12 or higher
- OpenAI API key (optional, for RAG features)
- Minimum 8GB RAM recommended for optimal performance

### Installation Steps

1. **Clone the repository and install dependencies**:
pip install -r requirements.txt
python -m spacy download en_core_web_sm

2. **Prepare ADGM KB source texts**
- Place `.txt` files in `data/adgm_sources/` — these should contain ADGM regulations, sample templates, etc.
- Include relevant ADGM regulations, templates, and legal frameworks.

3. **Run tests** (to confirm setup):
python test_setup.py

4. **Run the app**:
python app.py

---

## Usage Instructions

### Web Interface
1. **Access the application** at `http://localhost:7860` after launch
2. **Upload Documents**: Select one or more DOCX legal documents for analysis
3. **Choose Analysis Method**: Select classification approach (rule-based or ensemble recommended)
4. **Execute Analysis**: Click "Analyze Documents" to begin processing

### Review Results
- **Executive Summary**: High-level overview with key findings and recommendations
- **Compliance Dashboard**: Detailed compliance analysis with completeness percentages and missing documents
- **Issues & Red Flags**: Comprehensive table of identified problems with severity levels and ADGM references 
- **Download Center**: Access annotated documents with highlighted issues and legal commentary
- **Technical Details**: Complete JSON analysis results for integration with other systems

### Output Formats
- **Annotated DOCX Files**: Original documents with highlighted issues and inline legal commentary
- **Structured JSON**: Machine-readable analysis results with complete metadata
- **Executive Reports**: Markdown-formatted summaries with compliance status and recommendations
- **Analytics Data**: Document type distribution, confidence scores, and issue categorization

---

## System Requirements

### Minimum Requirements
- **Operating System**: Windows 10+, macOS 10.14+, or Ubuntu 18.04+
- **Memory**: 8GB RAM
- **Storage**: 2GB free disk space
- **Network**: Internet connection for RAG features (optional)

### Recommended Configuration
- **Memory**: 16GB RAM for optimal performance with large document sets
- **Storage**: SSD recommended for faster document processing
- **CPU**: Multi-core processor for parallel processing capabilities

---

## Important Notes

### Capabilities & Limitations
- **Document Format Support**: Currently limited to DOCX format; PDF support planned for future releases
- **Language Support**: Optimized for English legal documents; multilingual support under development 
- **Red Flag Detection**: Uses pattern-based analysis which may produce false positives; human review recommended
- **Compliance Checking**: Based on current ADGM regulations; users should verify against latest requirements

### Data Handling & Security
- **Document Processing**: All document processing occurs locally; no data transmitted to external services (except optional RAG features)
- **Temporary Files**: Uploaded documents are processed in temporary directories and automatically cleaned
- **Output Security**: Annotated documents are saved locally in the `marked_documents/` directory

### Performance Considerations
- **Processing Speed**: Analysis time varies based on document size and complexity (typically 5-30 seconds per document)
- **Batch Processing**: System supports simultaneous analysis of multiple documents
- **Resource Usage**: Memory usage scales with document size; monitor system resources for large document sets

---

## Support & Maintenance

### Troubleshooting
- **Installation Issues**: Ensure Python 3.12+ and all dependencies are properly installed
- **spaCy Model**: Run `python -m spacy download en_core_web_sm` if NLP features fail
- **Memory Errors**: Reduce batch size or increase system memory for large documents
- **API Integration**: Verify OpenAI API key configuration for RAG functionality
