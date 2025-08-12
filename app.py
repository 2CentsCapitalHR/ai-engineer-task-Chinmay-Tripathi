import gradio as gr
import os, sys, re, json
from datetime import datetime
from typing import List, Dict, Any
from dotenv import load_dotenv
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from parser import DocumentParser
from classifier import (
    DocumentClassifier, ADGMKnowledgeBase, ComplianceChecker, 
    RedFlagDetector, DocxAnnotator, LegalClauseSuggester
)

load_dotenv()
parser = DocumentParser()
classifier = DocumentClassifier()
kb = ADGMKnowledgeBase(openai_api_key=os.getenv("OpenAI-API-Key-Goes-Here"))
compliance_checker = ComplianceChecker()
redflag_detector = RedFlagDetector()
annotator = DocxAnnotator()
clause_suggester = LegalClauseSuggester()

def analyze_documents(files, classification_method="ensemble"):
    if not files:
        return {}, {}, [], [], "No files uploaded. Please upload DOCX documents.", [], ""

    results = {
        "processing_info": {
            "timestamp": datetime.now().isoformat(),
            "total_files": len(files),
            "classification_method": classification_method,
            "system_version": "production_v1.0"
        },
        "documents": [],
        "errors": [],
        "aggregate_analysis": {}
    }
    
    classified_types = []
    log_lines = []
    download_links = []
    review_table_data = []
    analytics_data = {
        "document_types": {},
        "confidence_scores": [],
        "red_flags_by_severity": {"High": 0, "Medium": 0, "Low": 0},
        "total_issues": 0,
        "avg_document_complexity": 0
    }

    if not kb.vectorstore:
        kb_texts = []
        src_folder = "data/adgm_sources"
        if os.path.isdir(src_folder):
            for fname in os.listdir(src_folder):
                if fname.lower().endswith(".txt"):
                    with open(os.path.join(src_folder, fname), encoding="utf-8") as f:
                        kb_texts.append(f.read())
        
        if kb_texts and kb.build_knowledge_base(kb_texts):
            log_lines.append(f"Knowledge base built from {len(kb_texts)} ADGM reference documents")
        else:
            log_lines.append("Knowledge base not available - proceeding without RAG")

    for idx, file in enumerate(files):
        try:
            log_lines.append(f"Processing {file.name}...")

            parsed_content = parser.parse_document(file.name)

            classification_result = classifier.classify_document(parsed_content, method=classification_method)
            classified_types.append(classification_result["predicted_type"])

            explanation = classifier.get_classification_explanation(classification_result)

            related_refs = []
            if kb.vectorstore:
                related_refs = kb.retrieve(parsed_content["content"]["full_text"], k=3)

            flags = redflag_detector.scan(parsed_content["content"]["full_text"])
            
            # For annotation
            review_comments = []
            for flag in flags:
                for para in parsed_content['structure']['paragraphs']:
                    if re.search(flag['pattern'], para['text'], flags=re.IGNORECASE):
                        review_comments.append({
                            "paragraph_index": para['index'],
                            "text": para['text'],
                            "comment": f"{flag['issue']} - {flag['suggestion']}",
                            "severity": flag['severity'],
                            "adgm_reference": flag.get('adgm_reference', '')
                        })

            marked_file_path = ""
            if review_comments:
                marked_file_path = annotator.add_comments(file.name, review_comments)
                if marked_file_path:
                    download_links.append(marked_file_path)

            for rc in review_comments:
                review_table_data.append([
                    parsed_content["metadata"]["filename"],
                    rc["text"][:80] + ("..." if len(rc["text"]) > 80 else ""),
                    rc["comment"],
                    rc["severity"],
                    rc.get("adgm_reference", "N/A")
                ])

            legal_suggestions = []
            for flag in flags:
                suggestion = clause_suggester.suggest_clause(flag.get('category', ''), flag.get('matched_text', ''))
                if suggestion:
                    legal_suggestions.append(suggestion)

            doc_result = {
                "file_info": {
                    "filename": parsed_content["metadata"]["filename"],
                    "file_size": parsed_content["metadata"]["file_size"],
                    "processing_order": idx + 1
                },
                "classification": classification_result,
                "explanation": explanation,
                "structure_analysis": {
                    "word_count": parsed_content["statistics"]["word_count"],
                    "paragraph_count": parsed_content["statistics"]["paragraph_count"],
                    "section_count": parsed_content["statistics"]["section_count"],
                    "complexity": parsed_content["analysis"]["document_complexity"]["overall_complexity"]
                },
                "content_analysis": {
                    "legal_elements": parsed_content["analysis"]["legal_elements"],
                    "readability": parsed_content["analysis"]["readability"],
                    "key_phrases": parsed_content["content"]["key_phrases"][:5],
                    "named_entities": parsed_content["content"]["named_entities"][:5]
                },
                "adgm_analysis": {
                    "related_references": related_refs,
                    "red_flags": flags,
                    "legal_suggestions": legal_suggestions,
                    "marked_file": marked_file_path,
                    "review_comments_count": len(review_comments)
                }
            }
            
            results["documents"].append(doc_result)

            doc_type = classification_result["predicted_type"]
            analytics_data["document_types"][doc_type] = analytics_data["document_types"].get(doc_type, 0) + 1
            analytics_data["confidence_scores"].append(classification_result["confidence"])
            
            for flag in flags:
                severity = flag.get("severity", "Medium")
                analytics_data["red_flags_by_severity"][severity] = analytics_data["red_flags_by_severity"].get(severity, 0) + 1
            
            analytics_data["total_issues"] += len(flags)
            
            log_lines.append(f">> {file.name}: {doc_type} ({classification_result['confidence']:.1%}) | {len(flags)} issues")
            
        except Exception as e:
            error_msg = f"Error processing {file.name}: {str(e)}"
            results["errors"].append({
                "filename": file.name,
                "error": str(e),
                "processing_order": idx + 1
            })
            log_lines.append(f"!! {error_msg}")

    comp_result = compliance_checker.check(classified_types)
    results["aggregate_analysis"]["compliance"] = comp_result

    if analytics_data["confidence_scores"]:
        analytics_data["avg_confidence"] = sum(analytics_data["confidence_scores"]) / len(analytics_data["confidence_scores"])
    else:
        analytics_data["avg_confidence"] = 0

    summary_report = _generate_summary_report(results, analytics_data, comp_result)
    
    return (
        results,             
        comp_result,           
        review_table_data,        
        download_links,         
        "\n".join(log_lines),   
        analytics_data,          
        summary_report           
    )

def _generate_summary_report(results, analytics, compliance):
    total_docs = len(results["documents"])
    total_issues = analytics["total_issues"]
    avg_confidence = analytics.get("avg_confidence", 0)
    
    report = f"""# ADGM Document Analysis Summary

## Overview
- **Documents Processed**: {total_docs}
- **Total Issues Identified**: {total_issues}
- **Average Classification Confidence**: {avg_confidence:.1%}
- **Compliance Status**: {compliance.get('status', 'Unknown')}

## Document Types Identified
"""
    
    for doc_type, count in analytics["document_types"].items():
        report += f"- **{doc_type.replace('_', ' ').title()}**: {count} document(s)\n"
    
    report += f"""
## Issue Breakdown by Severity
- **High Severity**: {analytics['red_flags_by_severity'].get('High', 0)} issues
- **Medium Severity**: {analytics['red_flags_by_severity'].get('Medium', 0)} issues  
- **Low Severity**: {analytics['red_flags_by_severity'].get('Low', 0)} issues

## Compliance Analysis
- **Process**: {compliance.get('process', 'Unknown').replace('_', ' ').title()}
- **Completeness**: {compliance.get('completeness_percentage', 0):.1f}%
- **Missing Documents**: {len(compliance.get('missing_documents', []))}
"""
    
    if compliance.get('recommendations'):
        report += "\n## Recommendations\n"
        for rec in compliance['recommendations'][:3]:  # Top 3 recommendations
            report += f"- {rec}\n"
    
    return report

# UI Design
def create_interface():    
    custom_css = """
    .header {
        text-align: center;
        padding: 2rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 15px;
        margin-bottom: 25px;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
    }
    .metric-box {
        padding: 20px;
        background: #f8fafc;
        border-radius: 10px;
        border-left: 4px solid #4f46e5;
        margin: 10px 0;
    }
    .status-complete {
        background-color: #d1fae5;
        color: #065f46;
        padding: 8px 16px;
        border-radius: 6px;
    }
    .status-incomplete {
        background-color: #fee2e2;
        color: #991b1b;
        padding: 8px 16px;
        border-radius: 6px;
    }
    """
    
    with gr.Blocks(css=custom_css, title="ADGM Corporate Agent - Production") as demo:
        gr.HTML("""
        <div class='header'>
            <h1>🏛️ ADGM Corporate Agent</h1>
            <h2>AI-Powered Legal Document Intelligence Platform</h2>
            <p>Professional document analysis with RAG-enhanced compliance checking, red flag detection, and legal annotation</p>
        </div>
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("## 📤 Document Upload")
                file_input = gr.File(
                    file_types=[".docx"], 
                    file_count="multiple",
                    label="Upload DOCX Legal Documents",
                    height=120
                )
                
                classification_method = gr.Dropdown(
                    choices=["rule_based", "ensemble"],
                    value="ensemble",
                    label="Classification Method",
                    info="Ensemble method recommended for best accuracy"
                )
                
                analyze_btn = gr.Button(
                    "Analyze Documents", 
                    variant="primary",
                    size="lg"
                )

                gr.Markdown("## Quick Stats")
                stats_display = gr.HTML("<p>Upload documents to see analysis statistics</p>")

            with gr.Column(scale=2):
                with gr.Tab("Executive Summary"):
                    summary_report = gr.Markdown("Upload and analyze documents to see summary report")
                
                with gr.Tab("Compliance Dashboard"):
                    compliance_status = gr.JSON(label="Compliance Analysis")
                
                with gr.Tab("Issues & Red Flags"):
                    issues_table = gr.DataFrame(
                        headers=["Document", "Text Excerpt", "Issue", "Severity", "ADGM Reference"],
                        label="Identified Issues",
                        wrap=True
                    )
                
                with gr.Tab("Download Center"):
                    gr.Markdown("### Annotated Documents")
                    gr.Markdown("Download documents with highlighted issues and legal comments")
                    annotated_files = gr.Files(
                        label="Annotated DOCX Files", 
                        file_types=[".docx"]
                    )
                
                with gr.Tab("Technical Details"):
                    full_results = gr.JSON(
                        label="Complete Analysis Results",
                        show_label=True
                    )
                
                with gr.Tab("Processing Log"):
                    processing_log = gr.Textbox(
                        label="System Log",
                        lines=12,
                        show_copy_button=True
                    )

        def update_stats(analytics_data):
            if not analytics_data:
                return "<p>No statistics available</p>"
            
            total_docs = sum(analytics_data.get("document_types", {}).values())
            total_issues = analytics_data.get("total_issues", 0)
            avg_confidence = analytics_data.get("avg_confidence", 0)
            
            return f"""
            <div class='metric-box'>
                <h4>Analysis Statistics</h4>
                <p><strong>Documents:</strong> {total_docs}</p>
                <p><strong>Issues Found:</strong> {total_issues}</p>
                <p><strong>Avg Confidence:</strong> {avg_confidence:.1%}</p>
            </div>
            """
        
        analyze_btn.click(
            analyze_documents,
            inputs=[file_input, classification_method],
            outputs=[
                full_results,   
                compliance_status, 
                issues_table,     
                annotated_files,  
                processing_log,   
                gr.State(),        
                summary_report    
            ]
        ).then(
            update_stats,
            inputs=[gr.State()],  
            outputs=[stats_display]
        )
    
    return demo

if __name__ == "__main__":
    print("Starting ADGM Corporate Agent...")
    print("Loading knowledge base and initializing components...")
    
    app = create_interface()
    
    print("System ready!")
    print("Launching web interface...")
    
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
        share=False
    )

