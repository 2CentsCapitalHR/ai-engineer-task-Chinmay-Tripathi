import gradio as gr
import os
import sys
from datetime import datetime
from typing import Dict, List, Any
from dotenv import load_dotenv

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from parser import DocumentParser
from classifier import DocumentClassifier, ADGMKnowledgeBase, ComplianceChecker, RedFlagDetector

load_dotenv()

parser = DocumentParser()
classifier = DocumentClassifier()
kb = ADGMKnowledgeBase(openai_api_key=os.getenv("openai-api-key"))
compliance_checker = ComplianceChecker()
redflag_detector = RedFlagDetector()

def analyze_documents(files, classification_method="ensemble"):
    if not files:
        return {}, {}, "<span style='color: red;'>!! No files uploaded. Please upload at least one .docx document.</span>"

    results = {
        "processing_info": {
            "timestamp": datetime.now().isoformat(),
            "total_files": len(files),
            "classification_method": classification_method,
            "parser_version": "enhanced_v2"
        },
        "documents": [],
        "errors": [],
        "aggregate_analysis": {}
    }
    processing_log = []
    processed_count = 0
    total_words = 0
    document_types_found = []
    classified_types = []
    
    if not kb.vectorstore:
        kb_texts = []
        src_folder = "data/adgm_sources"
        if os.path.isdir(src_folder):
            for fname in os.listdir(src_folder):
                if fname.lower().endswith(".txt"):
                    kb_texts.append(open(os.path.join(src_folder, fname), encoding="utf-8").read())
        if kb_texts:
            kb.build_knowledge_base(kb_texts)
            log_lines.append(f"** Knowledge base built from {len(kb_texts)} ADGM docs.")

    processing_log.append(f"🚀 Starting analysis of {len(files)} documents...")

    for idx, file in enumerate(files):
        try:
            processing_log.append(f" Processing {file.name} ...")

            parsed_content = parser.parse_document(file.name)
            classification_result = classifier.classify_document(parsed_content, method=classification_method)
            explanation = classifier.get_classification_explanation(classification_result)
            related_refs = kb.retrieve(parsed_content["content"]["full_text"], k=2) if kb.vectorstore else []
            flags = redflag_detector.scan(parsed_content["content"]["full_text"])
            classified_types.append(classification_result["predicted_type"])

            doc_result = {
                "file_info": {
                    "filename": parsed_content["metadata"]["filename"],
                    "file_size_bytes": parsed_content["metadata"]["file_size"],
                    "processing_order": idx + 1
                },
                "structure_analysis": {
                    "word_count": parsed_content["statistics"]["word_count"],
                    "paragraph_count": parsed_content["statistics"]["paragraph_count"],
                    "section_count": parsed_content["statistics"]["section_count"],
                    "table_count": parsed_content["statistics"]["table_count"],
                    "sentence_count": parsed_content["statistics"]["sentence_count"],
                    "avg_words_per_paragraph": parsed_content["statistics"]["avg_words_per_paragraph"],
                    "document_structure": parsed_content["structure"]["document_structure"]
                },
                "classification": {
                    "predicted_type": classification_result["predicted_type"],
                    "confidence": classification_result["confidence"],
                    "method_used": classification_result["method_used"],
                    "explanation": explanation,
                    "alternative_types": dict(sorted(
                        classification_result.get("all_scores", {}).items(),
                        key=lambda x: x[1], reverse=True
                    )[:3])
                },
                "content_analysis": {
                    "text_quality": parsed_content["analysis"].get("text_quality", {}),
                    "legal_elements": parsed_content["analysis"].get("legal_elements", {}),
                    "readability": parsed_content["analysis"].get("readability", {}),
                    "document_complexity": parsed_content["analysis"].get("document_complexity", {}),
                    "key_phrases": parsed_content["content"].get("key_phrases", [])[:5],
                    "named_entities": parsed_content["content"].get("named_entities", [])[:5]
                }
            }

            doc_result["related_adgm_references"] = related_refs
            doc_result["red_flags"] = flags
            results["documents"].append(doc_result)
            processed_count += 1
            total_words += doc_result["structure_analysis"]["word_count"]
            document_types_found.append(doc_result["classification"]["predicted_type"])
            processing_log.append(
                f"--> {file.name}: {doc_result['classification']['predicted_type']} ({doc_result['classification']['confidence']:.1%} confidence)"
            )

        except Exception as e:
            error_msg = f"!! Error processing {file.name}: {str(e)}"
            results["errors"].append({
                "filename": file.name,
                "error": str(e),
                "processing_order": idx + 1
            })
            processing_log.append(error_msg)
        
    comp_result = compliance_checker.check(classified_types)

    results["aggregate_analysis"] = {
        "total_documents": processed_count,
        "total_words": total_words,
        "document_types_distribution": {t: document_types_found.count(t) for t in set(document_types_found)},
        "compliance": comp_result
    }

    summary_stats = {
        "Total Docs": processed_count,
        "Total Words": total_words,
        "Most Common Type": max(document_types_found, key=document_types_found.count) if document_types_found else "N/A",
        "Avg Confidence": round(sum(d["classification"]["confidence"] for d in results["documents"]) / processed_count, 3) if processed_count > 0 else 0
    }

    return results, summary_stats, comp_result,"<br>".join(processing_log)

#FRONTEND - User Interface
def create_interface():
    custom_css = """
    .header {
        text-align: center;
        padding: 1.5rem;
        background: linear-gradient(90deg, #4f46e5 0%, #3b82f6 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .metric-card {
        padding: 12px;
        background: #f9fafb;
        border-radius: 8px;
        border: 1px solid #e5e7eb;
        text-align: center;
    }
    """

    with gr.Blocks(css=custom_css, title="ADGM Corporate Agent v2", theme=gr.themes.Default()) as demo:
        with gr.Row():
            gr.HTML("""
            <div class="header">
                <h1> ADGM Corporate Agent v2</h1>
                <p>AI-powered analysis for ADGM legal documents with advanced ML classification & compliance checks</p>
            </div>
            """)

        with gr.Row():
            with gr.Column(scale=1):
                file_input = gr.File(label="📂 Upload DOCX files", file_types=[".docx"], file_count="multiple")
                method_dropdown = gr.Dropdown(label="Classification Method", choices=["rule_based", "ensemble"], value="ensemble")
                analyze_button = gr.Button(" Analyze Documents", variant="primary")

            with gr.Column(scale=2):
                with gr.Tab(" Summary"):
                    summary_json = gr.JSON(label="Summary Insights")
                with gr.Tab(" Detailed Results"):
                    results_json = gr.JSON(label="Detailed Analysis Output")
                with gr.Tab(" Compliance Report"):
                    comp_json = gr.JSON(label="Compliance Report")
                with gr.Tab(" Processing Log"):
                    log_text = gr.HTML(label="Processing Log", elem_id="log-text")

        analyze_button.click(
            analyze_documents,
            inputs=[file_input, method_dropdown],
            outputs=[results_json, summary_json, comp_json, log_text]
        )

    return demo

if __name__ == "__main__":
    ui = create_interface()
    ui.launch()

