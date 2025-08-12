import gradio as gr
import os, sys, re
from dotenv import load_dotenv

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from parser import DocumentParser
from classifier import (
    DocumentClassifier, ADGMKnowledgeBase,
    ComplianceChecker, RedFlagDetector, DocxAnnotator
)

load_dotenv()
parser = DocumentParser()
classifier = DocumentClassifier()
kb = ADGMKnowledgeBase(openai_api_key=os.getenv("OpenAI-API-Key")) #API key goes here
compliance_checker = ComplianceChecker()
redflag_detector = RedFlagDetector()
annotator = DocxAnnotator()


def analyze_documents(files, classification_method="ensemble"):
    if not files:
        return {}, {}, [], [], "!! No files uploaded.", []

    results = {"documents": [], "errors": [], "aggregate_analysis": {}}
    classified_types = []
    log_lines = []
    download_links = []
    review_reports = []

    # Build KB from ADGM reference text files only once
    if not kb.vectorstore:
        kb_texts = []
        src_folder = "data/adgm_sources"
        if os.path.isdir(src_folder):
            for fname in os.listdir(src_folder):
                if fname.lower().endswith(".txt"):
                    kb_texts.append(open(os.path.join(src_folder, fname), encoding="utf-8").read())
        if kb_texts:
            kb.build_knowledge_base(kb_texts)
            log_lines.append(f" Knowledge base built from {len(kb_texts)} reference docs.")

    for idx, file in enumerate(files):
        try:
            parsed_content = parser.parse_document(file.name)
            classification_result = classifier.classify_document(parsed_content, method=classification_method)
            explanation = classifier.get_classification_explanation(classification_result)
            classified_types.append(classification_result["predicted_type"])

            related_refs = kb.retrieve(parsed_content["content"]["full_text"], k=2) if kb.vectorstore else []
            flags = redflag_detector.scan(parsed_content["content"]["full_text"])

            review_comments = []
            for flag in flags:
                for para in parsed_content['structure']['paragraphs']:
                    if re.search(flag['pattern'], para['text'], flags=re.IGNORECASE):
                        review_comments.append({
                            "paragraph_index": para['index'],
                            "text": para['text'],
                            "comment": f"{flag['issue']} – {flag['suggestion']} (Severity: {flag['severity']})"
                        })

            marked_file_path = ""
            if review_comments:
                marked_file_path = annotator.add_comments(file.name, review_comments)
                download_links.append(marked_file_path)

            review_report = []
            for rc in review_comments:
                review_report.append({
                    "Paragraph": rc["text"][:80] + ("..." if len(rc["text"]) > 80 else ""),
                    "Comment": rc["comment"]
                })

            flat_rows = []
            for row in review_report:
                flat_rows.append([parsed_content["metadata"]["filename"], row["Paragraph"], row["Comment"]])
            review_reports.extend(flat_rows)

            doc_result = {
                "file_info": parsed_content["metadata"],
                "classification": classification_result,
                "explanation": explanation,
                "related_adgm_references": related_refs,
                "red_flags": flags,
                "marked_file": marked_file_path,
                "review_report": review_report
            }
            results["documents"].append(doc_result)
            log_lines.append(f" {file.name}: {classification_result['predicted_type']} "
                             f"({classification_result['confidence']:.1%} confidence) | Flags: {len(flags)}")
        except Exception as e:
            results["errors"].append({"filename": file.name, "error": str(e)})
            log_lines.append(f" {file.name}: {str(e)}")

    comp_result = compliance_checker.check(classified_types)
    results["aggregate_analysis"]["compliance"] = comp_result

    return results, comp_result, review_reports, download_links, "\n".join(log_lines)


def create_interface():
    custom_css = """
    .header {
        text-align: center;
        padding: 1.2rem;
        background: linear-gradient(90deg, #4f46e5 0%, #3b82f6 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    """

    with gr.Blocks(css=custom_css, title="ADGM Corporate Agent") as demo:
        gr.HTML("""
        <div class='header'>
            <h1>⚖️ ADGM Corporate Agent</h1>
            <p>RAG-powered document analysis with compliance check, red flag detection & annotation</p>
        </div>
        """)

        with gr.Row():
            with gr.Column(scale=1):
                file_input = gr.File(file_types=[".docx"], file_count="multiple",
                                     label="📂 Upload DOCX legal documents")
                method_dropdown = gr.Dropdown(choices=["rule_based", "ensemble"],
                                              value="ensemble", label="Classification Method")
                analyze_btn = gr.Button(" Analyze Documents", variant="primary")

            with gr.Column(scale=3):
                with gr.Tab(" Summary"):
                    compliance_box = gr.JSON(label="Compliance Status")
                with gr.Tab(" Full JSON Result"):
                    results_json = gr.JSON(label="Full Analysis JSON")
                with gr.Tab(" Red Flags & Review Report"):
                    review_table = gr.DataFrame(
                        headers=["Document", "Paragraph", "Comment"], label="Flagged Issues", interactive=False
                    )
                with gr.Tab(" Download Marked Documents"):
                    marked_files = gr.Files(label="Download Annotated DOCX", file_types=[".docx"])
                with gr.Tab(" Processing Log"):
                    log_box = gr.Textbox(label="Log", lines=15)

        analyze_btn.click(
            analyze_documents,
            inputs=[file_input, method_dropdown],
            outputs=[results_json, compliance_box, review_table, marked_files, log_box]
        )

    return demo


if __name__ == "__main__":
    app = create_interface()
    app.launch()

