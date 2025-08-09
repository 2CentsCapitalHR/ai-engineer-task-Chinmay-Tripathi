import gradio as gr
import os
import tempfile
import json
from pathlib import Path
from document_parser import DocumentParser
from dotenv import load_dotenv

load_dotenv()

parser = DocumentParser()

def analyze_documents(files):

    if not files:
        return {}, "No files uploaded. Please upload at least one .docx document."
    
    results = {
        "total_documents": len(files),
        "processed_documents": [],
        "errors": [],
        "summary": {}
    }
    
    processed_count = 0
    
    for file in files:
        try:
            # Parsing the document
            content = parser.parse_document(file.name)
            
            # Getting document type hints
            doc_type_hints = parser.get_document_type_hints(content)
            
            # Addng to results
            doc_result = {
                "filename": content['filename'],
                "word_count": content['word_count'],
                "paragraph_count": content['paragraph_count'],
                "section_count": len(content['sections']),
                "table_count": len(content['tables']),
                "potential_document_types": doc_type_hints
            }
            
            results["processed_documents"].append(doc_result)
            processed_count += 1
            
        except Exception as e:
            error_msg = f"Error processing {file.name}: {str(e)}"
            results["errors"].append(error_msg)
    
    # Summary
    results["summary"] = {
        "successfully_processed": processed_count,
        "errors": len(results["errors"]),
        "total_words": sum([doc["word_count"] for doc in results["processed_documents"]]),
        "most_common_document_type": get_most_common_type(results["processed_documents"])
    }
    
    status_msg = f"Analysis complete! Processed {processed_count}/{len(files)} documents successfully."
    if results["errors"]:
        status_msg += f" {len(results['errors'])} errors occurred."
    
    return results, status_msg

def get_most_common_type(documents):
    type_counts = {}
    for doc in documents:
        for doc_type in doc["potential_document_types"]:
            type_counts[doc_type] = type_counts.get(doc_type, 0) + 1
    
    if type_counts:
        return max(type_counts, key=type_counts.get)
    return "unknown"

def create_interface():

    css = """
    .container {
        max-width: 1200px !important;
    }
    .header {
        text-align: center;
        margin-bottom: 2rem;
    }
    .upload-box {
        border: 2px dashed #ccc;
        padding: 2rem;
        text-align: center;
        border-radius: 10px;
    }
    """
    
    with gr.Blocks(css=css, title="ADGM Corporate Agent", theme=gr.themes.Soft()) as interface:
        gr.HTML("""
        <div class="header">
            <h1>🏛️ ADGM Corporate Agent</h1>
            <h2>Document Intelligence for ADGM Compliance</h2>
            <p>Upload your legal documents (.docx format) for analysis and compliance checking</p>
        </div>
        """)
        
        with gr.Tab("Document Analysis"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 📄 Upload Documents")
                    
                    file_upload = gr.File(
                        label="Select DOCX Documents",
                        file_types=[".docx"],
                        file_count="multiple",
                        elem_classes=["upload-box"]
                    )
                    
                    analyze_btn = gr.Button(
                        "🔍 Analyze Documents", 
                        variant="primary",
                        size="lg"
                    )
                    
                    status_display = gr.Textbox(
                        label="Status",
                        interactive=False,
                        placeholder="Upload documents and click 'Analyze Documents' to begin..."
                    )
                
                with gr.Column(scale=2):
                    gr.Markdown("### --> Analysis Results")
                    
                    results_json = gr.JSON(
                        label="Detailed Results",
                        show_label=True
                    )
        
        with gr.Tab("About"):
            gr.Markdown("""
            ## About ADGM Corporate Agent
            
            This AI-powered legal assistant helps with:
            
            ### --> Current Features
            - ... **Document Parsing**: Extract content from DOCX files
            - ... **Document Classification**: Identify document types
            - ... **Basic Analysis**: Word count, structure analysis
            
            
            ### --> Supported Document Types
            - Articles of Association
            - Memorandum of Association  
            - Incorporation Applications
            - Board Resolutions
            - Register of Members and Directors
            
            ### !! How to Use
            1. Upload one or more .docx documents
            2. Click "Analyze Documents"
            3. Review the analysis results
            4. More features coming in upcoming phases!
            
            ---
            **Version**: 1 - Basic Document Processing  
            **Status**: --> Active Development
            """)
        
        # Event handlers
        analyze_btn.click(
            fn=analyze_documents,
            inputs=[file_upload],
            outputs=[results_json, status_display],
            show_progress=True
        )
    
    return interface

def main():
    print("--> Starting ADGM Corporate Agent...")
    print("... Step 1: Basic Document Processing")

    app = create_interface()

    app.launch(
        server_name="0.0.0.0",  # Allow external connections
        server_port=int(os.getenv("GRADIO_SERVER_PORT", 7860)),
        share=False,  
        debug=True,   
        show_error=True
    )

if __name__ == "__main__":
    main()
