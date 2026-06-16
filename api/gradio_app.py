import os
import sys
import torch
import gradio as gr

# Apply bitsandbytes Windows matching override
os.environ["BNB_CUDA_VERSION"] = "129"

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rag.vector_store import LocalRAG
from search.searxng_client import SearXNGClient
from memory.history import SessionMemory
from api.router import QueryRouter
from rag.doc_generator import DocumentGenerator
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# System variables
SYSTEM_PROMPT = (
    "I am Zenthi-AI, a lightweight educational AI assistant developed as a custom Small Language Model project. "
    "I am educational, helpful, accurate, concise, student-friendly, and programming-aware."
)
LOCAL_MODEL_DIR = os.getenv("HF_MODEL_ID", "KATHIR2006/zenthi-ai")
local_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "Zenthi-AI-merged"))
if os.path.exists(local_path):
    LOCAL_MODEL_DIR = local_path
OUTPUT_DOCS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "reports", "generated_docs"))
os.makedirs(OUTPUT_DOCS_DIR, exist_ok=True)

# Initialize components
print("Initializing RAG database, Search Client, and Query Router...")
rag = LocalRAG()
search_client = SearXNGClient()
memory = SessionMemory(max_turns=10)
router = QueryRouter()

print(f"Loading local 4-bit tokenizer and model from: {LOCAL_MODEL_DIR}...")
tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_DIR, trust_remote_code=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Targeting inference device: {device}")

# Configure 4-bit bitsandbytes configuration
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

try:
    model = AutoModelForCausalLM.from_pretrained(
        LOCAL_MODEL_DIR,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
except Exception as e:
    print(f"[WARNING] Quantized GPU loading failed: {e}. Loading on CPU fallback format (slower)...")
    model = AutoModelForCausalLM.from_pretrained(
        LOCAL_MODEL_DIR,
        device_map="cpu",
        trust_remote_code=True
    )

print("Inference model loaded successfully.")

# Session tracker
session_id = "spaces_production_session"

def chat_interface(query, history_ui, mode, temperature, top_p):
    # Convert query routing mode
    run_mode = mode
    if run_mode == "Auto Router":
        run_mode = router.route(query)
        
    citations = []
    context = ""
    
    # Query ChromaDB RAG
    if "RAG" in run_mode or run_mode == "HYBRID":
        rag_results = rag.query(query, top_k=3)
        if rag_results:
            context += "\n=== Local Document Context ===\n"
            for r in rag_results:
                context += f"- [{r['source']}]: {r['text']}\n"
                citations.append(f"Local Doc: {r['source']}")
                
    # Query Web Search
    if "Web" in run_mode or run_mode == "HYBRID" or run_mode == "SEARCH":
        web_results = search_client.search(query, top_k=3)
        if web_results:
            context += "\n=== Live Web Search Context ===\n"
            context += search_client.format_context(web_results)
            for w in web_results:
                citations.append(f"Web: {w['title']}")

    # Retrieve memory history
    messages = memory.get_messages_for_prompt(session_id, system_prompt=SYSTEM_PROMPT)
    
    # Format message with context
    user_message = query
    if context:
        user_message = (
            f"Context:\n{context}\n\n"
            f"User Query: {query}\n\n"
            f"Instructions: Answer the query accurately using the context provided if relevant, otherwise reply using your standard knowledge."
        )
        
    memory.add_message(session_id, "user", user_message)
    
    # Query native model
    try:
        inputs_list = memory.get_session(session_id)
        text = tokenizer.apply_chat_template(inputs_list, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer([text], return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=tokenizer.eos_token_id
            )
            
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]
        ai_reply = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    except Exception as e:
        ai_reply = f"[ERROR] Native inference failed: {e}"

    memory.add_message(session_id, "assistant", ai_reply)
    
    # Format UI response
    ui_reply = ai_reply
    if citations:
        ui_reply += "\n\n**Sources & Citations:**\n" + "\n".join([f"* {c}" for c in set(citations)])
        
    history_ui.append((query, ui_reply))
    return "", history_ui

def upload_file(file):
    if file is None:
        return "No file selected."
    filepath = file.name
    success = rag.ingest_document(filepath)
    if success:
        return f"Successfully uploaded and indexed: {os.path.basename(filepath)}"
    return "Failed to parse and index document."

def generate_document(prompt, format_type, filename):
    if not prompt.strip() or not filename.strip():
        return "Prompt and Filename cannot be empty.", None
        
    # Generate content using model
    messages = [
        {"role": "system", "content": "You are a professional technical report generator. Write comprehensive reports based on instructions."},
        {"role": "user", "content": f"Write a detailed, structured document about: {prompt}"}
    ]
    
    try:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer([text], return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id
            )
            
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]
        generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    except Exception as e:
        return f"Generation failed: {e}", None

    # Save to output formats
    if format_type == "Markdown (.md)":
        filepath = DocumentGenerator.export_markdown(generated_text, filename, OUTPUT_DOCS_DIR)
    elif format_type == "PDF (.pdf)":
        filepath = DocumentGenerator.export_pdf(generated_text, filename, OUTPUT_DOCS_DIR)
    else:
        filepath = DocumentGenerator.export_text(generated_text, filename, OUTPUT_DOCS_DIR)
        
    return f"Document successfully generated and saved to: {os.path.basename(filepath)}", filepath

def clear_session():
    memory.clear_session(session_id)
    return None

# Build Gradio UI
with gr.Blocks(theme=gr.themes.Soft(primary_hue="purple", secondary_hue="indigo"), title="Zenthi-AI Workspace") as demo:
    gr.HTML(
        """
        <div style='text-align: center; padding: 20px;'>
            <h1 style='background: linear-gradient(to right, #9d4edd, #b5179e); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 2.5rem; font-weight: 800;'>Zenthi-AI Workspace</h1>
            <p style='color: #6c757d;'>Industrial Production-Grade Small Language Model Assistant with RAG, Search, and Document Generation</p>
        </div>
        """
    )
    
    with gr.Tab("Conversational Chat"):
        with gr.Row():
            with gr.Column(scale=4):
                chatbot = gr.Chatbot(label="Chat History")
                with gr.Row():
                    query_input = gr.Textbox(placeholder="Ask Zenthi-AI anything...", show_label=False, scale=8)
                    send_btn = gr.Button("Send", variant="primary", scale=1)
                
                with gr.Row():
                    clear_btn = gr.Button("Clear History")
                    
            with gr.Column(scale=2):
                gr.Markdown("### Ingestion & Query Mode")
                mode_dropdown = gr.Dropdown(
                    choices=["Auto Router", "Direct SLM Answer", "Local Document RAG", "SearXNG Web Search", "Hybrid (RAG + Web)"], 
                    value="Auto Router", 
                    label="Routing Mode"
                )
                
                file_uploader = gr.File(label="Upload PDF, TXT, or MD to ChromaDB RAG", file_types=[".pdf", ".txt", ".md"])
                upload_status_txt = gr.Textbox(label="Ingestion Status", interactive=False)
                
                with gr.Accordion("Hyperparameters", open=False):
                    temp_slider = gr.Slider(minimum=0.1, maximum=1.5, value=0.7, step=0.1, label="Temperature")
                    top_p_slider = gr.Slider(minimum=0.1, maximum=1.0, value=0.9, step=0.05, label="Top P")

        # Chat triggers
        send_btn.click(chat_interface, [query_input, chatbot, mode_dropdown, temp_slider, top_p_slider], [query_input, chatbot])
        query_input.submit(chat_interface, [query_input, chatbot, mode_dropdown, temp_slider, top_p_slider], [query_input, chatbot])
        clear_btn.click(clear_session, None, chatbot)
        file_uploader.change(upload_file, file_uploader, upload_status_txt)

    with gr.Tab("Document Generation"):
        gr.Markdown("### Create reports, assignments, or code structures natively with Zenthi-AI")
        with gr.Row():
            with gr.Column():
                doc_prompt = gr.Textbox(label="What document should Zenthi-AI generate?", placeholder="Write a detailed prompt (e.g. Write a report on recursion including code examples)...", lines=5)
                doc_format = gr.Dropdown(choices=["Markdown (.md)", "Plain Text (.txt)", "PDF (.pdf)"], value="Markdown (.md)", label="Document Format")
                doc_filename = gr.Textbox(label="Target Filename", value="generated_report")
                gen_btn = gr.Button("Generate Document", variant="primary")
            
            with gr.Column():
                gen_status = gr.Textbox(label="Generation Status", interactive=False)
                file_download = gr.File(label="Download Generated Document")
                
        gen_btn.click(generate_document, [doc_prompt, doc_format, doc_filename], [gen_status, file_download])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
