import os
import sys

# Add current path to python system path for modules imports
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from api.gradio_app import demo

if __name__ == "__main__":
    # Launch Gradio interface (default port 7860 for Hugging Face Spaces integration)
    demo.launch(server_name="0.0.0.0", server_port=7860)
