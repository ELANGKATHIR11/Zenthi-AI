import os
import sys
import uvicorn

# Add current path to python system path for modules imports
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from api.main import app

if __name__ == "__main__":
    # Launch Zenthi-AI OS FastAPI backend serving frontend dashboard
    print("Starting Zenthi-AI OS on http://localhost:7860")
    uvicorn.run(app, host="0.0.0.0", port=7860)
