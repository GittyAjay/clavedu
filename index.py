"""
YouTube Audio Transcriber API
FastAPI-based API that serves locally with CORS enabled

Installation required:
!pip install yt-dlp transformers torch moviepy fastapi uvicorn nest-asyncio

Usage:
1. Run this code
2. The API will be available at http://localhost:8000
3. Use the /transcribe endpoint with a POST request containing the YouTube URL
"""

import yt_dlp
from transformers import pipeline
import os
import torch
import json
from pathlib import Path
import tempfile
import shutil
import asyncio
import uvicorn
import nest_asyncio
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
from typing import Optional, Dict, Any
import threading
import time
import requests

# Enable nested asyncio for Colab/Jupyter
nest_asyncio.apply()

class TranscriptionRequest(BaseModel):
    url: str
    whisper_model: Optional[str] = "openai/whisper-base"
    output_format: Optional[str] = "text"  # "text", "json", or "both"
    chunk_length_s: Optional[int] = 30
    include_timestamps: Optional[bool] = True

class TranscriptionResponse(BaseModel):
    status: str
    message: str
    text: Optional[str] = None
    detailed: Optional[Dict[Any, Any]] = None
    video_info: Optional[Dict[str, Any]] = None

class YouTubeTranscriberAPI:
    def __init__(self):
        self.app = FastAPI(
            title="YouTube Transcriber API",
            description="API for transcribing YouTube videos using Whisper",
            version="1.0.0"
        )
        
        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Allows all origins
            allow_credentials=True,
            allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
            allow_headers=["*"],  # Allows all headers
        )
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.models = {}  # Cache for loaded models
        self.setup_routes()

        print(f"🚀 API initialized with device: {self.device}")
        print("✅ CORS middleware enabled - web browsers can now access the API")

    def setup_routes(self):
        """Setup FastAPI routes"""

        @self.app.get("/")
        async def root():
            return {"message": "YouTube Transcriber API", "status": "running", "cors_enabled": True}

        @self.app.get("/health")
        async def health_check():
            return {
                "status": "healthy",
                "device": self.device,
                "models_loaded": list(self.models.keys()),
                "cors_enabled": True
            }

        @self.app.post("/transcribe", response_model=TranscriptionResponse)
        async def transcribe_video(request: TranscriptionRequest):
            try:
                result = await self.process_transcription(request)
                return result
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/models")
        async def available_models():
            return {
                "available_models": [
                    "openai/whisper-tiny",
                    "openai/whisper-base",
                    "openai/whisper-small",
                    "openai/whisper-medium",
                    "openai/whisper-large-v2"
                ],
                "currently_loaded": list(self.models.keys())
            }

        # Add a test endpoint for CORS verification
        @self.app.options("/{path:path}")
        async def options_handler(path: str):
            return JSONResponse(
                content={"message": "CORS preflight successful"},
                headers={
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
                    "Access-Control-Allow-Headers": "*",
                }
            )

    def load_whisper_model(self, model_name: str):
        """Load Whisper model with caching"""
        if model_name not in self.models:
            print(f"🧠 Loading Whisper model: {model_name}")
            try:
                self.models[model_name] = pipeline(
                    "automatic-speech-recognition",
                    model=model_name,
                    device=0 if self.device == "cuda" else -1,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
                )
                print(f"✅ Model {model_name} loaded successfully")
            except Exception as e:
                print(f"❌ Error loading model {model_name}: {e}")
                raise e
        return self.models[model_name]

    def download_audio(self, url: str) -> tuple:
        """Download audio from YouTube and return path + video info"""
        try:
            print("📥 Downloading audio from YouTube...")

            # Create temporary directory
            temp_dir = tempfile.mkdtemp()
            temp_audio_path = os.path.join(temp_dir, "temp_audio")

            # yt-dlp options
            ydl_opts = {
                'format': 'bestaudio/best',
                'outtmpl': temp_audio_path + '.%(ext)s',
                'extractaudio': True,
                'audioformat': 'wav',
                'audioquality': 0,
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'wav',
                }],
                'quiet': True,
                'no_warnings': True,
            }

            # Download and extract info
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                video_info = {
                    'title': info.get('title', 'Unknown'),
                    'duration': info.get('duration', 'Unknown'),
                    'uploader': info.get('uploader', 'Unknown'),
                    'upload_date': info.get('upload_date', 'Unknown')
                }

                print(f"📺 Video: {video_info['title']}")
                print(f"⏱️ Duration: {video_info['duration']} seconds")

                # Download the audio
                ydl.download([url])

            # Find downloaded file
            downloaded_file = None
            for file in os.listdir(temp_dir):
                if file.startswith("temp_audio"):
                    downloaded_file = os.path.join(temp_dir, file)
                    break

            if not downloaded_file or not os.path.exists(downloaded_file):
                raise Exception("Downloaded audio file not found")

            return downloaded_file, video_info, temp_dir

        except Exception as e:
            print(f"❌ Error downloading audio: {str(e)}")
            if 'temp_dir' in locals():
                shutil.rmtree(temp_dir, ignore_errors=True)
            raise e

    async def process_transcription(self, request: TranscriptionRequest) -> TranscriptionResponse:
        """Process transcription request"""
        temp_dir = None
        try:
            # Download audio
            audio_path, video_info, temp_dir = self.download_audio(request.url)

            # Load model
            pipe = self.load_whisper_model(request.whisper_model)

            print("🔁 Transcribing audio...")

            # Transcribe
            result = pipe(
                audio_path,
                return_timestamps=request.include_timestamps,
                chunk_length_s=request.chunk_length_s,
                stride_length_s=5
            )

            text = result["text"]

            # Prepare response based on format
            response_data = {
                "status": "success",
                "message": "Transcription completed successfully",
                "video_info": video_info
            }

            if request.output_format in ["text", "both"]:
                response_data["text"] = text

            if request.output_format in ["json", "both"]:
                response_data["detailed"] = result

            print(f"✅ Transcription completed for: {video_info['title']}")
            return TranscriptionResponse(**response_data)

        except Exception as e:
            print(f"❌ Error in transcription: {str(e)}")
            return TranscriptionResponse(
                status="error",
                message=str(e)
            )
        finally:
            # Cleanup
            if temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)

def run_api_server(host: str = "0.0.0.0", port: int = 8000):
    """Run the API server"""
    transcriber_api = YouTubeTranscriberAPI()

    print("🎬 YouTube Transcriber API (CORS Enabled)")
    print("=" * 60)
    print(f"🏠 Local URL: http://localhost:{port}")
    print(f"🌐 Network URL: http://{host}:{port}")
    print("✅ CORS is enabled - web browsers can access this API")
    print("📋 Available endpoints:")
    print(f"  GET  http://localhost:{port}/ - API info")
    print(f"  GET  http://localhost:{port}/health - Health check")
    print(f"  POST http://localhost:{port}/transcribe - Transcribe video")
    print(f"  GET  http://localhost:{port}/models - Available models")
    print("=" * 60)

    # Run server
    uvicorn.run(transcriber_api.app, host=host, port=port, log_level="info")

def start_api_in_background(host: str = "0.0.0.0", port: int = 8000):
    """Start API server in background thread (useful for Jupyter/Colab)"""
    def run_server():
        run_api_server(host=host, port=port)

    thread = threading.Thread(target=run_server, daemon=True)
    thread.start()

    # Wait a bit for server to start
    time.sleep(5)
    print(f"✅ API server running in background on http://localhost:{port}")
    return thread

def test_api(base_url: str = "http://localhost:8000"):
    """Test the API with a sample request"""
    test_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"  # Replace with actual video

    payload = {
        "url": test_url,
        "whisper_model": "openai/whisper-tiny",  # Use tiny model for faster testing
        "output_format": "both",
        "chunk_length_s": 30
    }

    try:
        print("🧪 Testing API with CORS...")
        
        # Test CORS preflight
        print("Testing CORS preflight...")
        options_response = requests.options(f"{base_url}/transcribe")
        print(f"OPTIONS response: {options_response.status_code}")
        
        # Test actual request
        response = requests.post(f"{base_url}/transcribe", json=payload, timeout=300)

        if response.status_code == 200:
            result = response.json()
            print("✅ API test successful!")
            print(f"📺 Video: {result.get('video_info', {}).get('title', 'Unknown')}")
            if result.get('text'):
                preview = result['text'][:200] + "..." if len(result['text']) > 200 else result['text']
                print(f"📝 Preview: {preview}")
            return True
        else:
            print(f"❌ API test failed: {response.status_code}")
            print(response.text)
            return False

    except Exception as e:
        print(f"❌ API test error: {e}")
        return False

def generate_browser_test_code(base_url: str = "http://localhost:8000"):
    """Generate JavaScript code to test the API from a browser"""
    js_code = f"""
// Test this in your browser's developer console
// Make sure you're on a page served from localhost or the same origin

const testTranscription = async () => {{
    const url = '{base_url}/transcribe';
    const payload = {{
        url: 'https://www.youtube.com/watch?v=dQw4w9WgXcQ',
        whisper_model: 'openai/whisper-tiny',
        output_format: 'both',
        chunk_length_s: 30
    }};

    try {{
        console.log('Testing API from browser...');
        const response = await fetch(url, {{
            method: 'POST',
            headers: {{
                'Content-Type': 'application/json',
            }},
            body: JSON.stringify(payload)
        }});

        if (response.ok) {{
            const result = await response.json();
            console.log('✅ Success!', result);
        }} else {{
            console.log('❌ Error:', response.status, await response.text());
        }}
    }} catch (error) {{
        console.log('❌ Network error:', error);
    }}
}};

// Run the test
testTranscription();
"""
    print("🌐 Browser Test Code:")
    print("Copy and paste this into your browser's developer console to test:")
    print("-" * 50)
    print(js_code)

def colab_setup():
    """
    Setup instructions for Google Colab/Jupyter
    Run this function to start the API server
    """
    print("📱 Jupyter/Colab Setup (CORS Enabled)")
    print("=" * 40)
    print("1. Installing dependencies...")

    install_cmd = """
!pip install yt-dlp transformers torch moviepy fastapi uvicorn nest-asyncio requests
    """
    print("Run this command in a cell:")
    print(install_cmd)

    print("\n2. Starting API server...")
    print("Use: start_api_in_background()")

    print("\n3. Testing API...")
    print("Use: test_api('http://localhost:8000')")
    
    print("\n4. ✅ CORS is now enabled!")
    print("Your API can be accessed from any web browser at http://localhost:8000")

if __name__ == "__main__":
    # For direct execution
    run_api_server()

# Usage examples:
"""
# Option 1: Run directly (blocking)
run_api_server()

# Option 2: Run in background (non-blocking, useful for Jupyter/Colab)
thread = start_api_in_background()

# Option 3: Test the API
test_api("http://localhost:8000")

# Option 4: Generate browser test code
generate_browser_test_code("http://localhost:8000")

# Example API usage with requests:
import requests

payload = {
    "url": "https://www.youtube.com/watch?v=YOUR_VIDEO_ID",
    "whisper_model": "openai/whisper-base",
    "output_format": "both"
}

response = requests.post("http://localhost:8000/transcribe", json=payload)
result = response.json()
print(result)
"""