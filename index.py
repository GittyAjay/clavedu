"""
Enhanced YouTube Audio Transcriber API
FastAPI-based API with cookie support to bypass YouTube bot detection

Installation required:
!pip install yt-dlp transformers torch moviepy fastapi uvicorn nest-asyncio browser-cookie3

New Features:
- Automatic cookie extraction from browsers
- Multiple fallback strategies for YouTube access
- Better error handling and retry mechanisms
- Support for age-restricted and private videos

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
from typing import Optional, Dict, Any, List
import threading
import time
import requests
import sys
import subprocess
import platform

# Try to import browser cookie extraction
try:
    import browser_cookie3
    BROWSER_COOKIES_AVAILABLE = True
except ImportError:
    BROWSER_COOKIES_AVAILABLE = False
    print("⚠️ browser_cookie3 not installed. Run: pip install browser_cookie3")

# Enable nested asyncio for Colab/Jupyter
nest_asyncio.apply()

class TranscriptionRequest(BaseModel):
    url: str
    whisper_model: Optional[str] = "openai/whisper-base"
    output_format: Optional[str] = "text"  # "text", "json", or "both"
    chunk_length_s: Optional[int] = 30
    include_timestamps: Optional[bool] = True
    use_cookies: Optional[bool] = True  # New option
    browser: Optional[str] = "chrome"  # chrome, firefox, safari, edge

class TranscriptionResponse(BaseModel):
    status: str
    message: str
    text: Optional[str] = None
    detailed: Optional[Dict[Any, Any]] = None
    video_info: Optional[Dict[str, Any]] = None
    retry_info: Optional[Dict[str, Any]] = None

class YouTubeTranscriberAPI:
    def __init__(self):
        self.app = FastAPI(
            title="Enhanced YouTube Transcriber API",
            description="API for transcribing YouTube videos using Whisper with cookie support",
            version="2.0.0"
        )
        
        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.models = {}
        self.cookie_jar = None
        self.setup_routes()

        print(f"🚀 Enhanced API initialized with device: {self.device}")
        print(f"🍪 Browser cookies support: {'✅' if BROWSER_COOKIES_AVAILABLE else '❌'}")
        print("✅ CORS middleware enabled")

    def setup_routes(self):
        """Setup FastAPI routes"""

        @self.app.get("/")
        async def root():
            return {
                "message": "Enhanced YouTube Transcriber API",
                "status": "running",
                "cors_enabled": True,
                "cookie_support": BROWSER_COOKIES_AVAILABLE
            }

        @self.app.get("/health")
        async def health_check():
            return {
                "status": "healthy",
                "device": self.device,
                "models_loaded": list(self.models.keys()),
                "cors_enabled": True,
                "cookie_support": BROWSER_COOKIES_AVAILABLE
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

        @self.app.post("/test-cookies")
        async def test_cookies(browser: str = "chrome"):
            """Test cookie extraction from browser"""
            try:
                cookies = self.extract_browser_cookies(browser)
                return {
                    "status": "success" if cookies else "no_cookies",
                    "browser": browser,
                    "cookies_found": len(cookies) if cookies else 0,
                    "message": "Cookies extracted successfully" if cookies else "No YouTube cookies found"
                }
            except Exception as e:
                return {
                    "status": "error",
                    "browser": browser,
                    "error": str(e)
                }

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

    def extract_browser_cookies(self, browser: str = "chrome"):
        """Extract cookies from browser"""
        if not BROWSER_COOKIES_AVAILABLE:
            return None
            
        try:
            print(f"🍪 Extracting cookies from {browser}...")
            
            if browser.lower() == "chrome":
                cj = browser_cookie3.chrome(domain_name='youtube.com')
            elif browser.lower() == "firefox":
                cj = browser_cookie3.firefox(domain_name='youtube.com')
            elif browser.lower() == "safari":
                cj = browser_cookie3.safari(domain_name='youtube.com')
            elif browser.lower() == "edge":
                cj = browser_cookie3.edge(domain_name='youtube.com')
            else:
                cj = browser_cookie3.chrome(domain_name='youtube.com')  # Default to Chrome
            
            cookies = list(cj)
            if cookies:
                print(f"✅ Found {len(cookies)} cookies from {browser}")
                return cj
            else:
                print(f"⚠️ No cookies found in {browser}")
                return None
                
        except Exception as e:
            print(f"❌ Error extracting cookies from {browser}: {e}")
            return None

    def create_cookie_file(self, browser: str = "chrome"):
        """Create a cookie file for yt-dlp"""
        cookies = self.extract_browser_cookies(browser)
        if not cookies:
            return None
            
        try:
            # Create temporary cookie file
            temp_dir = tempfile.mkdtemp()
            cookie_file = os.path.join(temp_dir, "cookies.txt")
            
            # Write cookies in Netscape format
            with open(cookie_file, 'w') as f:
                f.write("# Netscape HTTP Cookie File\n")
                for cookie in cookies:
                    f.write(f"{cookie.domain}\tTRUE\t{cookie.path}\t{'TRUE' if cookie.secure else 'FALSE'}\t{int(cookie.expires) if cookie.expires else 0}\t{cookie.name}\t{cookie.value}\n")
            
            return cookie_file
            
        except Exception as e:
            print(f"❌ Error creating cookie file: {e}")
            return None

    def get_ydl_options(self, use_cookies: bool = True, browser: str = "chrome"):
        """Get yt-dlp options with multiple fallback strategies"""
        temp_dir = tempfile.mkdtemp()
        temp_audio_path = os.path.join(temp_dir, "temp_audio")
        
        base_opts = {
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
        
        # Strategy 1: Use browser cookies
        if use_cookies and BROWSER_COOKIES_AVAILABLE:
            cookie_file = self.create_cookie_file(browser)
            if cookie_file:
                base_opts['cookiefile'] = cookie_file
                print(f"🍪 Using cookies from {browser}")
        
        return base_opts, temp_dir

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

    def download_audio_with_fallback(self, url: str, use_cookies: bool = True, browser: str = "chrome") -> tuple:
        """Download audio with multiple fallback strategies"""
        strategies = []
        
        # Strategy 1: Use browser cookies
        if use_cookies and BROWSER_COOKIES_AVAILABLE:
            strategies.append(("cookies", browser))
        
        # Strategy 2: No cookies but with user agent
        strategies.append(("user_agent", None))
        
        # Strategy 3: Basic download
        strategies.append(("basic", None))
        
        last_error = None
        
        for i, (strategy, browser_name) in enumerate(strategies):
            try:
                print(f"📥 Attempt {i+1}: Using {strategy} strategy...")
                
                ydl_opts, temp_dir = self.get_ydl_options(
                    use_cookies=(strategy == "cookies"), 
                    browser=browser_name or browser
                )
                
                # Add strategy-specific options
                if strategy == "user_agent":
                    ydl_opts['http_headers'] = {
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                    }
                elif strategy == "basic":
                    # Remove any authentication-related options
                    ydl_opts.pop('cookiefile', None)
                    ydl_opts.pop('http_headers', None)
                
                # Try to download
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(url, download=False)
                    video_info = {
                        'title': info.get('title', 'Unknown'),
                        'duration': info.get('duration', 'Unknown'),
                        'uploader': info.get('uploader', 'Unknown'),
                        'upload_date': info.get('upload_date', 'Unknown'),
                        'age_limit': info.get('age_limit', 0),
                        'availability': info.get('availability', 'Unknown')
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

                print(f"✅ Success with {strategy} strategy!")
                return downloaded_file, video_info, temp_dir, strategy

            except Exception as e:
                last_error = e
                print(f"❌ {strategy} strategy failed: {str(e)}")
                if 'temp_dir' in locals() and os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir, ignore_errors=True)
                continue

        # All strategies failed
        raise Exception(f"All download strategies failed. Last error: {str(last_error)}")

    async def process_transcription(self, request: TranscriptionRequest) -> TranscriptionResponse:
        """Process transcription request with enhanced error handling"""
        temp_dir = None
        try:
            # Download audio with fallback strategies
            audio_path, video_info, temp_dir, strategy = self.download_audio_with_fallback(
                request.url, 
                request.use_cookies, 
                request.browser
            )

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

            # Prepare response
            response_data = {
                "status": "success",
                "message": "Transcription completed successfully",
                "video_info": video_info,
                "retry_info": {
                    "strategy_used": strategy,
                    "cookies_used": request.use_cookies and strategy == "cookies",
                    "browser": request.browser if strategy == "cookies" else None
                }
            }

            if request.output_format in ["text", "both"]:
                response_data["text"] = text

            if request.output_format in ["json", "both"]:
                response_data["detailed"] = result

            print(f"✅ Transcription completed for: {video_info['title']}")
            return TranscriptionResponse(**response_data)

        except Exception as e:
            print(f"❌ Error in transcription: {str(e)}")
            
            # Provide helpful error messages
            error_message = str(e)
            if "Sign in to confirm you're not a bot" in error_message:
                error_message += "\n💡 Try: 1) Use cookies with use_cookies=true, 2) Try a different browser, 3) Wait a few minutes and retry"
            elif "Video unavailable" in error_message:
                error_message += "\n💡 Video may be private, deleted, or region-restricted"
            elif "age-restricted" in error_message.lower():
                error_message += "\n💡 Video is age-restricted. Make sure you're logged into YouTube in your browser"
            
            return TranscriptionResponse(
                status="error",
                message=error_message
            )
        finally:
            # Cleanup
            if temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)

def install_dependencies():
    """Install required dependencies"""
    packages = [
        "yt-dlp",
        "transformers", 
        "torch",
        "moviepy",
        "fastapi",
        "uvicorn",
        "nest-asyncio",
        "browser-cookie3",
        "requests"
    ]
    
    for package in packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✅ Installed {package}")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {package}: {e}")

def run_api_server(host: str = "0.0.0.0", port: int = 8000):
    """Run the enhanced API server"""
    transcriber_api = YouTubeTranscriberAPI()

    print("🎬 Enhanced YouTube Transcriber API (CORS + Cookies)")
    print("=" * 60)
    print(f"🏠 Local URL: http://localhost:{port}")
    print(f"🌐 Network URL: http://{host}:{port}")
    print("✅ CORS enabled - web browsers can access this API")
    print(f"🍪 Cookie support: {'✅' if BROWSER_COOKIES_AVAILABLE else '❌ (install browser-cookie3)'}")
    print("📋 Available endpoints:")
    print(f"  GET  http://localhost:{port}/ - API info")
    print(f"  GET  http://localhost:{port}/health - Health check")
    print(f"  POST http://localhost:{port}/transcribe - Transcribe video")
    print(f"  GET  http://localhost:{port}/models - Available models")
    print(f"  POST http://localhost:{port}/test-cookies - Test cookie extraction")
    print("=" * 60)
    print("💡 Tips:")
    print("  - Set use_cookies=true to bypass bot detection")
    print("  - Try different browsers if one doesn't work")
    print("  - Age-restricted videos need browser login")
    print("=" * 60)

    # Run server
    uvicorn.run(transcriber_api.app, host=host, port=port, log_level="info")

def start_api_in_background(host: str = "0.0.0.0", port: int = 8000):
    """Start API server in background thread"""
    def run_server():
        run_api_server(host=host, port=port)

    thread = threading.Thread(target=run_server, daemon=True)
    thread.start()

    time.sleep(5)
    print(f"✅ Enhanced API server running in background on http://localhost:{port}")
    return thread

def test_api_with_cookies(base_url: str = "http://localhost:8000"):
    """Test the API with cookie support"""
    # Use a more reliable test video
    test_url = "https://www.youtube.com/watch?v=jNQXAC9IVRw"  # "Me at the zoo" - first YouTube video
    
    payload = {
        "url": test_url,
        "whisper_model": "openai/whisper-tiny",
        "output_format": "both",
        "chunk_length_s": 30,
        "use_cookies": True,
        "browser": "chrome"
    }

    try:
        print("🧪 Testing Enhanced API with cookies...")
        
        # Test cookie extraction first
        print("Testing cookie extraction...")
        cookie_response = requests.post(f"{base_url}/test-cookies", json={"browser": "chrome"})
        print(f"Cookie test: {cookie_response.json()}")
        
        # Test transcription
        print("Testing transcription...")
        response = requests.post(f"{base_url}/transcribe", json=payload, timeout=300)

        if response.status_code == 200:
            result = response.json()
            print("✅ Enhanced API test successful!")
            print(f"📺 Video: {result.get('video_info', {}).get('title', 'Unknown')}")
            print(f"🔧 Strategy: {result.get('retry_info', {}).get('strategy_used', 'Unknown')}")
            if result.get('text'):
                preview = result['text'][:200] + "..." if len(result['text']) > 200 else result['text']
                print(f"📝 Preview: {preview}")
            return True
        else:
            print(f"❌ Enhanced API test failed: {response.status_code}")
            print(response.text)
            return False

    except Exception as e:
        print(f"❌ Enhanced API test error: {e}")
        return False

def setup_instructions():
    """Print setup instructions"""
    print("🚀 Enhanced YouTube Transcriber Setup")
    print("=" * 50)
    print("1. Install dependencies:")
    print("   pip install yt-dlp transformers torch moviepy fastapi uvicorn nest-asyncio browser-cookie3")
    print()
    print("2. Start the server:")
    print("   run_api_server()  # or start_api_in_background()")
    print()
    print("3. Test with cookies:")
    print("   test_api_with_cookies()")
    print()
    print("💡 Troubleshooting:")
    print("   - Make sure you're logged into YouTube in your browser")
    print("   - Try different browsers (chrome, firefox, safari, edge)")
    print("   - For age-restricted videos, login to YouTube first")
    print("   - If cookies don't work, the API will fallback automatically")

if __name__ == "__main__":
    run_api_server()

# Enhanced usage examples:
"""
# Install dependencies
install_dependencies()

# Start server
run_api_server()

# Or run in background
thread = start_api_in_background()

# Test with cookies
test_api_with_cookies("http://localhost:8000")

# Example API usage with enhanced options:
import requests

payload = {
    "url": "https://www.youtube.com/watch?v=YOUR_VIDEO_ID",
    "whisper_model": "openai/whisper-base",
    "output_format": "both",
    "use_cookies": True,
    "browser": "chrome"  # or "firefox", "safari", "edge"
}

response = requests.post("http://localhost:8000/transcribe", json=payload)
result = response.json()
print(f"Strategy used: {result['retry_info']['strategy_used']}")
print(f"Text: {result['text']}")
"""