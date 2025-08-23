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

        @self.app.get("/install-ffmpeg")
        async def install_ffmpeg_endpoint():
            """Install FFmpeg via API endpoint"""
            try:
                success = install_ffmpeg()
                return {
                    "status": "success" if success else "failed",
                    "message": "FFmpeg installation completed" if success else "FFmpeg installation failed",
                    "ffmpeg_available": success
                }
            except Exception as e:
                return {
                    "status": "error",
                    "message": str(e)
                }
        
        @self.app.get("/check-ffmpeg")
        async def check_ffmpeg():
            """Check if FFmpeg is available"""
            try:
                result = subprocess.run(['ffmpeg', '-version'], 
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    version_line = result.stdout.split('\n')[0]
                    return {
                        "status": "available",
                        "version": version_line,
                        "message": "FFmpeg is installed and working"
                    }
                else:
                    return {
                        "status": "not_working",
                        "message": "FFmpeg found but not working properly"
                    }
            except FileNotFoundError:
                return {
                    "status": "not_found",
                    "message": "FFmpeg not found. Use /install-ffmpeg endpoint to install it."
                }
            except subprocess.TimeoutExpired:
                return {
                    "status": "timeout",
                    "message": "FFmpeg check timed out"
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
        
        # Strategy 1: Use yt-dlp's built-in cookie extraction
        if use_cookies:
            strategies.append(("builtin_cookies", browser))
        
        # Strategy 2: Use browser cookies (if available)
        if use_cookies and BROWSER_COOKIES_AVAILABLE:
            strategies.append(("manual_cookies", browser))
        
        # Strategy 3: Use different user agents
        strategies.append(("user_agent_mobile", None))
        strategies.append(("user_agent_desktop", None))
        
        # Strategy 4: Basic download with headers
        strategies.append(("headers_only", None))
        
        # Strategy 5: Last resort - basic download
        strategies.append(("basic", None))
        
        last_error = None
        
        for i, (strategy, browser_name) in enumerate(strategies):
            try:
                print(f"📥 Attempt {i+1}: Using {strategy} strategy...")
                
                # Get base options
                temp_dir = tempfile.mkdtemp()
                temp_audio_path = os.path.join(temp_dir, "temp_audio")
                
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
                
                # Apply strategy-specific options
                if strategy == "builtin_cookies":
                    # Use yt-dlp's built-in cookie extraction
                    ydl_opts['cookiesfrombrowser'] = (browser_name, None, None, None)
                    print(f"🍪 Using yt-dlp built-in cookie extraction from {browser_name}")
                
                elif strategy == "manual_cookies":
                    cookie_file = self.create_cookie_file(browser_name)
                    if cookie_file:
                        ydl_opts['cookiefile'] = cookie_file
                        print(f"🍪 Using manual cookie file from {browser_name}")
                    else:
                        print("⚠️ No cookies found, skipping manual cookie strategy")
                        continue
                
                elif strategy == "user_agent_mobile":
                    ydl_opts['http_headers'] = {
                        'User-Agent': 'Mozilla/5.0 (iPhone; CPU iPhone OS 14_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0.3 Mobile/15E148 Safari/604.1'
                    }
                
                elif strategy == "user_agent_desktop":
                    ydl_opts['http_headers'] = {
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
                    }
                
                elif strategy == "headers_only":
                    ydl_opts['http_headers'] = {
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                        'Accept-Language': 'en-us,en;q=0.5',
                        'Accept-Encoding': 'gzip,deflate',
                        'Accept-Charset': 'ISO-8859-1,utf-8;q=0.7,*;q=0.7',
                        'Keep-Alive': '115',
                        'Connection': 'keep-alive',
                    }
                
                # Try to download
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    # First, try to extract info
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
                    
                    # If info extraction worked, try downloading
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

def install_ffmpeg():
    """Install FFmpeg automatically based on the operating system"""
    system = platform.system().lower()
    
    print("🎥 Installing FFmpeg...")
    
    try:
        # Check if FFmpeg is already installed
        result = subprocess.run(['ffmpeg', '-version'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("✅ FFmpeg is already installed!")
            return True
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    
    try:
        if system == "windows":
            # Windows: Use chocolatey or direct download
            print("🪟 Windows detected - installing FFmpeg...")
            
            # Try chocolatey first
            try:
                subprocess.check_call(['choco', 'install', 'ffmpeg', '-y'], 
                                    timeout=300)
                print("✅ FFmpeg installed via Chocolatey")
                return True
            except (subprocess.CalledProcessError, FileNotFoundError):
                print("⚠️ Chocolatey not found, trying alternative method...")
            
            # Alternative: Download and extract FFmpeg
            try:
                import urllib.request
                import zipfile
                
                print("📥 Downloading FFmpeg for Windows...")
                ffmpeg_url = "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip"
                
                # Create ffmpeg directory in temp
                ffmpeg_dir = os.path.join(tempfile.gettempdir(), "ffmpeg")
                os.makedirs(ffmpeg_dir, exist_ok=True)
                
                zip_path = os.path.join(ffmpeg_dir, "ffmpeg.zip")
                urllib.request.urlretrieve(ffmpeg_url, zip_path)
                
                # Extract
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(ffmpeg_dir)
                
                # Find the ffmpeg.exe and add to PATH
                for root, dirs, files in os.walk(ffmpeg_dir):
                    if 'ffmpeg.exe' in files:
                        ffmpeg_path = root
                        os.environ['PATH'] = ffmpeg_path + os.pathsep + os.environ['PATH']
                        print(f"✅ FFmpeg added to PATH: {ffmpeg_path}")
                        return True
                        
            except Exception as e:
                print(f"❌ Failed to download FFmpeg: {e}")
                
        elif system == "darwin":  # macOS
            print("🍎 macOS detected - installing FFmpeg via Homebrew...")
            try:
                # Try homebrew
                subprocess.check_call(['brew', 'install', 'ffmpeg'], timeout=300)
                print("✅ FFmpeg installed via Homebrew")
                return True
            except (subprocess.CalledProcessError, FileNotFoundError):
                print("⚠️ Homebrew not found. Please install Homebrew first:")
                print("   /bin/bash -c \"$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\"")
                
        elif system == "linux":
            print("🐧 Linux detected - installing FFmpeg...")
            
            # Try different package managers
            package_managers = [
                (['apt-get', 'update'], ['apt-get', 'install', '-y', 'ffmpeg']),  # Ubuntu/Debian
                (['yum', 'install', '-y', 'ffmpeg'],),  # CentOS/RHEL
                (['dnf', 'install', '-y', 'ffmpeg'],),  # Fedora
                (['pacman', '-S', '--noconfirm', 'ffmpeg'],),  # Arch
                (['zypper', 'install', '-y', 'ffmpeg'],),  # openSUSE
            ]
            
            for commands in package_managers:
                try:
                    for cmd in commands:
                        subprocess.check_call(cmd, timeout=300)
                    print("✅ FFmpeg installed via system package manager")
                    return True
                except (subprocess.CalledProcessError, FileNotFoundError):
                    continue
                    
        # If all else fails, try conda
        try:
            print("🐍 Trying conda installation...")
            subprocess.check_call(['conda', 'install', '-c', 'conda-forge', 'ffmpeg', '-y'], 
                                timeout=300)
            print("✅ FFmpeg installed via conda")
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass
        
        # Last resort: suggest manual installation
        print("❌ Automatic FFmpeg installation failed")
        print("💡 Please install FFmpeg manually:")
        if system == "windows":
            print("   Windows: https://ffmpeg.org/download.html#build-windows")
            print("   Or use: choco install ffmpeg")
        elif system == "darwin":
            print("   macOS: brew install ffmpeg")
        elif system == "linux":
            print("   Ubuntu/Debian: sudo apt install ffmpeg")
            print("   CentOS/RHEL: sudo yum install ffmpeg")
            print("   Fedora: sudo dnf install ffmpeg")
        
        return False
        
    except Exception as e:
        print(f"❌ Error installing FFmpeg: {e}")
        return False


def install_dependencies():
    """Install required dependencies including FFmpeg"""
    print("🚀 Installing all dependencies...")
    
    # Install Python packages first
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
    
    failed_packages = []
    for package in packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package], 
                                timeout=120)
            print(f"✅ Installed {package}")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {package}: {e}")
            failed_packages.append(package)
    
    # Install FFmpeg
    ffmpeg_success = install_ffmpeg()
    
    # Summary
    print("\n" + "="*50)
    print("📋 Installation Summary:")
    print(f"✅ Python packages: {len(packages) - len(failed_packages)}/{len(packages)}")
    if failed_packages:
        print(f"❌ Failed packages: {', '.join(failed_packages)}")
    print(f"{'✅' if ffmpeg_success else '❌'} FFmpeg: {'Installed' if ffmpeg_success else 'Failed'}")
    
    if not failed_packages and ffmpeg_success:
        print("🎉 All dependencies installed successfully!")
        return True
    else:
        print("⚠️ Some dependencies failed to install")
        return False

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

def quick_fix_cookie_test(url: str = "https://www.youtube.com/watch?v=RGaW82k4dK4"):
    """Quick test to see which cookie method works"""
    browsers = ["chrome", "firefox", "edge", "safari"]
    
    print("🔧 Quick Cookie Fix Test")
    print("=" * 40)
    
    for browser in browsers:
        print(f"\n🔍 Testing {browser} cookies...")
        
        try:
            # Test yt-dlp's built-in cookie extraction
            temp_dir = tempfile.mkdtemp()
            
            ydl_opts = {
                'quiet': True,
                'no_warnings': True,
                'cookiesfrombrowser': (browser, None, None, None),
                'format': 'bestaudio/best',
                'extract_flat': True,  # Just extract info
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                
            print(f"✅ {browser} cookies work! Video: {info.get('title', 'Unknown')}")
            return browser  # Return the working browser
            
        except Exception as e:
            print(f"❌ {browser} cookies failed: {str(e)[:100]}...")
        
        finally:
            if 'temp_dir' in locals() and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)
    
    print("\n❌ No browser cookies are working")
    print("💡 Solutions:")
    print("   1. Make sure you're logged into YouTube in your browser")
    print("   2. Close and reopen your browser")
    print("   3. Clear YouTube cookies and login again")
    print("   4. Try using a VPN if you're in a restricted region")
    
    return None


def create_manual_cookie_file():
    """Create a manual cookie file - user provides their own cookies"""
    print("🍪 Manual Cookie Setup")
    print("=" * 30)
    print("1. Go to YouTube in your browser")
    print("2. Open Developer Tools (F12)")
    print("3. Go to Application/Storage > Cookies > https://youtube.com")
    print("4. Find these important cookies and copy their values:")
    
    cookies_needed = [
        "SAPISID",
        "APISID", 
        "HSID",
        "SSID",
        "SID",
        "__Secure-3PAPISID",
        "__Secure-3PSID"
    ]
    
    cookie_values = {}
    for cookie in cookies_needed:
        value = input(f"Enter {cookie} value (or press Enter to skip): ").strip()
        if value:
            cookie_values[cookie] = value
    
    if not cookie_values:
        print("No cookies provided")
        return None
    
    # Create cookie file
    temp_dir = tempfile.mkdtemp()
    cookie_file = os.path.join(temp_dir, "manual_cookies.txt")
    
    with open(cookie_file, 'w') as f:
        f.write("# Netscape HTTP Cookie File\n")
        for name, value in cookie_values.items():
            f.write(f".youtube.com\tTRUE\t/\tTRUE\t0\t{name}\t{value}\n")
    
    print(f"✅ Cookie file created: {cookie_file}")
    return cookie_file

def setup_complete_environment():
    """Complete setup including FFmpeg and all dependencies"""
    print("🎯 Complete YouTube Transcriber Setup")
    print("=" * 50)
    
    # Step 1: Install Python dependencies
    print("📦 Step 1: Installing Python packages...")
    install_success = install_dependencies()
    
    # Step 2: Check/Install FFmpeg
    print("\n🎥 Step 2: Checking FFmpeg...")
    ffmpeg_success = install_ffmpeg()
    
    # Step 3: Test everything
    print("\n🧪 Step 3: Testing installation...")
    
    # Test imports
    try:
        import yt_dlp
        import transformers
        import torch
        import fastapi
        print("✅ All Python imports successful")
        python_ok = True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        python_ok = False
    
    # Test FFmpeg
    try:
        result = subprocess.run(['ffmpeg', '-version'], 
                              capture_output=True, text=True, timeout=5)
        ffmpeg_ok = result.returncode == 0
        if ffmpeg_ok:
            print("✅ FFmpeg is working")
        else:
            print("❌ FFmpeg not working")
    except:
        ffmpeg_ok = False
        print("❌ FFmpeg not found")
    
    # Summary
    print("\n" + "="*50)
    print("🏁 Setup Complete!")
    print(f"Python packages: {'✅' if python_ok else '❌'}")
    print(f"FFmpeg: {'✅' if ffmpeg_ok else '❌'}")
    
    if python_ok and ffmpeg_ok:
        print("🎉 Everything is ready! You can now:")
        print("   • Run: run_api_server()")
        print("   • Or: start_api_in_background()")
        return True
    else:
        print("⚠️ Some components failed. Check the errors above.")
        return False

if __name__ == "__main__":
    run_api_server()

# Enhanced usage examples:
"""
# Complete setup (installs everything including FFmpeg)
setup_complete_environment()

# Or install just FFmpeg
install_ffmpeg()

# Start server
run_api_server()

# Or run in background
thread = start_api_in_background()

# Test which cookies work
working_browser = quick_fix_cookie_test()

# Enhanced usage examples:
import requests

# Check if FFmpeg is installed
response = requests.get("http://localhost:8000/check-ffmpeg")
print("FFmpeg status:", response.json())

# Install FFmpeg via API if needed
if response.json()["status"] != "available":
    install_response = requests.get("http://localhost:8000/install-ffmpeg")
    print("FFmpeg installation:", install_response.json())

# Use the API with enhanced options:
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