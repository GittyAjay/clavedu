"""
Enhanced YouTube Audio Transcriber API - Render Optimized
FastAPI-based API with cookie support to bypass YouTube bot detection

Render-specific optimizations:
- Automatic cookie detection for Render environment
- Fallback strategies that work on servers
- Environment variable support for manual cookies
- Disabled browser cookie extraction on Render

Usage:
1. Deploy to Render
2. Set YOUTUBE_COOKIES environment variable if needed
3. API will be available at your Render URL
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
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import browser cookie extraction (disabled on Render)
IS_RENDER = os.environ.get('RENDER', False)
BROWSER_COOKIES_AVAILABLE = False

if not IS_RENDER:
    try:
        import browser_cookie3
        BROWSER_COOKIES_AVAILABLE = True
    except ImportError:
        BROWSER_COOKIES_AVAILABLE = False
        print("⚠️ browser_cookie3 not installed. Run: pip install browser-cookie3")
else:
    print("🔧 Render environment detected - disabling browser cookie extraction")

# Enable nested asyncio for Colab/Jupyter
nest_asyncio.apply()

class TranscriptionRequest(BaseModel):
    url: str
    whisper_model: Optional[str] = "openai/whisper-base"
    output_format: Optional[str] = "text"  # "text", "json", or "both"
    chunk_length_s: Optional[int] = 30
    include_timestamps: Optional[bool] = True
    use_cookies: Optional[bool] = not IS_RENDER  # Auto-disable on Render
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
            version="2.1.0"
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
        print(f"🏭 Render environment: {'✅' if IS_RENDER else '❌'}")
        print(f"🍪 Browser cookies support: {'✅' if BROWSER_COOKIES_AVAILABLE else '❌'}")
        print("✅ CORS middleware enabled")

    def setup_routes(self):
        """Setup FastAPI routes"""

        @self.app.get("/")
        async def root():
            return {
                "message": "Enhanced YouTube Transcriber API",
                "status": "running",
                "environment": "render" if IS_RENDER else "local",
                "cors_enabled": True,
                "cookie_support": BROWSER_COOKIES_AVAILABLE
            }

        @self.app.get("/health")
        async def health_check():
            return {
                "status": "healthy",
                "device": self.device,
                "environment": "render" if IS_RENDER else "local",
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

        @self.app.get("/environment")
        async def environment_info():
            """Get information about the current environment"""
            return {
                "is_render": IS_RENDER,
                "browser_cookies_available": BROWSER_COOKIES_AVAILABLE,
                "python_version": sys.version,
                "platform": platform.platform(),
                "device": self.device
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

    def get_render_cookie_options(self):
        """Get cookie options for Render deployment"""
        # Try to use a pre-uploaded cookie file
        cookie_paths = [
            "/etc/render/cookies.txt",
            "/app/cookies.txt", 
            "./cookies.txt",
            "/tmp/cookies.txt"
        ]
        
        for path in cookie_paths:
            if os.path.exists(path):
                logger.info(f"Found cookie file at {path}")
                return {'cookiefile': path}
        
        # Fallback: try environment variable with cookie content
        cookie_content = os.environ.get('YOUTUBE_COOKIES')
        if cookie_content:
            try:
                temp_dir = tempfile.mkdtemp()
                cookie_file = os.path.join(temp_dir, "cookies.txt")
                with open(cookie_file, 'w') as f:
                    f.write(cookie_content)
                logger.info("Using cookies from YOUTUBE_COOKIES environment variable")
                return {'cookiefile': cookie_file}
            except Exception as e:
                logger.error(f"Error creating cookie file from env var: {e}")
        
        return {}

    def extract_browser_cookies(self, browser: str = "chrome"):
        """Extract cookies from browser - disabled on Render"""
        if IS_RENDER:
            logger.info("Browser cookie extraction disabled on Render")
            return None
            
        if not BROWSER_COOKIES_AVAILABLE:
            return None
            
        try:
            logger.info(f"Extracting cookies from {browser}...")
            
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
                logger.info(f"Found {len(cookies)} cookies from {browser}")
                return cj
            else:
                logger.warning(f"No cookies found in {browser}")
                return None
                
        except Exception as e:
            logger.error(f"Error extracting cookies from {browser}: {e}")
            return None

    def create_cookie_file(self, browser: str = "chrome"):
        """Create a cookie file for yt-dlp - with Render support"""
        if IS_RENDER:
            # Use Render-specific cookie approach
            return self.get_render_cookie_options()
            
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
            logger.error(f"Error creating cookie file: {e}")
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
        
        # Strategy 1: Use browser cookies (disabled on Render)
        if use_cookies and not IS_RENDER and BROWSER_COOKIES_AVAILABLE:
            cookie_file = self.create_cookie_file(browser)
            if cookie_file:
                base_opts['cookiefile'] = cookie_file
                logger.info(f"Using cookies from {browser}")
        
        # Strategy 2: Use Render-specific cookies
        elif use_cookies and IS_RENDER:
            cookie_opts = self.get_render_cookie_options()
            if cookie_opts:
                base_opts.update(cookie_opts)
                logger.info("Using Render-specific cookies")
        
        return base_opts, temp_dir

    def load_whisper_model(self, model_name: str):
        """Load Whisper model with caching"""
        if model_name not in self.models:
            logger.info(f"Loading Whisper model: {model_name}")
            try:
                self.models[model_name] = pipeline(
                    "automatic-speech-recognition",
                    model=model_name,
                    device=0 if self.device == "cuda" else -1,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
                )
                logger.info(f"Model {model_name} loaded successfully")
            except Exception as e:
                logger.error(f"Error loading model {model_name}: {e}")
                raise e
        return self.models[model_name]

    def download_audio_with_fallback(self, url: str, use_cookies: bool = True, browser: str = "chrome") -> tuple:
        """Download audio with multiple fallback strategies optimized for Render"""
        # Auto-disable cookies on Render if not explicitly enabled
        if IS_RENDER and not use_cookies:
            use_cookies = False
            logger.info("Render environment - cookie usage disabled")
        
        strategies = []
        
        # Render-specific strategies first
        if IS_RENDER:
            strategies.append(("render_no_cookies", None))
            strategies.append(("render_with_headers", None))
            strategies.append(("render_aggressive", None))
        
        # Local strategies (only if not on Render)
        if not IS_RENDER:
            if use_cookies:
                strategies.append(("builtin_cookies", browser))
            
            if use_cookies and BROWSER_COOKIES_AVAILABLE:
                strategies.append(("manual_cookies", browser))
        
        # Universal strategies
        strategies.append(("user_agent_mobile", None))
        strategies.append(("user_agent_desktop", None))
        strategies.append(("headers_only", None))
        strategies.append(("basic", None))
        
        last_error = None
        
        for i, (strategy, browser_name) in enumerate(strategies):
            try:
                logger.info(f"Attempt {i+1}: Using {strategy} strategy...")
                
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
                if strategy == "builtin_cookies" and not IS_RENDER:
                    ydl_opts['cookiesfrombrowser'] = (browser_name, None, None, None)
                    logger.info(f"Using yt-dlp built-in cookie extraction from {browser_name}")
                
                elif strategy == "manual_cookies" and not IS_RENDER:
                    cookie_file = self.create_cookie_file(browser_name)
                    if cookie_file:
                        ydl_opts['cookiefile'] = cookie_file
                        logger.info(f"Using manual cookie file from {browser_name}")
                    else:
                        logger.warning("No cookies found, skipping manual cookie strategy")
                        continue
                
                elif strategy == "render_no_cookies":
                    # Minimal options for Render
                    ydl_opts.update({
                        'http_headers': {
                            'User-Agent': 'Mozilla/5.0 (compatible; Render-Transcriber/1.0)',
                            'Accept': '*/*'
                        }
                    })
                
                elif strategy == "render_with_headers":
                    # More comprehensive headers for Render
                    ydl_opts.update({
                        'http_headers': {
                            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                            'Accept-Language': 'en-US,en;q=0.5',
                            'Accept-Encoding': 'gzip, deflate',
                            'DNT': '1',
                            'Connection': 'keep-alive',
                        }
                    })
                
                elif strategy == "render_aggressive":
                    # Aggressive approach for difficult videos
                    ydl_opts.update({
                        'http_headers': {
                            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                            'Accept-Language': 'en-US,en;q=0.5',
                            'Accept-Encoding': 'gzip, deflate, br',
                            'DNT': '1',
                            'Connection': 'keep-alive',
                            'Upgrade-Insecure-Requests': '1',
                            'Sec-Fetch-Dest': 'document',
                            'Sec-Fetch-Mode': 'navigate',
                            'Sec-Fetch-Site': 'none',
                            'Sec-Fetch-User': '?1',
                        }
                    })
                
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

                    logger.info(f"Video: {video_info['title']}")
                    logger.info(f"Duration: {video_info['duration']} seconds")
                    
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

                logger.info(f"Success with {strategy} strategy!")
                return downloaded_file, video_info, temp_dir, strategy

            except Exception as e:
                last_error = e
                logger.error(f"{strategy} strategy failed: {str(e)}")
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

            logger.info("Transcribing audio...")

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
                    "cookies_used": request.use_cookies and "cookies" in strategy,
                    "browser": request.browser if "cookies" in strategy else None,
                    "environment": "render" if IS_RENDER else "local"
                }
            }

            if request.output_format in ["text", "both"]:
                response_data["text"] = text

            if request.output_format in ["json", "both"]:
                response_data["detailed"] = result

            logger.info(f"Transcription completed for: {video_info['title']}")
            return TranscriptionResponse(**response_data)

        except Exception as e:
            logger.error(f"Error in transcription: {str(e)}")
            
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

# ... (keep the rest of your utility functions the same: install_ffmpeg, install_dependencies, run_api_server, etc.)
# The utility functions remain unchanged from your original code

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

    print("🎬 Enhanced YouTube Transcriber API (Render Optimized)")
    print("=" * 60)
    print(f"🏠 Local URL: http://localhost:{port}")
    print(f"🌐 Network URL: http://{host}:{port}")
    print(f"🏭 Environment: {'Render' if IS_RENDER else 'Local'}")
    print("✅ CORS enabled - web browsers can access this API")
    print(f"🍪 Cookie support: {'✅' if BROWSER_COOKIES_AVAILABLE else '❌'}")
    print("📋 Available endpoints:")
    print(f"  GET  http://localhost:{port}/ - API info")
    print(f"  GET  http://localhost:{port}/health - Health check")
    print(f"  POST http://localhost:{port}/transcribe - Transcribe video")
    print(f"  GET  http://localhost:{port}/models - Available models")
    print(f"  GET  http://localhost:{port}/environment - Environment info")
    print("=" * 60)
    print("💡 Render Tips:")
    print("  - Set YOUTUBE_COOKIES environment variable for authentication")
    print("  - Use use_cookies=false for public videos")
    print("  - Age-restricted videos need manual cookie setup")
    print("=" * 60)

    # Run server
    uvicorn.run(transcriber_api.app, host=host, port=port, log_level="info")

if __name__ == "__main__":
    run_api_server()
