import os
import subprocess
import tempfile
import shutil
from fastapi import FastAPI, Form, HTTPException, Depends
from fastapi.responses import Response, StreamingResponse
from fastapi.security import APIKeyHeader
import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

def get_api_key(api_key: str = Depends(api_key_header)):
    expected_key = os.getenv("API_KEY")
    if not expected_key or api_key != expected_key:
        raise HTTPException(status_code=401, detail="Invalid or missing API Key")
    return api_key

# Thread pool for CPU-intensive operations
executor = ThreadPoolExecutor(max_workers=2)

# Predefined voices mapping
VOICES_DIR = "voices"  # Directory where voice files are stored
PREDEFINED_VOICES = {
    "voice_one": {
        "wav_path": os.path.join(VOICES_DIR, "voice_one.wav"),
        "transcription": "This is the transcription for voice one"
    },
    "voice_two": {
        "wav_path": os.path.join(VOICES_DIR, "voice_two.wav"),
        "transcription": "This is the transcription for voice two"
    },
    # Add more voices as needed
}

@app.post("/tts")
async def generate_tts(
    voice_name: str = Form(..., description="Name of the predefined voice"),
    text: str = Form(..., description="Text to synthesize"),
    model_name: str = Form(default="zipvoice_distill", description="Model name: zipvoice or zipvoice_distill"),
    use_onnx: bool = Form(default=True, description="Use ONNX for faster CPU inference"),
    api_key: str = Depends(get_api_key)
):
    # Check if voice exists
    if voice_name not in PREDEFINED_VOICES:
        available_voices = list(PREDEFINED_VOICES.keys())
        raise HTTPException(
            status_code=400, 
            detail=f"Voice '{voice_name}' not found. Available voices: {available_voices}"
        )
    
    voice_info = PREDEFINED_VOICES[voice_name]
    
    # Verify the voice file exists
    if not os.path.exists(voice_info["wav_path"]):
        raise HTTPException(
            status_code=500, 
            detail=f"Voice file not found: {voice_info['wav_path']}"
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        # Prepare output path
        res_wav_path = os.path.join(tmpdir, "output.wav")

        # Build the inference command
        module = "zipvoice.bin.infer_zipvoice_onnx" if use_onnx else "zipvoice.bin.infer_zipvoice"
        cmd = [
            "python", "-m", module,
            "--model-name", model_name,
            "--prompt-text", voice_info["transcription"],
            "--prompt-wav", voice_info["wav_path"],
            "--text", text,
            "--res-wav-path", res_wav_path
        ]

        # Run inference in thread pool
        try:
            await asyncio.get_event_loop().run_in_executor(
                executor,
                lambda: run_inference(cmd)
        except Exception as e:
            logger.error(f"Inference failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

        # Stream the response back
        def file_generator():
            with open(res_wav_path, "rb") as f:
                while chunk := f.read(8192):
                    yield chunk

        return StreamingResponse(
            file_generator(),
            media_type="audio/wav",
            headers={"Content-Disposition": f'attachment; filename="{voice_name}_output.wav"'}
        )

def run_inference(cmd):
    """Run the inference command and handle output"""
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise Exception(f"Inference failed: {result.stderr}")
    elif result.stderr:
        logger.warning(f"Inference warnings: {result.stderr}")

# Keep the custom endpoint for URL-based voices if needed
@app.post("/tts/custom")
async def generate_tts_custom(
    model_name: str = Form(default="zipvoice_distill", description="Model name: zipvoice or zipvoice_distill"),
    prompt_text: str = Form(..., description="Transcription of the prompt WAV"),
    prompt_wav_url: str = Form(..., description="URL to the prompt WAV file"),
    text: str = Form(..., description="Text to synthesize"),
    use_onnx: bool = Form(default=True, description="Use ONNX for faster CPU inference"),
    api_key: str = Depends(get_api_key)
):
    # Implementation for custom voices (similar to your original code)
    pass

@app.get("/voices")
def list_voices():
    """Endpoint to list available voices"""
    return {
        "voices": list(PREDEFINED_VOICES.keys()),
        "details": PREDEFINED_VOICES
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))