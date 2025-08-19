import os
import subprocess
import tempfile
import asyncio
import logging
import uuid
import io
from fastapi import FastAPI, Form, HTTPException, Depends, status
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.security import APIKeyHeader
from concurrent.futures import ThreadPoolExecutor
from redis import Redis
import rq

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

# Redis and RQ setup
redis_conn = Redis.from_url(os.getenv("REDIS_URL"))
queue = rq.Queue(connection=redis_conn)

# Thread pool for CPU-intensive operations (used inside worker if needed)
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

def run_inference(cmd):
    """Run the inference command and handle output"""
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise Exception(f"Inference failed: {result.stderr}")
    elif result.stderr:
        logger.warning(f"Inference warnings: {result.stderr}")

def inference_task(voice_info, text, model_name, use_onnx, voice_name):
    """Task function to run in RQ worker"""
    job = rq.get_current_job()
    job_id = job.id  # For logging
    logger.info(f"[{job_id}] Starting inference task")

    fd, res_wav_path = tempfile.mkstemp(suffix='.wav')
    try:
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

        logger.info(f"[{job_id}] Running inference command")
        run_inference(cmd)

        with open(res_wav_path, "rb") as f:
            wav_bytes = f.read()

        job.meta['status'] = 'finished'
        job.meta['wav_bytes'] = wav_bytes
        job.meta['voice_name'] = voice_name
        job.save_meta()
        logger.info(f"[{job_id}] Inference task completed successfully")
    except Exception as e:
        logger.error(f"[{job_id}] Inference task failed: {str(e)}")
        raise  # RQ will mark as failed
    finally:
        os.close(fd)
        if os.path.exists(res_wav_path):
            os.remove(res_wav_path)

@app.post("/tts", status_code=status.HTTP_202_ACCEPTED)
async def generate_tts(
    voice_name: str = Form(..., description="Name of the predefined voice"),
    text: str = Form(..., description="Text to synthesize"),
    model_name: str = Form(default="zipvoice_distill", description="Model name: zipvoice or zipvoice_distill"),
    use_onnx: bool = Form(default=True, description="Use ONNX for faster CPU inference"),
    api_key: str = Depends(get_api_key)
):
    request_id = str(uuid.uuid4())[:8]
    logger.info(f"[{request_id}] Request started: voice_name={voice_name}, text='{text[:50]}...', model={model_name}, onnx={use_onnx}")

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

    # Enqueue the job
    job = queue.enqueue(inference_task, voice_info, text, model_name, use_onnx, voice_name)
    logger.info(f"[{request_id}] Job enqueued: {job.id}")

    return JSONResponse({"job_id": job.id, "status": "queued"})

@app.get("/jobs/{job_id}")
async def get_job_result(job_id: str):
    try:
        job = rq.Job.fetch(job_id, connection=redis_conn)
    except rq.exceptions.NoSuchJobError:
        raise HTTPException(status_code=404, detail="Job not found")

    status = job.get_status()
    if status == 'failed':
        raise HTTPException(status_code=500, detail=f"Job failed: {job.exc_info}")

    if status != 'finished':
        return JSONResponse({"status": status})

    # Job is finished: stream the result and cleanup
    wav_bytes = job.meta['wav_bytes']
    voice_name = job.meta.get('voice_name', 'output')

    def file_generator():
        yield wav_bytes
        # Cleanup job after successful fetch
        job.delete()

    return StreamingResponse(
        file_generator(),
        media_type="audio/wav",
        headers={"Content-Disposition": f'attachment; filename="{voice_name}_output.wav"'}
    )

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
    raise HTTPException(status_code=501, detail="Custom TTS endpoint not implemented yet")

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