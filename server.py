import os
import subprocess
import tempfile
import shutil
from fastapi import FastAPI, Form, HTTPException, Depends
from fastapi.security import APIKeyHeader
import requests

app = FastAPI()

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

def get_api_key(api_key: str = Depends(api_key_header)):
    expected_key = os.getenv("API_KEY")
    if not expected_key or api_key != expected_key:
        raise HTTPException(status_code=401, detail="Invalid or missing API Key")
    return api_key

@app.post("/tts")
def generate_tts(
    model_name: str = Form(default="zipvoice_distill", description="Model name: zipvoice or zipvoice_distill"),
    prompt_text: str = Form(..., description="Transcription of the prompt WAV"),
    prompt_wav_url: str = Form(..., description="URL to the prompt WAV file"),
    text: str = Form(..., description="Text to synthesize"),
    use_onnx: bool = Form(default=True, description="Use ONNX for faster CPU inference"),
    api_key: str = Depends(get_api_key)
):
    with tempfile.TemporaryDirectory() as tmpdir:
        # Download prompt WAV from URL
        prompt_wav_path = os.path.join(tmpdir, "prompt.wav")
        response = requests.get(prompt_wav_url)
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail="Failed to download prompt WAV from URL")
        with open(prompt_wav_path, "wb") as f:
            f.write(response.content)

        # Prepare output path
        res_wav_path = os.path.join(tmpdir, "output.wav")

        # Build the inference command
        module = "zipvoice.bin.infer_zipvoice_onnx" if use_onnx else "zipvoice.bin.infer_zipvoice"
        cmd = [
            "python", "-m", module,
            "--model-name", model_name,
            "--prompt-text", prompt_text,
            "--prompt-wav", prompt_wav_path,
            "--text", text,
            "--res-wav-path", res_wav_path
        ]

        # Run inference
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise HTTPException(status_code=500, detail=f"Inference failed: {result.stderr}")
        # If returncode == 0, log stderr as warning but continue
        elif result.stderr:
            print("Inference warnings:", result.stderr)


        # Return the generated WAV
        headers = {"Content-Disposition": 'attachment; filename="output.wav"'}
        return Response(content=open(res_wav_path, "rb").read(), media_type="audio/wav", headers=headers)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))