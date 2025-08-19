import os
import tempfile
import requests
from fastapi import FastAPI, Form, HTTPException, Depends
from fastapi.responses import Response
from fastapi.security import APIKeyHeader

# --- ZipVoice imports ---
from zipvoice.infer import ZipVoice, infer_once

app = FastAPI()

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

def get_api_key(api_key: str = Depends(api_key_header)):
    expected_key = os.getenv("API_KEY")
    if not expected_key or api_key != expected_key:
        raise HTTPException(status_code=401, detail="Invalid or missing API Key")
    return api_key


# --- Load model at startup ---
zipvoice_model = None

@app.on_event("startup")
def load_model():
    global zipvoice_model
    model_name = os.getenv("MODEL_NAME", "zipvoice_distill")
    use_onnx = os.getenv("USE_ONNX", "true").lower() == "true"

    zipvoice_model = ZipVoice(model_name=model_name, use_onnx=use_onnx)
    print(f"✅ Loaded ZipVoice model: {model_name} (use_onnx={use_onnx})")


@app.post("/tts")
def generate_tts(
    prompt_text: str = Form(..., description="Transcription of the prompt WAV"),
    prompt_wav_url: str = Form(..., description="URL to the prompt WAV file"),
    text: str = Form(..., description="Text to synthesize"),
    api_key: str = Depends(get_api_key)
):
    global zipvoice_model
    if zipvoice_model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Download prompt WAV
        prompt_wav_path = os.path.join(tmpdir, "prompt.wav")
        response = requests.get(prompt_wav_url)
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail="Failed to download prompt WAV")
        with open(prompt_wav_path, "wb") as f:
            f.write(response.content)

        # Output path
        res_wav_path = os.path.join(tmpdir, "output.wav")

        # Run inference directly
        try:
            infer_once(
                zipvoice_model,
                prompt_text=prompt_text,
                prompt_wav=prompt_wav_path,
                text=text,
                res_wav_path=res_wav_path,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

        # Return the generated WAV
        headers = {"Content-Disposition": 'attachment; filename="output.wav"'}
        return Response(content=open(res_wav_path, "rb").read(), media_type="audio/wav", headers=headers)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
