"""
omniASR Streaming Server - OpenAI-compatible ASR API.

Endpoints:
- POST /v1/audio/transcriptions - Transcribe audio file (OpenAI compatible)
- WS /v1/audio/transcriptions - Streaming transcription via WebSocket
- GET /health - Health check
"""

import asyncio
import json
import tempfile
from pathlib import Path
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, File, Form, UploadFile, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, JSONResponse

from config import config
from model import ASRModel
from streaming import AsyncStreamingTranscriber
from schemas import (
    ResponseFormat,
    TranscriptionResponse,
    VerboseTranscriptionResponse,
    StreamingMessage,
    HealthResponse,
)


# Global model instance
model: ASRModel = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup."""
    global model
    model = ASRModel.get_instance()

    # Load model in background
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, model.load)

    yield

    # Cleanup (if needed)


app = FastAPI(
    title="omniASR Streaming Server",
    description="OpenAI-compatible ASR API with streaming support",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="ok",
        model_loaded=model.is_loaded if model else False,
        device=model.device_name if model else "unknown",
        model_name=config.model.model_card,
    )


@app.post("/v1/audio/transcriptions")
async def transcribe_audio(
    file: UploadFile = File(...),
    model_name: str = Form(default=None, alias="model"),
    language: str = Form(default=None),
    response_format: str = Form(default="json"),
):
    """
    Transcribe audio file (OpenAI-compatible endpoint).

    Accepts audio file upload and returns transcription.
    """
    global model

    if not model or not model.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Validate model parameter (if provided)
    if model_name and model_name != model.model_card:
        raise HTTPException(
            status_code=400,
            detail=f"Requested model '{model_name}' not available. Server is running '{model.model_card}'"
        )

    # Validate format
    try:
        fmt = ResponseFormat(response_format)
    except ValueError:
        fmt = ResponseFormat.JSON

    # Save uploaded file
    suffix = Path(file.filename).suffix if file.filename else ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        content = await file.read()
        f.write(content)
        temp_path = f.name

    try:
        # Transcribe (use long audio method to handle files > 40s)
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: model.transcribe_long_file(temp_path, lang=language)
        )

        # Return based on format
        if fmt == ResponseFormat.TEXT:
            return PlainTextResponse(result.text)
        elif fmt == ResponseFormat.VERBOSE_JSON:
            # Use provided language or fall back to config default
            effective_lang = language or config.model.default_lang
            response_data = {
                "text": result.text,
                "language": effective_lang,
                "duration": round(result.duration, 3),
                "model": model.model_card,
                "processing_time": round(result.latency, 3),
                "rtf": round(result.rtf, 4),
            }
            # Return with indentation for readability
            return JSONResponse(
                content=response_data,
                media_type="application/json"
            )
        else:
            return TranscriptionResponse(text=result.text)

    finally:
        # Cleanup
        Path(temp_path).unlink(missing_ok=True)


@app.websocket("/v1/audio/transcriptions")
async def websocket_transcription(websocket: WebSocket):
    """
    WebSocket endpoint for streaming transcription.

    Protocol:
    1. Client connects
    2. Client sends config message (optional): {"type": "config", "sample_rate": 16000, ...}
    3. Client sends raw PCM audio chunks (16-bit, mono)
    4. Server sends transcription updates: {"text": "...", "is_final": false, ...}
    5. Client sends {"type": "end"} or closes connection
    """
    await websocket.accept()

    transcriber = AsyncStreamingTranscriber()
    sample_rate = config.audio.sample_rate

    try:
        await transcriber.start()

        # Send ready message
        await websocket.send_json({
            "type": "ready",
            "sample_rate": sample_rate,
            "model": config.model.model_card,
        })

        while True:
            # Receive message
            message = await websocket.receive()

            if message["type"] == "websocket.disconnect":
                break

            # Handle text messages (config, control)
            if "text" in message:
                try:
                    data = json.loads(message["text"])

                    if data.get("type") == "config":
                        sample_rate = data.get("sample_rate", sample_rate)
                        continue

                    if data.get("type") == "end":
                        # End stream and send final result
                        final = await transcriber.end()
                        if final:
                            await websocket.send_json(
                                StreamingMessage(
                                    text=final.text,
                                    confirmed_text=final.confirmed_text,
                                    pending_text=final.pending_text,
                                    is_final=True,
                                    latency_ms=final.latency_ms,
                                    audio_duration=final.audio_duration,
                                ).model_dump()
                            )
                        break

                except json.JSONDecodeError:
                    pass

            # Handle binary messages (audio data)
            if "bytes" in message:
                audio_bytes = message["bytes"]

                # Convert to numpy array (assuming 16-bit PCM)
                audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

                # Process and get results
                results = await transcriber.process(audio)

                # Send results
                for result in results:
                    await websocket.send_json(
                        StreamingMessage(
                            text=result.text,
                            confirmed_text=result.confirmed_text,
                            pending_text=result.pending_text,
                            is_final=result.is_final,
                            latency_ms=result.latency_ms,
                            audio_duration=result.audio_duration,
                        ).model_dump()
                    )

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except:
            pass
    finally:
        transcriber.reset()


@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "name": "omniASR Streaming Server",
        "version": "0.1.0",
        "endpoints": {
            "health": "GET /health",
            "transcribe": "POST /v1/audio/transcriptions",
            "stream": "WS /v1/audio/transcriptions",
        },
        "model": config.model.model_card,
    }


def main():
    """Run the server."""
    import uvicorn
    uvicorn.run(
        "server:app",
        host=config.server.host,
        port=config.server.port,
        reload=False,
    )


if __name__ == "__main__":
    main()
