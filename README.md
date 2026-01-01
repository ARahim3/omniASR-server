# omniASR Streaming Server

An OpenAI-compatible ASR (Automatic Speech Recognition) API server powered by Meta's [omniASR](https://github.com/facebookresearch/omnilingual-asr) model. Supports real-time streaming via WebSocket and batch transcription via REST API.

## Features

- **OpenAI-Compatible API** - Drop-in replacement for OpenAI's `/v1/audio/transcriptions` endpoint
- **Real-time Streaming** - WebSocket support for live transcription
- **Long Audio Support** - Automatically handles files longer than 40 seconds
- **Multi-device Support** - CUDA (NVIDIA), MPS (Apple Silicon), CPU
- **Voice Agent Ready** - Works with Pipecat, LiveKit, and other frameworks
- **Docker Support** - One-command deployment with GPU support

## Quick Start

### Option 1: Local Installation (Supports CUDA/MPS/CPU)

```bash
# Clone the repository
git clone https://github.com/ARahim3/omniASR-server.git
cd omniASR-server

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the server
python server.py
```

Server starts at `http://localhost:8000`

### Option 2: Docker (Recommended for Production) (Supports CUDA and CPU, no MPS)

```bash
# With NVIDIA GPU
docker compose up -d

# CPU only
docker compose --profile cpu up -d
```

## Usage

### REST API (OpenAI Compatible)

```bash
# Transcribe an audio file
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F file=@audio.wav \
  -F model=omniASR_CTC_300M_v2

# Response
{"text": "Hello world, this is a test."}
```

### Python (OpenAI SDK)

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"  # API key not required
)

with open("audio.wav", "rb") as f:
    transcript = client.audio.transcriptions.create(
        model="omniASR_CTC_300M_v2",
        file=f
    )
print(transcript.text)
```

### WebSocket Streaming

```python
import asyncio
import websockets
import json

async def stream_audio():
    async with websockets.connect("ws://localhost:8000/v1/audio/transcriptions") as ws:
        # Wait for ready
        ready = await ws.recv()
        print(f"Server ready: {ready}")

        # Send audio chunks (16kHz, 16-bit PCM, mono)
        with open("audio.raw", "rb") as f:
            while chunk := f.read(3200):  # 100ms chunks
                await ws.send(chunk)

        # Send end signal
        await ws.send(json.dumps({"type": "end"}))

        # Receive transcriptions
        async for message in ws:
            data = json.loads(message)
            print(f"Transcript: {data['text']}")
            if data.get("is_final"):
                break

asyncio.run(stream_audio())
```

### Voice Agent Integration (Pipecat)

```python
from pipecat.services.openai.stt import OpenAISTTService

stt = OpenAISTTService(
    base_url="http://localhost:8000/v1",
    api_key="not-needed",
)
```

## API Reference

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/audio/transcriptions` | POST | Transcribe audio file |
| `/v1/audio/transcriptions` | WebSocket | Real-time streaming |
| `/health` | GET | Health check |

### POST /v1/audio/transcriptions

**Request:**
```
Content-Type: multipart/form-data

file: <audio file>
model: omniASR_CTC_300M_v2 (optional)
language: eng_Latn (optional)
response_format: json | text | verbose_json (optional)
```

**Example 1: Basic transcription**
```bash
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F file=@audio.wav
```
```json
{"text": "transcribed text here"}
```

**Example 2: With metrics (for developers)**
```bash
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F file=@audio.wav \
  -F response_format=verbose_json | jq
```
```json
{
  "text": "transcribed text here",
  "language": "eng_Latn",
  "duration": 5.234,
  "model": "omniASR_CTC_300M_v2",
  "processing_time": 0.847,
  "rtf": 0.1618
}
```

### WebSocket /v1/audio/transcriptions

**Protocol:**
1. Connect to WebSocket
2. Receive `{"type": "ready", ...}` message
3. Send raw PCM audio (16kHz, 16-bit, mono)
4. Receive transcription updates: `{"text": "...", "is_final": false}`
5. Send `{"type": "end"}` to finish
6. Receive final transcription with `"is_final": true`

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_CARD` | `omniASR_CTC_300M_v2` | Model to use |
| `DEFAULT_LANG` | `eng_Latn` | Default language (empty for auto-detect) |
| `DEVICE` | auto | `cuda`, `mps`, `cpu`, or auto-detect |
| `HOST` | `0.0.0.0` | Server host |
| `PORT` | `8000` | Server port |
| `CHUNK_DURATION` | `5.0` | Streaming chunk size (seconds) |
| `VAD_ENABLED` | `true` | Voice activity detection |

### Using .env file

```bash
cp .env.example .env
# Edit .env with your settings
```

### Available Models

This server supports all models from [Meta's omniASR](https://github.com/facebookresearch/omnilingual-asr), some of these are:

| Model | Type | Parameters | Speed (RTF) | Use Case |
|-------|------|------------|-------------|----------|
| `omniASR_CTC_300M_v2` | CTC | 300M | 0.001 | Fast, good for streaming |
| `omniASR_CTC_1B_v2` | CTC | 1B | 0.003 | Better accuracy |
| `omniASR_CTC_7B_v2` | CTC | 7B | 0.006 | Best accuracy |
| `omniASR_LLM_1B_v2` | LLM | 1B | 0.09 | Language conditioning |
| `omniASR_LLM_7B_v2` | LLM | 7B | 0.10 | Best with context |

*RTF (Real-Time Factor) measured on A100 GPU with batch=1, 30s audio*

> **Note:** CTC models are faster and recommended for streaming. LLM models support language conditioning but are slower. See [omniASR docs](https://github.com/facebookresearch/omnilingual-asr) for full model list.

## Deployment

### Docker Compose

```yaml
# docker-compose.yml is included
# GPU deployment
docker compose up -d

# CPU deployment
docker compose --profile cpu up -d

# Custom configuration
MODEL_CARD=omniASR_CTC_1B_v2 docker compose up -d
```

### Manual Docker Build

```bash
docker build -t omniasr-server .
docker run -d -p 8000:8000 --gpus all omniasr-server
```

### Kubernetes

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: omniASR-server
spec:
  replicas: 1
  selector:
    matchLabels:
      app: omniASR-server
  template:
    metadata:
      labels:
        app: omniASR-server
    spec:
      containers:
      - name: omniASR
        image: omniASR-server:latest
        ports:
        - containerPort: 8000
        env:
        - name: MODEL_CARD
          value: "omniASR_CTC_300M_v2"
        resources:
          limits:
            nvidia.com/gpu: 1
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                 omniASR Streaming Server                    │
├──────────────────────┬──────────────────────────────────────┤
│   REST API           │   WebSocket API                      │
│   (OpenAI compat)    │   (real-time streaming)              │
├──────────────────────┴──────────────────────────────────────┤
│                   StreamingTranscriber                      │
│  ┌───────────┐  ┌──────────────┐  ┌──────────────────────┐ │
│  │  Audio    │→ │   Chunked    │→ │   LocalAgreement     │ │
│  │  Buffer   │  │   Inference  │  │   (stable output)    │ │
│  └───────────┘  └──────────────┘  └──────────────────────┘ │
├─────────────────────────────────────────────────────────────┤
│              omniASR CTC Model (CUDA/MPS/CPU)               │
└─────────────────────────────────────────────────────────────┘
```

### Key Components

- **AudioBuffer** - Ring buffer with overlap chunking for streaming
- **AudioChunker** - VAD-based segmentation for long files
- **LocalAgreement** - Stabilizes streaming output (prevents flickering)
- **ASRModel** - Wrapper with auto device detection and long audio support

## Performance

### Latency (Streaming)

| Device | Chunk Size | Latency |
|--------|------------|---------|
| A100 GPU | 5s | ~50ms |
| RTX 4050 (laptop) | 5s | ~100ms |
| M4 Pro (MPS) | 5s | ~50ms |
| CPU | 5s | ~5s |

### Throughput

| Device | RTF | 1 hour audio processed in |
|--------|-----|---------------------------|
| A100 GPU | 0.001 | 3.6 seconds |
| RTX 4050 (laptop) | 0.01 | 36 seconds |
| M4 Pro (MPS) | 0.0095 | ~6 minutes |
| CPU | ~1.0 | 1 hour |

## Troubleshooting

### Model not loading

```bash
# Check if model is downloading
docker compose logs -f

# Manually download model
python -c "from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline; ASRInferencePipeline('omniASR_CTC_300M_v2')"
```

### CUDA out of memory

```bash
# Use smaller model
MODEL_CARD=omniASR_CTC_300M_v2 docker compose up -d

# Or reduce batch size
BATCH_SIZE=1 docker compose up -d
```

### Poor transcription quality in streaming

1. Increase chunk duration: `CHUNK_DURATION=5.0`
2. Force language: `DEFAULT_LANG=eng_Latn`
3. Use larger model: `MODEL_CARD=omniASR_CTC_1B_v2`

### WebSocket connection issues

```bash
# Check server health
curl http://localhost:8000/health

# Test WebSocket
python test_streaming.py websocket
```

## Development

### Project Structure

```
omniASR_server/
├── server.py            # FastAPI app
├── config.py            # Configuration (env vars)
├── model.py             # ASR model wrapper
├── streaming.py         # Streaming transcriber
├── audio_buffer.py      # Audio buffering
├── audio_chunker.py     # Long audio chunking
├── local_agreement.py   # Output stabilization
├── schemas.py           # API schemas
├── test_streaming.py    # Test script
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

### Running Tests

```bash
# Test microphone (local, no server)
python test_streaming.py mic

# Test REST API
python test_streaming.py rest --file audio.wav

# Test WebSocket
python test_streaming.py websocket
```

## License

MIT License - see [LICENSE](LICENSE)

## Acknowledgments

- [Meta's omniASR](https://github.com/facebookresearch/omnilingual-asr) - The underlying ASR model
- [faster-whisper-server](https://github.com/etalab-ia/faster-whisper-server) - Inspiration for streaming architecture
- [whisper_streaming](https://github.com/ufal/whisper_streaming) - LocalAgreement algorithm

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.
