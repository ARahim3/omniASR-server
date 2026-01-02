"""OpenAI-compatible request/response schemas."""

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class ResponseFormat(str, Enum):
    """Supported response formats."""
    JSON = "json"
    TEXT = "text"
    VERBOSE_JSON = "verbose_json"
    SSE = "sse"  # Server-Sent Events for streaming long audio


class TranscriptionRequest(BaseModel):
    """Request for audio transcription (matches OpenAI API)."""
    model: str = Field(default="omniASR_CTC_300M_v2", description="Model to use")
    language: Optional[str] = Field(default=None, description="Language code (e.g., 'eng_Latn')")
    response_format: ResponseFormat = Field(default=ResponseFormat.JSON)
    temperature: Optional[float] = Field(default=None, description="Not used, for compatibility")
    prompt: Optional[str] = Field(default=None, description="Not used, for compatibility")


class TranscriptionResponse(BaseModel):
    """Response for audio transcription (matches OpenAI API)."""
    text: str


class VerboseTranscriptionResponse(BaseModel):
    """Verbose response with additional info including timing metrics."""
    text: str
    language: Optional[str] = None
    duration: Optional[float] = None          # audio duration in seconds
    model: str = "omniASR_CTC_300M_v2"
    # Timing metrics (useful for developers)
    processing_time: Optional[float] = None   # inference time in seconds
    rtf: Optional[float] = None               # real-time factor (lower = faster)


class StreamingMessage(BaseModel):
    """Message format for WebSocket streaming."""
    text: str
    confirmed_text: str = ""
    pending_text: str = ""
    is_final: bool = False
    latency_ms: float = 0
    audio_duration: float = 0


class WebSocketConfig(BaseModel):
    """Configuration message for WebSocket connection."""
    type: str = "config"
    sample_rate: int = 16000
    language: Optional[str] = None
    model: str = "omniASR_CTC_300M_v2"


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = "ok"
    model_loaded: bool = False
    device: str = "cpu"
    model_name: str = "omniASR_CTC_300M_v2"
    # Connection stats
    active_requests: Optional[int] = None
    active_websockets: Optional[int] = None
