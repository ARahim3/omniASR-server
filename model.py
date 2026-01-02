"""omniASR model wrapper with device detection and caching."""

import tempfile
import time
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import torch

from config import config


@dataclass
class TranscribeResult:
    """Result from transcription."""
    text: str
    duration: float      # audio duration in seconds
    latency: float       # inference time in seconds
    rtf: float           # real-time factor


def get_device() -> str:
    """Get best available device (can be overridden via config/env)."""
    # Check if device is explicitly set in config
    if config.model.device:
        return config.model.device

    # Auto-detect
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class ASRModel:
    """
    Wrapper for omniASR pipeline with lazy loading and device management.
    """

    _instance: "ASRModel | None" = None

    def __init__(
        self,
        model_card: str = None,
        device: str = None,
        lang: str = None,
    ):
        self.model_card = model_card or config.model.model_card
        self.device = device or get_device()
        self.lang = lang if lang is not None else config.model.default_lang

        self._pipeline = None
        self._load_time: float = 0

    @classmethod
    def get_instance(cls) -> "ASRModel":
        """Get singleton instance of ASRModel."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton (for testing)."""
        cls._instance = None

    def load(self) -> None:
        """Load the model (lazy loading)."""
        if self._pipeline is not None:
            return

        from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline

        print(f"Loading ASR model '{self.model_card}' on {self.device}...")
        start = time.perf_counter()
        self._pipeline = ASRInferencePipeline(
            model_card=self.model_card,
            device=self.device,
        )
        self._load_time = time.perf_counter() - start
        print(f"Model loaded in {self._load_time:.2f}s")

    def ensure_loaded(self) -> None:
        """Ensure model is loaded."""
        if self._pipeline is None:
            self.load()

    def transcribe_file(self, audio_path: str | Path, lang: str = None) -> TranscribeResult:
        """
        Transcribe audio file.

        Args:
            audio_path: Path to audio file
            lang: Language code (e.g., "eng_Latn") or None for auto-detect

        Returns:
            TranscribeResult with text and timing info
        """
        self.ensure_loaded()

        # Get audio duration
        import soundfile as sf
        info = sf.info(str(audio_path))
        duration = info.duration

        # Transcribe
        lang_param = lang if lang is not None else self.lang
        start = time.perf_counter()
        result = self._pipeline.transcribe(
            [str(audio_path)],
            lang=[lang_param] if lang_param else None,
            batch_size=config.model.batch_size,
        )
        latency = time.perf_counter() - start

        text = result[0] if result else ""

        return TranscribeResult(
            text=text,
            duration=duration,
            latency=latency,
            rtf=latency / duration if duration > 0 else 0,
        )

    def transcribe_audio(
        self,
        audio: np.ndarray,
        sample_rate: int = None,
        lang: str = None,
    ) -> TranscribeResult:
        """
        Transcribe audio array.

        Args:
            audio: Audio samples (numpy array)
            sample_rate: Sample rate (default from config)
            lang: Language code or None for auto-detect

        Returns:
            TranscribeResult with text and timing info
        """
        self.ensure_loaded()

        sample_rate = sample_rate or config.audio.sample_rate
        duration = len(audio) / sample_rate

        # Save to temp file (omniASR requires file path)
        import soundfile as sf
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, audio.astype(np.float32), sample_rate)
            temp_path = f.name

        try:
            # Transcribe
            lang_param = lang if lang is not None else self.lang
            start = time.perf_counter()
            result = self._pipeline.transcribe(
                [temp_path],
                lang=[lang_param] if lang_param else None,
                batch_size=config.model.batch_size,
            )
            latency = time.perf_counter() - start

            text = result[0] if result else ""

            return TranscribeResult(
                text=text,
                duration=duration,
                latency=latency,
                rtf=latency / duration if duration > 0 else 0,
            )
        finally:
            # Cleanup temp file
            Path(temp_path).unlink(missing_ok=True)

    def transcribe_long_file(
        self,
        audio_path: str | Path,
        lang: str = None,
        max_chunk_duration: float = 35.0,
    ) -> TranscribeResult:
        """
        Transcribe a long audio file by chunking.

        Automatically handles files longer than the model's limit (40s)
        by splitting at natural boundaries (silences) and concatenating results.

        Args:
            audio_path: Path to audio file
            lang: Language code or None for auto-detect
            max_chunk_duration: Maximum chunk duration (default 35s for safety)

        Returns:
            TranscribeResult with concatenated text and total timing info
        """
        self.ensure_loaded()

        import soundfile as sf
        from audio_chunker import AudioChunker, ChunkerConfig, TranscriptionSegment, merge_transcriptions

        # Load audio file
        audio, sample_rate = sf.read(str(audio_path), dtype='float32')
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)  # Convert stereo to mono

        total_duration = len(audio) / sample_rate

        # If short enough, use regular transcription
        if total_duration <= max_chunk_duration:
            return self.transcribe_file(audio_path, lang=lang)

        # Chunk the audio
        chunker = AudioChunker(ChunkerConfig(max_chunk_duration=max_chunk_duration))
        segments = chunker.chunk_audio(audio, sample_rate)

        print(f"Long audio ({total_duration:.1f}s) split into {len(segments)} chunks")

        # Transcribe each chunk
        transcription_segments = []
        total_latency = 0.0

        for i, segment in enumerate(segments):
            print(f"  Transcribing chunk {i+1}/{len(segments)} ({segment.duration:.1f}s)...")

            result = self.transcribe_audio(
                segment.audio,
                sample_rate=segment.sample_rate,
                lang=lang,
            )

            total_latency += result.latency

            transcription_segments.append(TranscriptionSegment(
                text=result.text,
                start_time=segment.start_time,
                end_time=segment.end_time,
            ))

        # Merge transcriptions
        merged_text = merge_transcriptions(transcription_segments)

        return TranscribeResult(
            text=merged_text,
            duration=total_duration,
            latency=total_latency,
            rtf=total_latency / total_duration if total_duration > 0 else 0,
        )

    def transcribe_long_audio(
        self,
        audio: np.ndarray,
        sample_rate: int = None,
        lang: str = None,
        max_chunk_duration: float = 35.0,
    ) -> TranscribeResult:
        """
        Transcribe a long audio array by chunking.

        Args:
            audio: Audio samples (numpy array)
            sample_rate: Sample rate (default from config)
            lang: Language code or None for auto-detect
            max_chunk_duration: Maximum chunk duration

        Returns:
            TranscribeResult with concatenated text
        """
        self.ensure_loaded()

        from audio_chunker import AudioChunker, ChunkerConfig, TranscriptionSegment, merge_transcriptions

        sample_rate = sample_rate or config.audio.sample_rate

        # Normalize audio
        audio = np.asarray(audio, dtype=np.float32).flatten()
        if np.abs(audio).max() > 1.0:
            audio = audio / 32768.0

        total_duration = len(audio) / sample_rate

        # If short enough, use regular transcription
        if total_duration <= max_chunk_duration:
            return self.transcribe_audio(audio, sample_rate=sample_rate, lang=lang)

        # Chunk the audio
        chunker = AudioChunker(ChunkerConfig(max_chunk_duration=max_chunk_duration))
        segments = chunker.chunk_audio(audio, sample_rate)

        # Transcribe each chunk
        transcription_segments = []
        total_latency = 0.0

        for segment in segments:
            result = self.transcribe_audio(
                segment.audio,
                sample_rate=segment.sample_rate,
                lang=lang,
            )
            total_latency += result.latency

            transcription_segments.append(TranscriptionSegment(
                text=result.text,
                start_time=segment.start_time,
                end_time=segment.end_time,
            ))

        # Merge transcriptions
        merged_text = merge_transcriptions(transcription_segments)

        return TranscribeResult(
            text=merged_text,
            duration=total_duration,
            latency=total_latency,
            rtf=total_latency / total_duration if total_duration > 0 else 0,
        )

    def transcribe_long_file_streaming(
        self,
        audio_path: str | Path,
        lang: str = None,
        max_chunk_duration: float = 35.0,
    ):
        """
        Generator that yields transcription results as chunks are processed.

        Useful for SSE streaming - yields progress for long files.

        Yields:
            dict with keys: text, chunk_index, total_chunks, is_final, duration, processing_time
        """
        self.ensure_loaded()

        import soundfile as sf
        from audio_chunker import AudioChunker, ChunkerConfig

        # Load audio file
        audio, sample_rate = sf.read(str(audio_path), dtype='float32')
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)

        total_duration = len(audio) / sample_rate

        # If short enough, yield single result
        if total_duration <= max_chunk_duration:
            result = self.transcribe_file(audio_path, lang=lang)
            yield {
                "text": result.text,
                "chunk_index": 1,
                "total_chunks": 1,
                "is_final": True,
                "duration": result.duration,
                "processing_time": result.latency,
                "rtf": result.rtf,
            }
            return

        # Chunk the audio
        chunker = AudioChunker(ChunkerConfig(max_chunk_duration=max_chunk_duration))
        segments = chunker.chunk_audio(audio, sample_rate)
        total_chunks = len(segments)

        accumulated_text = []
        total_processing_time = 0.0

        for i, segment in enumerate(segments):
            result = self.transcribe_audio(
                segment.audio,
                sample_rate=segment.sample_rate,
                lang=lang,
            )
            total_processing_time += result.latency

            if result.text.strip():
                accumulated_text.append(result.text.strip())

            is_final = (i == total_chunks - 1)

            yield {
                "text": " ".join(accumulated_text),
                "chunk_index": i + 1,
                "total_chunks": total_chunks,
                "is_final": is_final,
                "duration": segment.end_time,
                "processing_time": total_processing_time,
                "rtf": total_processing_time / segment.end_time if segment.end_time > 0 else 0,
            }

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._pipeline is not None

    @property
    def load_time(self) -> float:
        """Get model load time."""
        return self._load_time

    @property
    def device_name(self) -> str:
        """Get device name."""
        return self.device
