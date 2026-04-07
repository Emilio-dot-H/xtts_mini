#!/usr/bin/env python3
"""
xtts_minimal_tutorial.py

A very barebones, heavily commented example of how to generate narration with
Coqui TTS using XTTS v2.

GOAL
----
This file is intentionally simple. It is *not* meant to replace your advanced
`run_xtts.py` pipeline. Instead, it shows the foundational structure of a
minimal XTTS narration program:

    1) Read text from a file
    2) Load a TTS model
    3) Provide one or more reference speaker WAV files
    4) Generate audio
    5) Write a WAV file to disk

WHY THIS EXISTS
---------------
Your production pipeline does a lot more:
- sentence-aware chunking
- retry logic
- pause normalization
- trimming
- risk detection
- debug artifact generation
- end-of-sentence protection

This tutorial file deliberately strips all of that away so you can see the
core technology more clearly.

OFFICIAL DOCUMENTATION BASIS
----------------------------
This example is based on Coqui's inference documentation for:
- Python API usage
- XTTS v2 model usage
- speaker_wav voice cloning flow
- language argument for multilingual models
- tts_to_file style inference

In the docs, the basic XTTS usage pattern is conceptually:

    from TTS.api import TTS
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
    tts.tts_to_file(
        text="Hello world!",
        speaker_wav="my/cloning/audio.wav",
        language="en",
        file_path="output.wav"
    )

This script adapts that into a small command-line tool that works with files
like the ones you already use:
- demo_script.txt
- speaker1.wav
- speaker2.wav
- etc.

IMPORTANT DESIGN CHOICES
------------------------
This script intentionally:
- keeps the architecture small
- uses the high-level Python API
- uses file-based input
- accepts multiple speaker WAV paths
- combines text chunks in a very easy-to-understand way
- avoids custom DSP, trimming, retry systems, and advanced post-processing

This makes it good for *learning the bones* of the system.

LIMITATIONS
-----------
This is a teaching script, so it has real limitations:
- chunking is basic and naive
- audio joins are simple concatenation
- no smart pause shaping
- no clipping/risk detection
- no debug package
- no advanced model loading tricks

If the output sounds less polished than your main pipeline, that is expected.

EXAMPLE USAGE
-------------
1) Simplest use with one reference file:

    python xtts_minimal_tutorial.py \
        --input-txt demo_script.txt \
        --speaker-wav speaker1.wav \
        --output-wav demo_output_minimal.wav

2) With multiple reference WAVs:

    python xtts_minimal_tutorial.py \
        --input-txt demo_script.txt \
        --speaker-wav speaker1.wav speaker2.wav \
        --output-wav demo_output_minimal.wav

3) Change language:

    python xtts_minimal_tutorial.py \
        --input-txt demo_script.txt \
        --speaker-wav speaker1.wav \
        --language en \
        --output-wav demo_output_minimal.wav

4) Limit chunk size a bit:

    python xtts_minimal_tutorial.py \
        --input-txt demo_script.txt \
        --speaker-wav speaker1.wav speaker2.wav \
        --max-chars 350 \
        --output-wav demo_output_minimal.wav

NOTES ON ENVIRONMENT
--------------------
You should already have a working environment if your main XTTS setup works.

Typically this script expects:
- Python
- torch
- coqui-tts / TTS package
- soundfile
- numpy

If you are already running XTTS in your current environment, this file is meant
to sit beside your existing project files and reuse the same input assets.
"""

from __future__ import annotations

import argparse
import os
import sys
from contextlib import redirect_stdout, redirect_stderr
import io
import re
import random
import time
from pathlib import Path
from typing import Iterable, List
from scipy.signal import resample_poly

import numpy as np
import soundfile as sf
import torch
from TTS.api import TTS

#In Unix-like operating systems, tee is a command-line utility that reads data from standard input (stdin) and writes it simultaneously to standard output (stdout) and one or more files.
class Tee:
    """
    Write everything to multiple streams at once.

    In this script, we use it so prints still appear in the terminal
    while also being written to a log file.
    """
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()

# -----------------------------------------------------------------------------
# PATCH: Override XTTS audio loader to avoid torchaudio / TorchCodec entirely
# -----------------------------------------------------------------------------

def safe_load_audio(path, sr):
    """
    Replacement for XTTS internal audio loader.

    Uses soundfile instead of torchaudio to avoid TorchCodec dependency.

    Returns a torch tensor shaped [1, num_samples], because XTTS expects
    channel-first audio when building conditioning latents.
    """
    audio, file_sr = sf.read(path, dtype="float32")

    # Convert stereo / multi-channel -> mono
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1, dtype=np.float32)

    audio = np.asarray(audio, dtype=np.float32)

    # Guard against empty or corrupt audio
    if audio.size == 0:
        raise ValueError(f"Reference audio is empty: {path}")

    # Resample only when needed
    if file_sr != sr:
        audio = resample_poly(audio, sr, file_sr).astype(np.float32)

    # XTTS expects [channels, samples]
    audio = np.expand_dims(audio, axis=0)

    return torch.from_numpy(audio)

################################################################################
#---CONTEXT: These 2 blocks collect information on the input.wav and provide diagnostics 
# FOR DEBUGGING: CUT LINE FROM HERE ------------------------------------------>
def analyze_reference_wav(path: str) -> dict:
    """
    Inspect a speaker reference file and return simple diagnostics.

    This helps separate:
    - cloning problems caused by bad reference audio
    from
    - synthesis problems caused by text/chunking/decoding
    """
    audio, sr = sf.read(path, dtype="float32")

    channels = 1 if audio.ndim == 1 else audio.shape[1]

    if audio.ndim > 1:
        mono = np.mean(audio, axis=1, dtype=np.float32)
    else:
        mono = np.asarray(audio, dtype=np.float32)

    mono = np.asarray(mono, dtype=np.float32)

    duration_sec = len(mono) / float(sr) if sr > 0 else 0.0
    peak = float(np.max(np.abs(mono))) if mono.size else 0.0
    rms = float(np.sqrt(np.mean(np.square(mono)))) if mono.size else 0.0

    # Simple silence estimates
    abs_audio = np.abs(mono)
    lead_idx = 0
    trail_idx = len(abs_audio)

    silence_threshold = 0.001

    while lead_idx < len(abs_audio) and abs_audio[lead_idx] < silence_threshold:
        lead_idx += 1

    while trail_idx > 0 and abs_audio[trail_idx - 1] < silence_threshold:
        trail_idx -= 1

    leading_silence_sec = lead_idx / float(sr) if sr > 0 else 0.0
    trailing_silence_sec = (len(abs_audio) - trail_idx) / float(sr) if sr > 0 else 0.0

    clipped_samples = int(np.sum(np.abs(mono) >= 0.999))

    return {
        "path": path,
        "sample_rate": sr,
        "channels": channels,
        "duration_sec": duration_sec,
        "peak": peak,
        "rms": rms,
        "leading_silence_sec": leading_silence_sec,
        "trailing_silence_sec": trailing_silence_sec,
        "clipped_samples": clipped_samples,
    }


def print_reference_report(paths: List[str]) -> None:
    """
    Print speaker reference diagnostics before synthesis starts.
    """
    print("[debug] Speaker reference analysis:")
    for path in paths:
        info = analyze_reference_wav(path)
        print(
            f"  - {info['path']} | "
            f"sr={info['sample_rate']} | "
            f"channels={info['channels']} | "
            f"duration={info['duration_sec']:.2f}s | "
            f"peak={info['peak']:.4f} | "
            f"rms={info['rms']:.4f} | "
            f"lead_silence={info['leading_silence_sec']:.2f}s | "
            f"trail_silence={info['trailing_silence_sec']:.2f}s | "
            f"clipped={info['clipped_samples']}"
        )
# TO HERE <-------------------------------------------------------------------
################################################################################

# Monkey patch XTTS so speaker reference files are loaded through soundfile
# instead of torchaudio/TorchCodec.
import TTS.tts.models.xtts as xtts_module
xtts_module.load_audio = safe_load_audio

# =============================================================================
# SECTION 1 — VERY SMALL TEXT HELPERS
# =============================================================================
#
# In a real narration system, text preparation is a huge part of quality.
# Your production pipeline probably does much more here.
#
# This tutorial version keeps it intentionally light:
# - load text
# - normalize whitespace a bit
# - split into rough chunks so XTTS is not asked to read an enormous blob all at once
#
# We use rough sentence splitting plus a max character budget.
# That is easy to understand, even if it is not as robust as your full system.
# =============================================================================


def read_text_file(path: Path) -> str:
    """
    Load the narration text from disk.

    Why this exists:
    - keeps file I/O separate from model logic
    - makes debugging simpler
    - mirrors your real pipeline where text comes from `demo_script.txt`

    Raises:
        FileNotFoundError: if the file is missing
        ValueError: if the file is empty after stripping whitespace
    """
    if not path.exists():
        raise FileNotFoundError(f"Input text file not found: {path}")

    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Input text file is empty: {path}")

    return text


def normalize_text(text: str) -> str:
    """
    Very small cleanup pass. It simply makes the text a little cleaner:
    - normalizes line breaks
    - collapses repeated whitespace
    - trims edges
    - Converets [PAUSE] and [SHORT PAUSE] into simple tokens that XTTS can handle more easily.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\s+", " ", text)

    #CATCH [PAUSE] and [SHORT PAUSE] and replace with a simple token that XTTS can handle more easily.
    text = text.replace("[PAUSE]", " ")
    text = text.replace("[SHORT PAUSE]", " ")

    return text.strip()


def split_into_sentences(text: str) -> List[str]:
    """
    Naive sentence splitter.

    This is intentionally simple:
    it splits on punctuation like '.', '!', '?'
    while preserving the punctuation with the sentence.

    Important:
    This is only for educational clarity.
    It will not handle every edge case in natural language.

    Example:
        "Hello world. How are you?" ->
        ["Hello world.", "How are you?"]
    """

    # We use a regex split that keeps the punctuation with the sentence. Splits on '.', '!', '?' followed by whitespace.
    parts = re.split(r"(?<=[.!?])\s+", text)

    # Strip whitespace from each item in parts and keep only non-empty results
    sentences = [p.strip() for p in parts if p.strip()]
    
    # Strip PERIOD @ end of sentence ONLY (not all punctuation) - SWAP PERIOD w/ SEMICOLON
    # for i, sentence in enumerate(sentences):
    #     if sentence.endswith("."):
    #         sentences[i] = sentence[:-1] + random.choice(SAFE_ENDINGS)

    return sentences


def chunk_sentences(sentences: Iterable[str], max_chars: int) -> List[str]:
    """
    Pack sentences into rough chunks under a character budget.

    Why do we chunk at all?
    -----------------------
    Feeding a very long documentary script into TTS in one single call is usually
    a bad idea. In practice, you almost always chunk narration.

    What this function does:
    ------------------------
    - walks sentence by sentence
    - groups them into chunks until `max_chars` would be exceeded
    - starts a new chunk when needed

    This gives us a small, understandable approximation of a real TTS chunker.

    Example:
        If max_chars=100, several short sentences may be grouped together,
        while a long sentence may end up alone.
    """

    chunks: List[str] = []          # Initialize the final list of chunks (each chunk will be a string of sentences)
    current: List[str] = []         # Temporary list to hold sentences for the current chunk being built
    current_len = 0                 # Tracks the total character length of the current chunk

    # THE BEST ONES: " ; ", " — ", " : "
    SAFE_ENDINGS = ["; -..."]
    
    # Temporary list to track sentences that may be risky (e.g., very long sentences that approach the max_chars limit)
    risk_tracker = {
        "complexity": None,
        "density": None,
        "fragmentation": None,
    }    
    risk_tracked_flag = False  # This flag will be set to True if we detect any risk factors that suggest the chunk may be too difficult for XTTS to handle well

    # Loop through each sentence from your pre-split sentence list
    for sentence in sentences:

        # Calculate how many characters this sentence would add to the chunk  
        extra = len(sentence) + (1 if current else 0)   # +1 accounts for the space between sentences (only if current is not empty)

        # Before we decide whether to add this sentence to the current chunk or start a new chunk, we run our risk detection logic to see if this sentence might push us into a risky area for synthesis.
        risk_tracker, risk_tracked_flag = chunk_risk_detection(sentence, current, current_len, extra, max_chars, risk_tracked_flag, risk_tracker)

        print(sentence)
        print(risk_tracker)
        print(f"risk_tracked_flag: {risk_tracked_flag}")
        # Check if:
        # 1. we already have content in the current chunk
        # 2. adding this sentence would exceed the max allowed characters
        if (current and (current_len + extra > max_chars)) or (risk_tracked_flag):
            print("--ENTER SANDMAND--") #TO_DELETE: FOR DBUG
            # Finalize the current chunk:
            # - join sentences with spaces
            # - strip any leading/trailing whitespace
            chunk_text = " ".join(current).strip()

            # If there is a period in the chunk, we replace only the last period with an custom punctuation to help XTTS chunk it better. 
            if chunk_text.endswith("."):
                chunk_text = chunk_text[:-1] + random.choice(SAFE_ENDINGS)  
            
            chunks.append(chunk_text)       # Add the finalized chunk to our list of chunks 
            current = [sentence]            # Start a new chunk with the current sentence
            current_len = len(sentence)     # Reset current_len to the length of this new sentence

            # 🔁 RESET RISK STATE FOR NEW CHUNK
            risk_tracker = {
                "complexity": None,
                "density": None,
                "fragmentation": None,
            } 
            risk_tracked_flag = False   # Reset the risk flag for the new chunk since we're starting fresh

        else:
                                        # Otherwise, we can safely add the sentence to the current chunk
            current.append(sentence)    # Add sentence to the current chunk
            current_len += extra        # Increase the current chunk length by the size of the sentence (+ space if applicable)

    # After the loop, there may still be a partially filled chunk left
    if current:
        # Finalize and add the last chunk
        chunk_text = " ".join(current).strip()

        if chunk_text.endswith("."):
            chunk_text = chunk_text[:-1] + random.choice(SAFE_ENDINGS)  

        chunks.append(chunk_text)       # Add the finalized chunk to our list of chunks 
    
    # Return the list of fully built chunks
    return chunks

def chunk_risk_detection(sentence: str, current: List[str], current_len: int, extra: int, max_chars: int, risk_tracked_flag: bool, risk_tracker: dict) -> tuple[dict, bool]:

    # Initializings
    SENTENCE_DIVIDER = ["."]
    KEY_CLAUSES = [":", "and", "but", "because"]  # Define characters that might indicate a dense sentence structure (e.g., lists, multiple clauses)
    COMMA_COUNT = sum(s.count(",") for s in current) + sentence.count(",")
    # Count how many sentence dividers are present in the current chunk plus the new sentence, as a simple proxy for fragmentation risk.
    SENTENCE_DIVIDER_COUNT = (sum(s.count(d) for s in current for d in SENTENCE_DIVIDER) + sum(sentence.count(d) for d in SENTENCE_DIVIDER)) 

    # _________RISK DETECTION FOR COMPLEXITY: LIST-LIKE/PIVOT STRUCTURES _________*STARTS HERE*
    # If the sentence contains any of the key characters that suggest it might be a list or have multiple clauses, we check how many commas it has to assess risk.
    if any(x in sentence for x in KEY_CLAUSES):
        if COMMA_COUNT >= 3:
            risk_tracker["complexity"] = "HIGH"
        elif COMMA_COUNT == 2:
            risk_tracker["complexity"] = "MED"
        elif COMMA_COUNT < 2:
            risk_tracker["complexity"] = "LOW"
    # _________RISK DETECTION FOR COMPLEXITY: LIST-LIKE/PIVOT STRUCTURES _________*ENDS HERE*   
    # 
    # _________RISK DETECTION FOR DENSITY + LENGTH _________*STARTS HERE*  
    if current_len + extra >= int(max_chars * 0.80):
        risk_tracker["density"] = "HIGH"
    elif current_len + extra < int(max_chars * 0.80) and extra >= int(max_chars * 0.67):
        risk_tracker["density"] = "MED"
    elif current_len + extra < int(max_chars * 0.67):
        risk_tracker["density"] = "LOW"
    # _________RISK DETECTION FOR DENSITY + LENGTH _________*ENDS HERE* 
    # 
    # _________RISK DETECTION FOR FRAGMENTATION  _________*STARTS HERE* 
    if SENTENCE_DIVIDER_COUNT  >= 4:
        risk_tracker["fragmentation"] = "HIGH"
    elif SENTENCE_DIVIDER_COUNT == 3:
        risk_tracker["fragmentation"] = "MED"
    elif SENTENCE_DIVIDER_COUNT <= 2:
        risk_tracker["fragmentation"] = "LOW"
    # _________RISK DETECTION FOR FRAGMENTATION  _________*ENDS HERE* 

    HIGH_count = sum(1 for v in risk_tracker.values() if v == "HIGH")
    MED_count = sum(1 for v in risk_tracker.values() if v == "MED")

    # FINAL CHECK: If one (1) HIGH risk or two (2) MED risks are detected across any categories, we flag this chunk as risky for synthesis.
    if HIGH_count >= 1 or MED_count >= 2:
        risk_tracked_flag = True

    return risk_tracker, risk_tracked_flag

# =============================================================================
# SECTION 2 — SPEAKER REFERENCE HANDLING
# =============================================================================
#
# XTTS can clone a voice from a reference WAV via `speaker_wav`.
# The docs show passing a single path. In practice, many users also work with
# multiple reference clips to give the model more voice evidence.
#
# This helper:
# - validates the incoming paths
# - returns them in the form expected by the API call
# =============================================================================


def validate_speaker_wavs(paths: List[str]) -> List[str]:
    """
    Ensure all provided reference WAV files exist.

    We keep the return type as a list of strings because that is a convenient,
    API-friendly form for downstream calls.

    Example accepted input:
        ["speaker1.wav", "speaker2.wav"]
    """
    if not paths:
        raise ValueError("At least one --speaker-wav path is required.")

    validated: List[str] = []
    for raw in paths:
        p = Path(raw)
        if not p.exists():
            raise FileNotFoundError(f"Speaker WAV not found: {p}")
        validated.append(str(p))

    return validated


# =============================================================================
# SECTION 3 — MODEL LOADING
# =============================================================================
#
# This is the heart of the program.
#
# According to the Coqui inference docs, the high-level Python API flow is:
#
#   from TTS.api import TTS
#   tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
#
# We mirror that directly here.
#
# Why this matters:
# - it shows the minimal entry point
# - it hides less than your advanced system
# - it helps you understand what your larger architecture is built on top of
# =============================================================================

def apply_pytorch_xtts_compatibility() -> None:
    """
    PyTorch 2.6+ changed torch.load() so weights_only=True is the default.

    Some XTTS checkpoints used by Coqui TTS are not plain tensor-only loads,
    so they can fail unless loading is relaxed in trusted environments.

    This mini tutorial script assumes you are loading a trusted local XTTS model
    in your own workspace, so we set the same compatibility env var used in the
    main project setup if it is not already present.

    IMPORTANT:
    Only do this for trusted checkpoints.
    """
    if not os.environ.get("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"):
        os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
        print("[info] Set TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 for XTTS compatibility.")


def choose_device() -> str:
    """
    Pick a compute device.

    The docs demonstrate:
        "cuda" if torch.cuda.is_available() else "cpu"

    That is exactly what we do here.

    Why?
    - CUDA is usually much faster if you have a compatible GPU
    - CPU is a fallback so the script still works conceptually everywhere
    """
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_xtts_model(model_name: str, device: str, model_dir: str | None = None) -> TTS:
    """
    Load the XTTS model using the high-level Coqui API.

    Two modes:
    1) Hosted model name from Coqui docs
    2) Local XTTS checkpoint + config files from a model directory

    For local XTTS usage, Coqui needs explicit file paths, not just the folder.
    """
    if model_dir:
        model_dir_path = Path(model_dir)
        checkpoint_path = model_dir_path / "model.pth"
        config_path = model_dir_path / "config.json"

        if not model_dir_path.exists():
            raise FileNotFoundError(f"Model directory not found: {model_dir_path}")
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"XTTS checkpoint not found: {checkpoint_path}\n"
                f"Expected a file named 'model.pth' inside the model directory."
            )
        if not config_path.exists():
            raise FileNotFoundError(
                f"XTTS config not found: {config_path}\n"
                f"Expected a file named 'config.json' inside the model directory."
            )

        print(f"[info] Loading local XTTS model dir: {model_dir_path}")
        print(f"[info] Checkpoint path: {checkpoint_path}")
        print(f"[info] Config path:     {config_path}")
        print(f"[info] Using device:    {device}")

        # IMPORTANT:
        # For this Coqui XTTS loading path, `model_path` should be the directory
        # that contains the checkpoint, not the checkpoint file itself.
        #
        # If we pass ".../model.pth", some internal XTTS loading paths append
        # "model.pth" again, which produces a bad path like:
        #   .../model.pth/model.pth
        #
        # So here we pass:
        # - model_path  = the XTTS model directory
        # - config_path = the explicit config.json path
        tts = TTS(
            model_path=str(model_dir_path),
            config_path=str(config_path),
        ).to(device)
        return tts

    print(f"[info] Loading model: {model_name}")
    print(f"[info] Using device: {device}")
    tts = TTS(model_name).to(device)
    return tts


# =============================================================================
# SECTION 4 — SYNTHESIS
# =============================================================================
#
# This section is where text actually becomes audio.
#
# In the docs, one core XTTS pattern is:
#
#   tts.tts_to_file(
#       text="Hello world!",
#       speaker_wav="my/cloning/audio.wav",
#       language="en",
#       file_path="output.wav"
#   )
#
# Here, instead of writing each chunk directly to a separate file, we use
# `tts.tts(...)` so we can collect audio arrays in memory and then join them.
#
# Why teach it this way?
# ----------------------
# Because this makes the architecture easier to understand:
# - generate per chunk
# - store arrays
# - concatenate
# - write one final WAV
#
# This closely reflects the structure underneath many real-world narration tools.
# =============================================================================


def synthesize_chunks(
    tts: TTS,
    tokenizer,
    chunks: List[str],
    speaker_wavs: List[str],
    language: str,
    speed: float,
    temperature: float,
    repetition_penalty: float,
    top_k: float,
    top_p: float,
    split_sentences: bool,
    inter_chunk_silence_ms: int,
) -> tuple[np.ndarray, int]:
    """
    Generate audio chunk by chunk and combine the results.

    Returns:
        (full_audio, sample_rate)

    Process:
    --------
    1) Call `tts.tts(...)` for each text chunk
    2) Convert returned audio to numpy
    3) Insert a little silence between chunks
    4) Concatenate everything into one waveform

    Why insert silence?
    -------------------
    If we concatenate chunks back-to-back with no gap at all, narration may sound
    unnaturally abrupt at chunk boundaries. This small pause is the simplest,
    most understandable boundary treatment.

    This is still very primitive compared to your main pipeline.
    """
    all_segments: List[np.ndarray] = []
    chunk_metrics = []
    
    run_start_time = time.perf_counter() # Start a run timer that resets per chunk

    #Initialize variables for tts_params
    v_speed = speed
    v_temperature = temperature
    v_repetition_penalty = repetition_penalty
    v_top_k = top_k
    v_top_p = top_p

    # Try to discover the output sample rate from the synthesizer when possible.
    # If unavailable, we fall back to a common XTTS rate.
    sample_rate = getattr(getattr(tts, "synthesizer", None), "output_sample_rate", 24000)

    silence = np.zeros(int(sample_rate * inter_chunk_silence_ms / 1000.0), dtype=np.float32)

    print(f"[debug] inter-chunk silence: {inter_chunk_silence_ms} ms @ {sample_rate} Hz → {len(silence)} samples ({len(silence)/sample_rate:.3f}s)")
    print(f"[debug] split-sentences: {split_sentences}")

    for i, chunk in enumerate(chunks, start=1):
        chunk_start_time = time.perf_counter() # Start a timer per chunk

        chars = len(chunk)                  # Count characters in the chunk for metrics and debugging.
        indexed_chunk = i                   # Track index of the chunk for progress reporting. We start at 1 for human-friendly counting.
        total_chunks = len(chunks)          # Total number of chunks, used for progress reporting.
        attempt_on_chunk = 0                # Initialize an attempt counter for this chunk.
        attempt_valid = True                # Flag to track if the current attempt is valid.
        lowest_penalty_score = -1.0         # This variable will track the best (lowest) penalty score observed across attempts for this chunk.
        wav_cached = None                   # This variable will hold the best audio waveform generated for this chunk across all attempts.

        #TOKEN COUNT INIT
        tokens = tokenizer.encode(chunk, lang= language)
        token_count = len(tokens)

        # PRINT HEADER FOR THIS CHUNK
        print("================================================================================================")
        print(f"----------------------------------------------***------------------------------------------------ - [Chunk {indexed_chunk}/{total_chunks}]")
        print("================================================================================================")
        print(f"[info] Synthesizing chunk {indexed_chunk}/{total_chunks} | chars={chars} | tokens={token_count}")
        
        while attempt_valid:
            # RETRY SUB HEADER FOR  CHUNK
            print(f"[info] >>> ... RETRY ATTEMPT [{attempt_on_chunk}] for chunk {indexed_chunk}/{total_chunks} ..........................<<<")

            #TOKEN COUNT DEBUGGING
            tokens = tokenizer.encode(chunk, lang= language)
            token_count = len(tokens)

            stdout_buffer = io.StringIO()   #Capture Coqui internal prints - give each chunk a fresh buffer inside the loop
            with redirect_stdout(stdout_buffer), redirect_stderr(stdout_buffer):         # 'The buffer': For capturing the internal prints from Coqui TTS, which can be helpful for debugging synthesis issues.
                wav = tts.tts(                                                           # The `tts.tts(...)` method generates audio for a given text chunk and returns it as a waveform array. 
                    text=chunk,
                    speaker_wav=speaker_wavs,
                    language=language,
                    speed=v_speed,
                    temperature=v_temperature,
                    repetition_penalty=v_repetition_penalty,
                    top_k=v_top_k,
                    top_p=v_top_p,
                    split_sentences=split_sentences,
                )
            
            # __________METRICS TRACKING STARTS HERE__________
            # Retrieve the captured output from the buffer
            captured_output = stdout_buffer.getvalue()
            if captured_output:     # If there is any captured output, we print it to the console.
                print(captured_output, end="")
            
            # Extract and print synthesis metrics for this chunk, and store them in our chunk_metrics list.
            chunk_metrics = chunks_metrics_report(captured_output, chunk_metrics, chars, indexed_chunk, total_chunks, attempt_on_chunk,
                v_speed=v_speed,
                v_temperature=v_temperature,
                v_repetition_penalty=v_repetition_penalty,
                v_top_k=v_top_k,
                v_top_p=v_top_p,
            ) 

            #Assign Flags + Values based on the latest metrics for this chunk
            processing_flag = chunk_metrics[-1]['processing_flag']
            audio_flag = chunk_metrics[-1]['audio_flag']
            penalty_score = chunk_metrics[-1]['penalty_score']

            if attempt_on_chunk == 0:
                lowest_penalty_score = penalty_score
                wav_cached = wav  # Cache the first attempt's audio as a baseline for this chunk. We will compare future attempts against this to see if we are improving or not.
            else:
                if penalty_score < lowest_penalty_score:
                    lowest_penalty_score = penalty_score    # Update the lowest penalty score if the current attempt is better than all previous attempts for this chunk.
                    wav_cached = wav                        # Update the cached waveform to the current attempt's audio if it.

            # Check if the processing_sec_per_char or audio_sec_per_char metrics indicate a potential issue with this chunk's synthesis. If either metric exceeds a certain threshold and this is the first two attempts on this chunk, we mark the attempt as valid and increment the attempt counter to allow for a retry. 
            if (processing_flag or audio_flag) and (attempt_on_chunk < 10):    # Up to 10 attempts per chunk: 

                #Qualifiers
                attempt_valid = True
                attempt_on_chunk += 1

                # UPDATE TTS PARAMS FOR NEXT ATTEMPT BASED ON METRICS
                v_speed = chunk_metrics[-1]['speed']
                v_temperature = chunk_metrics[-1]['temperature']
                v_repetition_penalty = chunk_metrics[-1]['repetition_penalty']
                v_top_k = chunk_metrics[-1]['top_k']
                v_top_p = chunk_metrics[-1]['top_p']
                 
                # -*- ATTEMPT #: DISABLED
                if attempt_on_chunk == 100:  # Temporaroly closed for testing - set to a high number to prevent triggering while we test other parts of the loop.

                    # _______________________ATTEMPT PUNCTUATION INJECTION _______________________ *STARTS HERE*
                    target_ratio = 0.39  # minimum proportion (40%) a segment must have relative to full chunk

                    segments = [(m.start(), m.end()) for m in re.finditer(r"[^.,]+", chunk)]             # find all substrings between periods/commas and store their start/end indices
                    seg_start, seg_end = max(segments, key=lambda s: len(chunk[s[0]:s[1]].strip()))     # select the longest segment (ignoring surrounding whitespace)

                    if len(chunk[seg_start:seg_end].strip()) >= len(chunk) * target_ratio:              # ensure the selected segment is large enough compared to the full chunk
                        print("[attention] Punctuation injection is <ENABLED> for this chunk based on synthesis metrics and segment size.")
                        
                        seg_mid = (seg_start + seg_end) // 2                                            # compute the midpoint index of the selected segment
                        words = list(re.finditer(r"\b\w{4,}\b", chunk[seg_start:seg_end]))              # find all words (length >= 4) within the segment

                        if words:  # ensure there are valid candidate words
                            
                            target = min(words, key=lambda m: abs((seg_start + m.start() + seg_start + m.end()) // 2 - seg_mid))    # choose the word whose center is closest to the segment midpoint
                            insert_at = seg_start + target.end()                        # calculate absolute index (in full chunk) right after the chosen word
                            chunk = chunk[:insert_at] + ";" + chunk[insert_at:]         # insert a punctuation immediately after the selected word

                    else:
                        #print(f"[attention] Punctuation injection is <DISABLED> for this chunk because the longest segment is too small relative to the full chunk.")
                        # attempt_valid = False  # If the longest segment is too small, we mark the attempt as invalid to exit the while loop.
                        # break 
                        continue  # Skip the rest of the loop and retry synthesis with the original chunk text.        
                    # _______________________ATTEMPT PUNCTUATION INJECTION _______________________ *ENDS HERE*

                
                continue  # Skip the rest of the loop and retry synthesis with the modified chunk text.

            # On the third attempt, we allow a slightly higher threshold to give the model one last chance to produce acceptable output before we stop retrying this chunk. 
            # elif (processing_sec_per_char >= 0.0275 or audio_sec_per_char >= 0.073) and ("," not in chunk and ":" not in chunk) and (attempt_on_chunk < 5):
            #     #Qualifiers
            #     attempt_valid = True
            #     attempt_on_chunk += 1

            else:
                attempt_valid = False  # If the metrics are within acceptable limits, we mark the attempt as invalid to exit the while loop.   
                
                # RESET TTS PARAMS FOR NEXT CHUNK
                v_speed = speed
                v_temperature = temperature
                v_repetition_penalty = repetition_penalty
                v_top_k = top_k
                v_top_p = top_p

                break  
            
            # __________METRICS TRACKING ENDS HERE__________
            

        # //////////////////////////////////////////////////////
        # SUB-SECTION: TIME TRACKING: CHUNK COMPLETE SUMMARY
        # \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
        #*****************************************
        chunk_total_elapsed = time.perf_counter() - chunk_start_time
        run_elapsed = time.perf_counter() - run_start_time

        completed_chunks = indexed_chunk
        avg_chunk_time = run_elapsed / completed_chunks
        remaining_chunks = total_chunks - indexed_chunk
        eta_seconds = avg_chunk_time * remaining_chunks

        print(
            f"[time] chunk_complete={indexed_chunk}/{total_chunks} | "
            f"chunk_time={format_seconds(chunk_total_elapsed)} | "
            f"avg_chunk_time={format_seconds(avg_chunk_time)} | "
            f"run_elapsed={format_seconds(run_elapsed)} | "
            f"eta={format_seconds(eta_seconds)}"
        )
        #*****************************************

        print("[debug] My Notes: ")

        if wav_cached is not None:
            print(f"[debug] - Lowest penalty score for this chunk across attempts: {lowest_penalty_score:.4f}")
            wav = wav_cached  # Use the best audio we cached across attempts for this chunk.

        wav_np = np.asarray(wav, dtype=np.float32)      # Coqui returns waveform amplitude values. We convert to float32 numpy.
        all_segments.append(wav_np)                     # Store the generated audio for this chunk in our list of segments.

        
        if i < total_chunks + 1:                         # Add a small pause after each chunk (including last one).
            all_segments.append(silence)

    if not all_segments:
        raise RuntimeError("No audio segments were generated.")

    full_audio = np.concatenate(all_segments)
    return full_audio, sample_rate

def chunks_metrics_report(captured_output: str, chunk_metrics: List[dict], chars: int, chunk_index: int, chunk_count: int, attempt_on_chunk: int, 
                          v_speed: float, v_temperature: float, v_repetition_penalty: float, v_top_k: float, v_top_p: float) -> List[dict]:
    """
    Extract and print synthesis metrics from Coqui's internal output for a single chunk.
    """
    PROCESSING_THRESHOLD_MAX = 0.0265  # seconds per character
    PROCESSING_THRESHOLD_MIN = 0.0220  # seconds per character
    AUDIO_THRESHOLD_MAX = 0.071       # seconds per character
    AUDIO_THRESHOLD_MIN = 0.062       # seconds per character
    PENALTY_THRESHOLD = 0.003   

    BAND_THRESHOLD = 0.003  #To control the if-gates for param_adjustment: 

    # TTS PARAMETER BOUNDS AND ADJUSTMENT STEPS
    TEMPERATURE_MIN = 0.50
    TEMPERATURE_MAX = 0.80
    TOP_P_MIN = 0.60
    TOP_P_MAX = 0.85
    REPETITION_PENALTY_MIN = 2.0
    REPETITION_PENALTY_MAX = 2.6
    SPEED_MIN = 0.96
    SPEED_MAX = 1.04    

    # Minimum adjustment steps for TTS parameters when synthesis metrics indicate potential issues. These values determine how much we tweak the parameters on each retry attempt.
    MIN_TEMPERATURE_ADJUSTMENT = 0.01
    MIN_TOP_P_ADJUSTMENT = 0.01
    MIN_REPETITION_PENALTY_ADJUSTMENT = 0.1
    MIN_SPEED_ADJUSTMENT = 0.01


    # Extract processing time, audio_duration, and real-time factor from the captured output using regular expressions.
    processing_time = round(float(m.group(1)), 6) if (m := re.search(r"Processing time:\s*([0-9.]+)", captured_output)) else None
    real_time_factor = round(float(m.group(1)), 6) if (m := re.search(r"Real-time factor:\s*([0-9.]+)", captured_output)) else None
    audio_duration = round(processing_time / real_time_factor, 2)
    # Calculate processing seconds per character and audio seconds per character, rounding to 5 decimal places for readability. 
    processing_sec_per_char = round(processing_time / chars, 5) if chars > 0 else None
    audio_sec_per_char = round(audio_duration / chars, 5) if chars > 0 else None

    # -------------------------------------------------------------------------
    # SECTION: PENALTY CALCULATIONS
    # Normalized penalties (dimensionless)
    # These are best for comparing processing vs audio on a common scale and for
    # building a combined penalty score.
    # -------------------------------------------------------------------------
    processing_penalty_over = max(0.0, (processing_sec_per_char / PROCESSING_THRESHOLD_MAX) - 1.0)   
    processing_penalty_under = max(0.0, 1.0 - (processing_sec_per_char / PROCESSING_THRESHOLD_MIN))
    audio_penalty_over = max(0.0, (audio_sec_per_char / AUDIO_THRESHOLD_MAX) - 1.0)
    audio_penalty_under = max(0.0, 1.0 - (audio_sec_per_char / AUDIO_THRESHOLD_MIN))

    # We take the maximum of over and under penalties to capture deviation in either direction.
    processing_penalty = max(processing_penalty_over, processing_penalty_under) 
    audio_penalty = max(audio_penalty_over, audio_penalty_under) 
    
    # Combine both penalties into a single score: Weight audio slightly higher (0.55) since it's more audible to users
    penalty_score = (processing_penalty ** 2) * 0.45 + (audio_penalty ** 2) * 0.55      #Lower score = better synthesis attempt

    # Penalty score must be greater than PENALTY THRESHOLD (0.0030) for us to trigger adjustments, else the result is close to the sweet spot, so try again.
    low_penalty = True if (penalty_score < PENALTY_THRESHOLD) else False   

    # ************************************************************************************************************************************************************************************
    # SECTION: Flag Processing and Audio based on thresholds for seconds per character. These flags will determine if we need to adjust TTS parameters and retry synthesis for this chunk.
    # ************************************************************************************************************************************************************************************
    processing_kickback_factor = 0.0
    audio_kickback_factor = 0.0

    # We set flags to indicate whether the processing time per character or audio duration per character are outside of acceptable thresholds.
    processing_flag = True if (processing_sec_per_char > PROCESSING_THRESHOLD_MAX or processing_sec_per_char < PROCESSING_THRESHOLD_MIN) else False     # > 0.0265 or < 0.0215
    audio_flag = True if (audio_sec_per_char > AUDIO_THRESHOLD_MAX or audio_sec_per_char < AUDIO_THRESHOLD_MIN) else False       # >= 0.070 or < 0.060
    under_flag = True if (processing_sec_per_char < PROCESSING_THRESHOLD_MIN or audio_sec_per_char < AUDIO_THRESHOLD_MIN) else False   # This flag indicates if we are under the minimum thresholds.

    # -------------------------------------------------------------------------
    # Distance from acceptable band (native units: seconds per character)
    # These are best for controller decisions because they preserve the real
    # amount by which we are over/under the target band.
    # -------------------------------------------------------------------------
    processing_over = max(0.0, processing_sec_per_char - PROCESSING_THRESHOLD_MAX)
    processing_under = max(0.0, PROCESSING_THRESHOLD_MIN - processing_sec_per_char) #UN-USED
    audio_over = max(0.0, audio_sec_per_char - AUDIO_THRESHOLD_MAX)
    audio_under = max(0.0, AUDIO_THRESHOLD_MIN - audio_sec_per_char)

    # Near-target "settle mode": When the penalty is already low, do NOT keep retuning the sampler aggressively. Freeze sampling knobs unless duration clearly still needs a tiny correction.
    settle_mode = penalty_score < (0.5 * PENALTY_THRESHOLD)

    # -------------------------------------------------------------------------
    # PROCESSING CONTROL
    # Important: processing_sec_per_char is only indirectly affected by sampling
    # knobs. So treat this branch conservatively.
    # -------------------------------------------------------------------------
    if processing_flag:

        # SETTLE MODE: Near target, do not keep retuning multiple sampling knobs. Leave processing mostly alone and let retry sampling / best-attempt selection do the work.
        if settle_mode:
            processing_kickback_factor = 0.10

        # UNDER-BAND PROCESSING: Do NOT use speed as the first fix here; speed is primarily a duration knob.
        # If processing is low, only widen the sampler a touch so the retry is not overly rigid.
        elif processing_sec_per_char < PROCESSING_THRESHOLD_MIN:
            v_temperature = min(TEMPERATURE_MAX, v_temperature + (0.25 * MIN_TEMPERATURE_ADJUSTMENT))   # +0.0025
            v_top_p = min(TOP_P_MAX, v_top_p + (0.25 * MIN_TOP_P_ADJUSTMENT))                           # +0.0025
            processing_kickback_factor = 0.08

        # TIER 1: Slightly above processing band
        elif processing_over <= BAND_THRESHOLD:  #0.003
            print("[debug]: LIGHT OF DAY (processing)")
            v_temperature = max(TEMPERATURE_MIN, v_temperature - (0.5 * MIN_TEMPERATURE_ADJUSTMENT))    # -0.005
            v_top_p = max(TOP_P_MIN, v_top_p - (0.25 * MIN_TOP_P_ADJUSTMENT))                           # -0.0025
            processing_kickback_factor = 0.08

        # TIER 2: Moderately above processing band
        elif processing_over <= BAND_THRESHOLD + 0.0045:  #0.0075
            v_temperature = max(TEMPERATURE_MIN, v_temperature - MIN_TEMPERATURE_ADJUSTMENT)            # -0.01
            v_top_p = max(TOP_P_MIN, v_top_p - (0.5 * MIN_TOP_P_ADJUSTMENT))                            # -0.005
            v_repetition_penalty = min(REPETITION_PENALTY_MAX, v_repetition_penalty + (0.5 * MIN_REPETITION_PENALTY_ADJUSTMENT))  # +0.05
            processing_kickback_factor = 0.12

        # TIER 3: Clearly bad processing / likely unstable token path
        else:
            v_temperature = max(TEMPERATURE_MIN, v_temperature - MIN_TEMPERATURE_ADJUSTMENT)            # -0.01
            v_top_p = max(TOP_P_MIN, v_top_p - MIN_TOP_P_ADJUSTMENT)                                    # -0.01
            v_repetition_penalty = min(REPETITION_PENALTY_MAX, v_repetition_penalty + MIN_REPETITION_PENALTY_ADJUSTMENT)          # +0.1
            processing_kickback_factor = 0.18
        
    
    # -------------------------------------------------------------------------
    # AUDIO / DURATION CONTROL
    # speed is the most justified primary control for audio_sec_per_char because
    # XTTS applies it after GPT sampling as latent time scaling.
    # -------------------------------------------------------------------------
    if audio_flag:

        # SETTLE MODE: only tiny duration corrections, freeze the sampler if possible
        if settle_mode:
            if audio_sec_per_char < AUDIO_THRESHOLD_MIN:        # Under audio threshold
                v_speed = max(SPEED_MIN, v_speed - (0.2 * MIN_SPEED_ADJUSTMENT))     # -0.002
                audio_kickback_factor = 0.08
            elif audio_sec_per_char > AUDIO_THRESHOLD_MAX:      # Above audio threshold
                v_speed = min(SPEED_MAX, v_speed + (0.2 * MIN_SPEED_ADJUSTMENT))     # +0.002
                audio_kickback_factor = 0.08

        # UNDER-BAND AUDIO: duration too short
        elif audio_sec_per_char < AUDIO_THRESHOLD_MIN:
            v_speed = max(SPEED_MIN, v_speed - MIN_SPEED_ADJUSTMENT)                  # -0.01
            # Slight widening only if we're clearly short; avoid stacking many changes
            if audio_under >= (2 * BAND_THRESHOLD):      # Audio is under more than DOUBLE the BAND_THRESHOLD = 0.006
                v_temperature = min(TEMPERATURE_MAX, v_temperature + (0.25 * MIN_TEMPERATURE_ADJUSTMENT))  # +0.0025
            audio_kickback_factor = 0.15

        # TIER 1: Slightly above audio band
        elif audio_over <= (2 * BAND_THRESHOLD):         # Audio is over less than DOUBLE the BAND_THRESHOLD = 0.006
            print("[debug]: LIGHT OF DAY (audio)")
            v_speed = min(SPEED_MAX, v_speed + MIN_SPEED_ADJUSTMENT)                  # +0.01
            audio_kickback_factor = 0.15

        # TIER 2: Clearly above audio band
        else:
            v_speed = min(SPEED_MAX, v_speed + (2 * MIN_SPEED_ADJUSTMENT))            # +0.02
            # Slightly tighten the sampler only when duration is clearly too long
            v_top_p = max(TOP_P_MIN, v_top_p - (0.25 * MIN_TOP_P_ADJUSTMENT))         # -0.0025
            audio_kickback_factor = 0.22

        

    # -------------------------------------------------------------------------
    # DELTA CALCULATIONS
    # Use deltas as a gentle correction layer, not a hard "undo" signal.
    # Single-retry worsening may just be stochastic variance.
    # -------------------------------------------------------------------------
    delta_processing = None 
    delta_audio = None
    processing_delta_flag = processing_flag     # If gatekeeper flag is True, we consider the processing metric for delta adjustments. We will calculate the change compared to the last attempt for this chunk.
    audio_delta_flag = audio_flag               # If gatekeeper flag is True, we consider the processing metric for delta adjustments. We will calculate the change compared to the last attempt for this chunk.  
    
    # We want to calculate deltas only on retry attempts, comparing the current metrics to the last attempt for this chunk.
    if attempt_on_chunk > 0:
        delta_processing = processing_sec_per_char - chunk_metrics[-1]['processing_sec_per_char']   # We calculate the change in processing seconds per character compared to the last attempt for this chunk.
        delta_audio = audio_sec_per_char - chunk_metrics[-1]['audio_sec_per_char']                  # We calculate the change in audio seconds per character compared to the last attempt for this chunk.

        # If both metrics have worsened compared to the last attempt, we want to identify which one is the dominant failure to inform our next adjustments.
        if (delta_processing > 0 and delta_audio > 0) and (processing_delta_flag and audio_delta_flag): 

            # Determine dominant failure (normalize against thresholds) 
            processing_severity = processing_penalty
            audio_severity = audio_penalty

            dominant = "processing" if processing_severity >= audio_severity else "audio"
            print(f"[debug] DOUBLE KICKBACK DOMINANT: {dominant} | processing_severity={processing_severity:.4f}, audio_severity={audio_severity:.4f}")

            # Reset flags so we can apply a delta kickback adjustment to the dominant metric on the next attempt.
            processing_delta_flag = True if dominant == "processing" else False
            audio_delta_flag = True if dominant == "audio" else False

        # PROCESSING DELTA KICKBACK:
        # Only make small backoff moves. Do not fully reverse the previous intent.
        if delta_processing > 0.001 and processing_delta_flag and not(settle_mode):
            print(f"[debug] DELTA KICKBACK: For processing_sec_per_char, the metric has increased by {delta_processing}.")

            if not(under_flag):
                v_temperature = min(TEMPERATURE_MAX, v_temperature + (processing_kickback_factor * MIN_TEMPERATURE_ADJUSTMENT))
                v_top_p = min(TOP_P_MAX, v_top_p + (processing_kickback_factor * MIN_TOP_P_ADJUSTMENT))
                v_repetition_penalty = max(REPETITION_PENALTY_MIN, v_repetition_penalty - (processing_kickback_factor * MIN_REPETITION_PENALTY_ADJUSTMENT))
            else:
                v_temperature = max(TEMPERATURE_MIN, v_temperature - (processing_kickback_factor * MIN_TEMPERATURE_ADJUSTMENT))
                v_top_p = max(TOP_P_MIN, v_top_p - (processing_kickback_factor * MIN_TOP_P_ADJUSTMENT))

        # AUDIO DELTA KICKBACK:
        # speed is still the main backoff lever for duration drift.
        if delta_audio > 0.0015 and audio_delta_flag:
            print(f"[debug] DELTA KICKBACK: For audio_sec_per_char, the metric has increased by {delta_audio}.")

            if not under_flag:
                v_speed = max(SPEED_MIN, v_speed - (audio_kickback_factor * MIN_SPEED_ADJUSTMENT))
                if not(settle_mode):
                    v_top_p = min(TOP_P_MAX, v_top_p + (audio_kickback_factor * MIN_TOP_P_ADJUSTMENT))
            else:
                v_speed = min(SPEED_MAX, v_speed + (audio_kickback_factor * MIN_SPEED_ADJUSTMENT))


    # Extract a preview of the chunk text as processed by Coqui. 
    chunk_preview_from_coqui = ( m.group(1).strip().replace("\\n", " ").replace("\n", " ")
        if (m := re.search(r"(\[\s*'.*?'\s*\])\s*> Processing time:", captured_output, re.DOTALL)) else None
    )

    # We collect some simple metrics about each chunk, which can be useful for debugging and understanding performance. 
    chunk_metrics.append({
        "chunk_index": f"{chunk_index}/{chunk_count}",
        "chars": chars,
        "text": chunk_preview_from_coqui,
        "processing_time": processing_time,
        "audio_duration": audio_duration,
        "rtf": real_time_factor,
        "processing_sec_per_char": processing_sec_per_char,
        "audio_sec_per_char": audio_sec_per_char,
        "processing_flag": processing_flag,
        "audio_flag": audio_flag,
        "delta_processing": delta_processing,
        "delta_audio": delta_audio,
        "speed": v_speed,
        "temperature": v_temperature,
        "repetition_penalty": v_repetition_penalty,
        "top_k": v_top_k,
        "top_p": v_top_p,
        "penalty_score": penalty_score,
        "settle_mode": settle_mode,
    })

    # Print the metrics for this chunk in a clear format. [-1] accesses the most recently added chunk's metrics.
    print(f"[info] text={chunk_metrics[-1]['text']}")
    print(f"[info] processing_time={chunk_metrics[-1]['processing_time']} | audio_duration={chunk_metrics[-1]['audio_duration']} | rtf={chunk_metrics[-1]['rtf']} | settle_mode={chunk_metrics[-1]['settle_mode']}")
    print(f"[info] deltas= Δ processing_sec_per_char={chunk_metrics[-1]['delta_processing']} | Δ audio_sec_per_char={chunk_metrics[-1]['delta_audio']}")
    print(f"[info] ________Scores_________")
    print(f"[info] processing_sec_per_char score={round(100*chunk_metrics[-1]['processing_sec_per_char'], 2)} ({processing_sec_per_char})")
    print(f"[info] audio_sec_per_char score={round(100*chunk_metrics[-1]['audio_sec_per_char'], 2)} ({audio_sec_per_char})")
    print(f"[info] TTS parameters for next attempt: speed={chunk_metrics[-1]['speed']:.3f}, temperature={chunk_metrics[-1]['temperature']:.3f}, repetition_penalty={chunk_metrics[-1]['repetition_penalty']:.3f}, top_p={chunk_metrics[-1]['top_p']:.3f} | penalty_score={round(100*chunk_metrics[-1]['penalty_score'], 2)} ({chunk_metrics[-1]['penalty_score']})")
    print(f"[info] _______________________")
    
    return chunk_metrics

# =============================================================================
# SECTION 5 — WRITING THE FINAL WAV
# =============================================================================
#
# Once you have a waveform array, writing it to disk is straightforward.
# This is one of the cleanest parts of the architecture:
#
#   audio array -> WAV file
#
# `soundfile.write` is used here because it is simple and reliable.
# =============================================================================


def write_wav_file(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    """
    Save the final narration waveform to disk as a WAV file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, sample_rate)
    print(f"[done] Wrote WAV to: {path}")


# =============================================================================
# SECTION 6 — COMMAND-LINE INTERFACE
# =============================================================================
#
# This section makes the script usable from the terminal.
# It mirrors the style of your existing workflow:
#
#   --input-txt demo_script.txt
#   --speaker-wav speaker1.wav speaker2.wav
#   --language en
#   --output-wav output.wav
#
# A good teaching script should be easy to run and easy to inspect.
# =============================================================================

def format_seconds(seconds: float) -> str:
    """
    Format elapsed seconds into a compact human-readable string.
    Examples:
        12.4   -> 12.4s
        75.2   -> 1m 15.2s
        3671.9 -> 1h 1m 11.9s
    """
    seconds = max(0.0, float(seconds))

    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60

    if hours > 0:
        return f"{hours}h {minutes}m {secs:.1f}s"
    if minutes > 0:
        return f"{minutes}m {secs:.1f}s"
    return f"{secs:.1f}s"

def build_arg_parser() -> argparse.ArgumentParser:
    """
    Build the CLI argument parser.

    We keep the interface intentionally small and familiar.
    """
    parser = argparse.ArgumentParser(
        description="Minimal educational XTTS v2 narration script using Coqui TTS."
    )
    parser.add_argument(
        "--input-txt",
        required=True,
        help="Path to the narration text file, e.g. demo_script.txt",
    )
    parser.add_argument(
        "--speaker-wav",
        nargs="+",
        required=True,
        help="One or more reference speaker WAV files, e.g. speaker1.wav speaker2.wav",
    )
    parser.add_argument(
        "--output-wav",
        default="demo_output_minimal.wav",
        help="Path for the final generated WAV file",
    )
    # ------ Inference Paramters ------
    parser.add_argument(
        "--language",
        default="en",
        help="Language code for XTTS, e.g. en",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="The speed rate of the generated audio. Defaults to 1.0. (can produce artifacts if far from 1.0)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.65,
        help="The softmax temperature of the autoregressive model. Defaults to 0.75",
    )
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=2.0,
        help="A penalty that prevents the autoregressive decoder from repeating itself during decoding. Can be used to reduce the incidence of long silences or “uhhhhhhs”, etc. Defaults to 5.0.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=50,
        help="Lower values mean the decoder produces more “likely” (aka boring) outputs. Defaults to 50.",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.80,
        help="Lower values mean the decoder produces more “likely” (aka boring) outputs. Defaults to 0.85",
    )
    parser.add_argument(
        "--split_sentences",
        action="store_true",
        help="Enable XTTS internal text splitting",
    )
    # --------------------------------
    parser.add_argument(
        "--model-name",
        default="tts_models/multilingual/multi-dataset/xtts_v2",
        help="Coqui model name to load. Defaults to XTTS v2 from the docs.",
    )
    parser.add_argument(
        "--model-dir",
        default=None,
        help="Optional local XTTS model directory containing config.json and model.pth, e.g. /workspace/xtts_job/xtts_v2_model",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=350,
        help="Approximate max characters per chunk",
    )
    parser.add_argument(
        "--inter-chunk-silence-ms",
        type=int,
        default=250,
        help="Simple silence inserted between chunks",
    )
    return parser


# =============================================================================
# SECTION 7 — MAIN PROGRAM FLOW
# =============================================================================
#
# This is the entire architecture in sequence:
#
#   parse args
#   -> read text
#   -> normalize text
#   -> split into sentences
#   -> group into chunks
#   -> validate speaker WAVs
#   -> choose device
#   -> load XTTS model
#   -> synthesize chunk by chunk
#   -> write final WAV
#
# That is the foundational skeleton of a file-based XTTS narration tool.
# =============================================================================


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    main_start_time = time.perf_counter()   #Full program timer

    # Set up a log file to capture all terminal output for later inspection. This is especially useful for reviewing the synthesis metrics and debugging any issues with specific chunks after the run is complete.
    log_file = open("terminal_log.txt", "w", encoding="utf-8")
    tee = Tee(sys.stdout, log_file)

    with redirect_stdout(tee), redirect_stderr(tee):    # Redirect all prints to both the terminal and the log file for comprehensive record-keeping.

        input_txt = Path(args.input_txt)
        output_wav = Path(args.output_wav)

        # -------------------------------------------------------------------------
        # Step 1: Load and clean the script
        # -------------------------------------------------------------------------
        raw_text = read_text_file(input_txt)
        clean_text = normalize_text(raw_text)

        # -------------------------------------------------------------------------
        # Step 2: Break the script into manageable pieces
        # -------------------------------------------------------------------------
        sentences = split_into_sentences(clean_text)
        chunks = chunk_sentences(sentences, max_chars=args.max_chars)

        # PRINT : Initialization information so we can verify the text processing steps before synthesis.
        print("Initialization information:")
        print(f"[info] Loaded text from: {input_txt}")
        print(f"[info] Sentence count: {len(sentences)}")
        print(f"[info] Chunk count:    {len(chunks)}")
        print(f"[info] Max chars:      {args.max_chars}")

        # -------------------------------------------------------------------------
        # DEBUGGING 
        # -------------------------------------------------------------------------
        print("[debug] Cleaned/Normalized Text:  ")
        print(clean_text)
        print("[debug] Sentences Split:  ")
        print(sentences)
        print("[debug] Sentences Chunked:  ")
        print(chunks)
        # -------------------------------------------------------------------------

        # -------------------------------------------------------------------------
        # Step 3: Validate the speaker reference audio
        # -------------------------------------------------------------------------
        speaker_wavs = validate_speaker_wavs(args.speaker_wav)
        print(f"[info] Speaker refs:   {speaker_wavs}")
        print_reference_report(speaker_wavs) # TO_DELETE: This is a debugging helper that prints diagnostics about the speaker reference files. 

        # -------------------------------------------------------------------------
        # Step 4: Load model
        # -------------------------------------------------------------------------
        apply_pytorch_xtts_compatibility()
        device = choose_device()
        tts = load_xtts_model(args.model_name, device, args.model_dir)

        # We will use the model's tokenizer to count tokens for debugging and potential future use in more advanced chunking strategies.
        tokenizer = tts.synthesizer.tts_model.tokenizer 

        # -------------------------------------------------------------------------
        # Step 5: Generate narration
        # -------------------------------------------------------------------------
        audio, sample_rate = synthesize_chunks(
            tts=tts,
            tokenizer=tokenizer,
            chunks=chunks,
            speaker_wavs=speaker_wavs,
            language=args.language,
            speed=args.speed,
            temperature=args.temperature,
            repetition_penalty=args.repetition_penalty,
            top_k=args.top_k,
            top_p=args.top_p,
            split_sentences=args.split_sentences,
            inter_chunk_silence_ms=args.inter_chunk_silence_ms,
        )

        # -------------------------------------------------------------------------
        # Step 6: Write output
        # -------------------------------------------------------------------------
        write_wav_file(output_wav, audio, sample_rate)


        # Total wall-clock runtime for the full script
        main_elapsed = time.perf_counter() - main_start_time
        print(f"[summary] Total elapsed runtime: {format_seconds(main_elapsed)}")

        print("[summary] Minimal XTTS narration run complete.")
        print("[summary] This script shows the core architecture only.")
        print("[summary] Your production pipeline adds the quality-control layers on top.")


if __name__ == "__main__":
    main()
