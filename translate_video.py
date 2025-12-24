"""
Video Translation Pipeline: English to Spanish
Tony Robbins Goal Setting Workshop - Day 12

Pipeline: Video -> Extract Audio -> Transcribe (Whisper) -> Translate (DeepL) -> TTS (OpenAI) -> Composite
Cost estimate: ~$1.77 for 45-minute video
"""

import os
import json
import subprocess
from pathlib import Path
from dotenv import load_dotenv
import requests

# Load environment variables
ENV_PATH = Path.home() / '.env'
load_dotenv(ENV_PATH, override=True)

# Paths
PROJECT_DIR = Path(__file__).parent
VIDEO_DIR = PROJECT_DIR / "video"
SOURCE_VIDEO = VIDEO_DIR / "tony-robbins-day12.mp4"
FFMPEG = r"C:\Users\MarieLexisDad\CAAASA Video\New Video\ffmpeg-8.0-essentials_build\bin\ffmpeg.exe"

# Output files
AUDIO_WAV = VIDEO_DIR / "tony-robbins-day12-audio.wav"
TRANSCRIPT_JSON = VIDEO_DIR / "tony-robbins-day12-transcript.json"
SPANISH_TEXT = VIDEO_DIR / "tony-robbins-day12-spanish.txt"
SPANISH_AUDIO = VIDEO_DIR / "tony-robbins-day12-spanish.mp3"
OUTPUT_VIDEO = VIDEO_DIR / "tony-robbins-day12-spanish.mp4"

# API Keys
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')


AUDIO_MP3 = VIDEO_DIR / "tony-robbins-day12-audio.mp3"


def step1_extract_audio():
    """Extract audio from video using FFmpeg (MP3 format, compressed for Whisper)"""
    print("\n" + "="*60)
    print("STEP 1: EXTRACT AUDIO")
    print("="*60)

    if AUDIO_MP3.exists():
        print(f"[SKIP] Audio already extracted: {AUDIO_MP3}")
        return True

    if not SOURCE_VIDEO.exists():
        print(f"[ERROR] Source video not found: {SOURCE_VIDEO}")
        return False

    # Extract as compressed MP3 to stay under 25MB Whisper limit
    cmd = [
        FFMPEG,
        "-i", str(SOURCE_VIDEO),
        "-vn",                    # No video
        "-acodec", "libmp3lame",  # MP3 codec
        "-ar", "16000",           # 16kHz sample rate (optimal for Whisper)
        "-ac", "1",               # Mono
        "-b:a", "32k",            # Low bitrate for small file
        "-y",                     # Overwrite
        str(AUDIO_MP3)
    ]

    print(f"Extracting audio to: {AUDIO_MP3}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        size_mb = AUDIO_MP3.stat().st_size / (1024*1024)
        print(f"[OK] Audio extracted ({size_mb:.2f} MB)")
        return True
    else:
        print(f"[ERROR] FFmpeg failed: {result.stderr}")
        return False


def split_and_transcribe():
    """Split large audio into chunks and transcribe each"""
    print("Splitting audio into chunks...")

    # Get audio duration using FFmpeg
    probe_cmd = [
        FFMPEG, "-i", str(AUDIO_MP3), "-f", "null", "-"
    ]
    result = subprocess.run(probe_cmd, capture_output=True, text=True)

    # Parse duration from stderr (FFmpeg outputs to stderr)
    import re
    duration_match = re.search(r'Duration: (\d+):(\d+):(\d+)', result.stderr)
    if duration_match:
        hours, mins, secs = map(int, duration_match.groups())
        total_seconds = hours * 3600 + mins * 60 + secs
    else:
        # Estimate from file size
        file_size_mb = AUDIO_MP3.stat().st_size / (1024*1024)
        total_seconds = int(file_size_mb / 0.24 * 60)

    print(f"Total duration: {total_seconds // 60} minutes {total_seconds % 60} seconds")

    # Split into 10-minute chunks (well under 25MB at 32kbps)
    chunk_duration = 600  # 10 minutes
    chunks = []
    offset = 0
    chunk_num = 0

    while offset < total_seconds:
        chunk_file = VIDEO_DIR / f"audio_chunk_{chunk_num:03d}.mp3"
        chunks.append(chunk_file)

        split_cmd = [
            FFMPEG,
            "-i", str(AUDIO_MP3),
            "-ss", str(offset),
            "-t", str(chunk_duration),
            "-acodec", "copy",
            "-y",
            str(chunk_file)
        ]

        subprocess.run(split_cmd, capture_output=True)
        offset += chunk_duration
        chunk_num += 1

    print(f"Created {len(chunks)} audio chunks")

    # Transcribe each chunk
    all_segments = []
    all_text = []
    time_offset = 0

    url = "https://api.openai.com/v1/audio/transcriptions"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}

    for i, chunk_file in enumerate(chunks):
        print(f"  Transcribing chunk {i+1}/{len(chunks)}...", end=" ")

        with open(chunk_file, 'rb') as f:
            files = {'file': (chunk_file.name, f, 'audio/mpeg')}
            data = {
                'model': 'whisper-1',
                'response_format': 'verbose_json',
                'timestamp_granularities[]': 'segment'
            }

            response = requests.post(url, headers=headers, files=files, data=data, timeout=600)

        if response.status_code == 200:
            chunk_transcript = response.json()
            all_text.append(chunk_transcript.get('text', ''))

            # Adjust segment timestamps
            for seg in chunk_transcript.get('segments', []):
                seg['start'] += time_offset
                seg['end'] += time_offset
                all_segments.append(seg)

            time_offset += chunk_duration
            print("done")
        else:
            print(f"FAILED ({response.status_code})")
            print(response.text)

        # Cleanup chunk file
        chunk_file.unlink(missing_ok=True)

    # Combine transcripts
    combined = {
        'text': ' '.join(all_text),
        'segments': all_segments
    }

    # Save combined transcript
    with open(TRANSCRIPT_JSON, 'w', encoding='utf-8') as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)

    print(f"[OK] Transcribed! {len(combined['text'])} characters total")
    return combined


def step2_transcribe():
    """Transcribe audio using OpenAI Whisper API"""
    print("\n" + "="*60)
    print("STEP 2: TRANSCRIBE (Whisper API)")
    print("="*60)

    if TRANSCRIPT_JSON.exists():
        print(f"[SKIP] Transcript already exists: {TRANSCRIPT_JSON}")
        with open(TRANSCRIPT_JSON, 'r', encoding='utf-8') as f:
            return json.load(f)

    if not AUDIO_MP3.exists():
        print(f"[ERROR] Audio file not found: {AUDIO_MP3}")
        return None

    # Check file size (Whisper limit is 25MB)
    file_size_mb = AUDIO_MP3.stat().st_size / (1024*1024)
    print(f"Audio file size: {file_size_mb:.2f} MB")

    if file_size_mb > 25:
        print("[ERROR] File exceeds 25MB Whisper limit. Need to split audio.")
        return split_and_transcribe()

    print("Calling OpenAI Whisper API...")
    # Cost is $0.006 per minute of audio
    # At 32kbps mono, file size in MB ≈ minutes * 0.24
    est_minutes = file_size_mb / 0.24
    print(f"Estimated duration: {est_minutes:.1f} minutes")
    print(f"Estimated cost: ${est_minutes * 0.006:.2f}")

    url = "https://api.openai.com/v1/audio/transcriptions"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}

    with open(AUDIO_MP3, 'rb') as f:
        files = {'file': (AUDIO_MP3.name, f, 'audio/mpeg')}
        data = {
            'model': 'whisper-1',
            'response_format': 'verbose_json',
            'timestamp_granularities[]': 'segment'
        }

        response = requests.post(url, headers=headers, files=files, data=data, timeout=600)

    if response.status_code == 200:
        transcript = response.json()

        # Save transcript
        with open(TRANSCRIPT_JSON, 'w', encoding='utf-8') as f:
            json.dump(transcript, f, indent=2, ensure_ascii=False)

        print(f"[OK] Transcribed! {len(transcript.get('text', ''))} characters")
        print(f"Saved to: {TRANSCRIPT_JSON}")
        return transcript
    else:
        print(f"[ERROR] Whisper API failed: {response.status_code}")
        print(response.text)
        return None


def step3_translate(transcript):
    """Translate transcript to Spanish using DeepL"""
    print("\n" + "="*60)
    print("STEP 3: TRANSLATE TO SPANISH (DeepL)")
    print("="*60)

    if SPANISH_TEXT.exists():
        print(f"[SKIP] Spanish text already exists: {SPANISH_TEXT}")
        with open(SPANISH_TEXT, 'r', encoding='utf-8') as f:
            return f.read()

    if not transcript:
        print("[ERROR] No transcript provided")
        return None

    # Get full text
    english_text = transcript.get('text', '')
    if not english_text:
        print("[ERROR] Empty transcript")
        return None

    print(f"English text: {len(english_text)} characters")

    try:
        from deep_translator import GoogleTranslator

        # DeepL has a 5000 char limit per request, so chunk if needed
        MAX_CHUNK = 4500
        chunks = []

        if len(english_text) > MAX_CHUNK:
            # Split by sentences
            sentences = english_text.replace('. ', '.|').split('|')
            current_chunk = ""

            for sentence in sentences:
                if len(current_chunk) + len(sentence) < MAX_CHUNK:
                    current_chunk += sentence
                else:
                    if current_chunk:
                        chunks.append(current_chunk)
                    current_chunk = sentence
            if current_chunk:
                chunks.append(current_chunk)
        else:
            chunks = [english_text]

        print(f"Translating {len(chunks)} chunk(s)...")

        translator = GoogleTranslator(source='en', target='es')
        spanish_chunks = []

        for i, chunk in enumerate(chunks):
            print(f"  Chunk {i+1}/{len(chunks)}...", end=" ")
            translated = translator.translate(chunk)
            spanish_chunks.append(translated)
            print("done")

        spanish_text = " ".join(spanish_chunks)

        # Save Spanish text
        with open(SPANISH_TEXT, 'w', encoding='utf-8') as f:
            f.write(spanish_text)

        print(f"[OK] Translated! {len(spanish_text)} characters")
        print(f"Saved to: {SPANISH_TEXT}")
        return spanish_text

    except ImportError:
        print("[ERROR] deep-translator not installed. Run: pip install deep-translator")
        return None
    except Exception as e:
        print(f"[ERROR] Translation failed: {e}")
        return None


def step4_generate_tts(spanish_text):
    """Generate Spanish audio using OpenAI TTS"""
    print("\n" + "="*60)
    print("STEP 4: GENERATE SPANISH AUDIO (OpenAI TTS)")
    print("="*60)

    if SPANISH_AUDIO.exists():
        print(f"[SKIP] Spanish audio already exists: {SPANISH_AUDIO}")
        return True

    if not spanish_text:
        print("[ERROR] No Spanish text provided")
        return False

    # OpenAI TTS has a 4096 character limit per request
    MAX_CHARS = 4000

    print(f"Spanish text: {len(spanish_text)} characters")
    print(f"Voice: onyx (male, deep, authoritative)")
    print(f"Estimated cost: ${len(spanish_text) * 0.000015:.2f}")

    url = "https://api.openai.com/v1/audio/speech"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }

    if len(spanish_text) <= MAX_CHARS:
        # Single request
        payload = {
            "model": "tts-1-hd",
            "input": spanish_text,
            "voice": "onyx"
        }

        print("Generating audio...")
        response = requests.post(url, headers=headers, json=payload, timeout=300)

        if response.status_code == 200:
            with open(SPANISH_AUDIO, 'wb') as f:
                f.write(response.content)

            size_mb = SPANISH_AUDIO.stat().st_size / (1024*1024)
            print(f"[OK] Audio generated ({size_mb:.2f} MB)")
            print(f"Saved to: {SPANISH_AUDIO}")
            return True
        else:
            print(f"[ERROR] TTS API failed: {response.status_code}")
            print(response.text)
            return False
    else:
        # Need to chunk and concatenate
        print(f"Text too long ({len(spanish_text)} chars), splitting into chunks...")

        # Split by sentences
        sentences = spanish_text.replace('. ', '.|').split('|')
        chunks = []
        current_chunk = ""

        for sentence in sentences:
            if len(current_chunk) + len(sentence) < MAX_CHARS:
                current_chunk += sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = sentence
        if current_chunk:
            chunks.append(current_chunk)

        print(f"Split into {len(chunks)} chunks")

        # Generate audio for each chunk
        chunk_files = []
        for i, chunk in enumerate(chunks):
            chunk_file = VIDEO_DIR / f"spanish_chunk_{i:03d}.mp3"
            chunk_files.append(chunk_file)

            print(f"  Generating chunk {i+1}/{len(chunks)}...", end=" ")

            payload = {
                "model": "tts-1-hd",
                "input": chunk,
                "voice": "onyx"
            }

            response = requests.post(url, headers=headers, json=payload, timeout=300)

            if response.status_code == 200:
                with open(chunk_file, 'wb') as f:
                    f.write(response.content)
                print("done")
            else:
                print(f"FAILED ({response.status_code})")
                return False

        # Concatenate chunks using FFmpeg
        print("Concatenating audio chunks...")

        # Create file list for FFmpeg
        list_file = VIDEO_DIR / "chunk_list.txt"
        with open(list_file, 'w') as f:
            for chunk_file in chunk_files:
                f.write(f"file '{chunk_file}'\n")

        concat_cmd = [
            FFMPEG,
            "-f", "concat",
            "-safe", "0",
            "-i", str(list_file),
            "-c", "copy",
            "-y",
            str(SPANISH_AUDIO)
        ]

        result = subprocess.run(concat_cmd, capture_output=True, text=True)

        # Cleanup chunk files
        for chunk_file in chunk_files:
            chunk_file.unlink(missing_ok=True)
        list_file.unlink(missing_ok=True)

        if result.returncode == 0:
            size_mb = SPANISH_AUDIO.stat().st_size / (1024*1024)
            print(f"[OK] Audio generated and concatenated ({size_mb:.2f} MB)")
            return True
        else:
            print(f"[ERROR] Concatenation failed: {result.stderr}")
            return False


def step5_composite_video():
    """Composite Spanish audio onto original video"""
    print("\n" + "="*60)
    print("STEP 5: COMPOSITE FINAL VIDEO")
    print("="*60)

    if OUTPUT_VIDEO.exists():
        print(f"[SKIP] Output video already exists: {OUTPUT_VIDEO}")
        return True

    if not SPANISH_AUDIO.exists():
        print(f"[ERROR] Spanish audio not found: {SPANISH_AUDIO}")
        return False

    # Replace audio track
    cmd = [
        FFMPEG,
        "-i", str(SOURCE_VIDEO),
        "-i", str(SPANISH_AUDIO),
        "-c:v", "copy",           # Keep original video
        "-c:a", "aac",            # Encode audio as AAC
        "-map", "0:v:0",          # Take video from first input
        "-map", "1:a:0",          # Take audio from second input
        "-shortest",              # End when shortest stream ends
        "-y",
        str(OUTPUT_VIDEO)
    ]

    print(f"Creating Spanish video: {OUTPUT_VIDEO}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        size_mb = OUTPUT_VIDEO.stat().st_size / (1024*1024)
        print(f"[OK] Video created ({size_mb:.2f} MB)")
        print(f"Saved to: {OUTPUT_VIDEO}")
        return True
    else:
        print(f"[ERROR] Video composite failed: {result.stderr}")
        return False


def main():
    print("="*60)
    print("VIDEO TRANSLATION PIPELINE")
    print("Tony Robbins Goal Setting Workshop - English to Spanish")
    print("="*60)

    # Check prerequisites
    if not OPENAI_API_KEY:
        print("[ERROR] OPENAI_API_KEY not found in ~/.env")
        return 1

    if not Path(FFMPEG).exists():
        print(f"[ERROR] FFmpeg not found at: {FFMPEG}")
        return 1

    if not SOURCE_VIDEO.exists():
        print(f"[ERROR] Source video not found: {SOURCE_VIDEO}")
        return 1

    print(f"[OK] Prerequisites checked")
    print(f"Source: {SOURCE_VIDEO}")
    print(f"Output: {OUTPUT_VIDEO}")

    # Run pipeline
    if not step1_extract_audio():
        return 1

    transcript = step2_transcribe()
    if not transcript:
        return 1

    spanish_text = step3_translate(transcript)
    if not spanish_text:
        return 1

    if not step4_generate_tts(spanish_text):
        return 1

    if not step5_composite_video():
        return 1

    print("\n" + "="*60)
    print("PIPELINE COMPLETE!")
    print("="*60)
    print(f"Spanish video: {OUTPUT_VIDEO}")
    print()
    print("To test, open the video in VLC or update index.html to use the Spanish version.")

    return 0


if __name__ == "__main__":
    exit(main())
