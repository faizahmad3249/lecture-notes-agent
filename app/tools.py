# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tools for the Lecture Voice-to-Notes agent."""

import base64
import os
from pathlib import Path

from google import genai
from google.genai import types


def _get_client() -> genai.Client:
    """Return a configured Gemini client."""
    use_vertexai = (
        os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "False").lower() == "true"
    )
    if use_vertexai:
        return genai.Client(
            vertexai=True,
            project=os.environ.get("GOOGLE_CLOUD_PROJECT"),
            location=os.environ.get("GOOGLE_CLOUD_LOCATION", "global"),
        )
    return genai.Client()


def transcribe_audio(audio_file_path: str) -> str:
    """Transcribe a lecture audio file to text using Gemini's multimodal understanding.

    The audio file is read from disk, encoded as base64, and sent directly to
    Gemini for transcription. Supported formats: mp3, wav, ogg, flac, m4a, aac.

    Args:
        audio_file_path: Absolute or relative path to the audio file on disk.

    Returns:
        A plain-text transcript of the spoken content in the audio file.
        Returns an error message string if the file cannot be read or processed.
    """
    path = Path(audio_file_path)

    if not path.exists():
        return f"Error: Audio file not found at path '{audio_file_path}'. Please check the file path and try again."

    if not path.is_file():
        return f"Error: '{audio_file_path}' is not a file."

    suffix = path.suffix.lower().lstrip(".")
    mime_map = {
        "mp3": "audio/mpeg",
        "wav": "audio/wav",
        "ogg": "audio/ogg",
        "flac": "audio/flac",
        "m4a": "audio/mp4",
        "aac": "audio/aac",
        "webm": "audio/webm",
    }
    mime_type = mime_map.get(suffix, f"audio/{suffix}")

    try:
        audio_bytes = path.read_bytes()
    except OSError as e:
        return f"Error: Could not read audio file: {e}"

    audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")

    client = _get_client()
    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=[
            types.Content(
                role="user",
                parts=[
                    types.Part(
                        inline_data=types.Blob(
                            mime_type=mime_type,
                            data=audio_b64,
                        )
                    ),
                    types.Part(
                        text=(
                            "Please provide a complete, verbatim transcript of this audio. "
                            "Include all spoken words. Do not summarize — output only the raw transcript text."
                        )
                    ),
                ],
            )
        ],
    )
    return response.text
