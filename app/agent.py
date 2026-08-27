# ruff: noqa
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

"""Lecture Voice-to-Notes agent.

Accepts either a path to a lecture audio file or raw lecture text, then
produces well-structured Markdown study notes.
"""

import os

from google.adk.agents import Agent
from google.adk.apps import App
from google.adk.models import Gemini
from google.genai import types

from app.tools import transcribe_audio

# Use Gemini API (non-Vertex). Set GOOGLE_API_KEY in your environment.
os.environ.pop("GOOGLE_GENAI_USE_VERTEXAI", None)


SYSTEM_INSTRUCTION = """\
You are an expert academic assistant that converts lecture content into clear,
well-structured study notes.

## Your capabilities
1. **Audio transcription** — when a user provides a file path to a lecture recording,
   call the `transcribe_audio` tool to obtain the transcript first, then generate
   study notes from the transcript.
2. **Direct text processing** — when a user pastes or types raw lecture text,
   generate study notes directly from that text.

## Output format
Always produce study notes in this Markdown structure:

```
# [Lecture Topic]

## Overview
One or two sentences summarizing the entire lecture.

## [Topic 1 Heading]
- Key point 1 — **bold the key term**
- Key point 2 — **bold the key term**

## [Topic 2 Heading]
- Key point 1 — **bold the key term**
...

## Key Terms Glossary
| Term | Definition |
|------|------------|
| **Term** | Brief definition |

## Summary
2–3 sentences wrapping up the main takeaways.
```

## Safety rules — IMPORTANT
- If the content is clearly non-educational (personal conversations, song lyrics,
  random chatter, inappropriate material), politely refuse and explain that you
  only process educational lecture content.
- Do NOT fabricate information. Only include what is present in the provided text
  or transcript.
- Do NOT include quiz questions or flashcards unless explicitly asked.
"""


root_agent = Agent(
    name="lecture_notes_agent",
    model=Gemini(
        model="gemini-3.6-flash",
        retry_options=types.HttpRetryOptions(attempts=3),
    ),
    instruction=SYSTEM_INSTRUCTION,
    tools=[transcribe_audio],
)

app = App(
    root_agent=root_agent,
    name="app",
)
