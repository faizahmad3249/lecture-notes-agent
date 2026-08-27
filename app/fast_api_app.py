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
import logging
import os

from fastapi import FastAPI
from google.adk.cli.fast_api import get_fast_api_app

from app.app_utils.telemetry import setup_telemetry
from app.app_utils.typing import Feedback

logger = logging.getLogger(__name__)

setup_telemetry()

# Default to wildcard so the ADK Dev UI works on any domain (including Render).
# Override by setting ALLOW_ORIGINS=https://example.com,https://other.com
_allow_origins_env = os.getenv("ALLOW_ORIGINS", "")
allow_origins: list[str] = (
    _allow_origins_env.split(",") if _allow_origins_env else ["*"]
)

AGENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# In-memory session configuration - no persistent storage
session_service_uri = None

app: FastAPI = get_fast_api_app(
    agents_dir=AGENT_DIR,
    web=True,
    allow_origins=allow_origins,
    session_service_uri=session_service_uri,
)
app.title = "lecture-notes-agent"
app.description = "API for interacting with the Agent lecture-notes-agent"


@app.post("/feedback")
def collect_feedback(feedback: Feedback) -> dict[str, str]:
    """Collect and log feedback.

    Args:
        feedback: The feedback data to log

    Returns:
        Success message
    """
    logger.info("Feedback received: %s", feedback.model_dump())
    return {"status": "success"}


# Main execution
if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
