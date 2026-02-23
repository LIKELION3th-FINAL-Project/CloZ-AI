import os
import time
from pathlib import Path
from typing import Optional

from loguru import logger


class GeminiVideoPreviewGenerator:
    """
    Generate short video previews from a generated outfit image.

    Toggle with env:
      - ENABLE_VEO_VIDEO_PREVIEW=true|false (default: true)

    Note:
      - This module is intentionally isolated so it can be disabled/removed
        without touching the core outfit generation flow.
    """

    DEFAULT_PROMPT = (
        "Use smooth, natural camera movement to orbit around the subject and "
        "showcase the full front view, while the person holds relaxed, natural poses."
    )

    def __init__(self):
        self.enabled = os.getenv("ENABLE_VEO_VIDEO_PREVIEW", "true").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        self.api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        self.model_name = "veo-3.1-fast-generate-preview"
        self.duration_seconds = 4
        self.aspect_ratio = "9:16"
        self.resolution = "720p"
        self.max_poll_seconds = int(os.getenv("VEO_MAX_POLL_SECONDS", "240"))
        self.poll_interval_seconds = int(os.getenv("VEO_POLL_INTERVAL_SECONDS", "8"))

    def generate_from_image(
        self,
        image_path: str,
        prompt: Optional[str] = None,
        output_path: Optional[str] = None,
    ) -> Optional[str]:
        if not self.enabled:
            logger.info("[VEO] video preview disabled by ENABLE_VEO_VIDEO_PREVIEW")
            return None
        if not image_path:
            return None
        if not self.api_key:
            logger.warning("[VEO] GOOGLE_API_KEY/GEMINI_API_KEY not found. Skipping video.")
            return None

        src = Path(image_path)
        if not src.exists():
            logger.warning(f"[VEO] source image not found: {image_path}")
            return None

        try:
            from google import genai
        except Exception as e:
            logger.warning(f"[VEO] google-genai not available ({e}). Skipping video.")
            return None

        if output_path:
            out_path = Path(output_path)
        else:
            out_path = src.with_name(f"{src.stem}_veo.mp4")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        user_prompt = prompt or self.DEFAULT_PROMPT

        try:
            client = genai.Client(api_key=self.api_key)
            from google.genai import types

            operation = client.models.generate_videos(
                model=self.model_name,
                source=types.GenerateVideosSource(
                    prompt=user_prompt,
                    image=types.Image.from_file(location=str(src)),
                ),
                config=types.GenerateVideosConfig(
                    duration_seconds=self.duration_seconds,
                    aspect_ratio=self.aspect_ratio,
                    resolution=self.resolution,
                    number_of_videos=1,
                ),
            )

            started = time.time()
            while not getattr(operation, "done", False):
                if time.time() - started > self.max_poll_seconds:
                    logger.warning("[VEO] polling timeout. Skipping video save.")
                    return None
                time.sleep(self.poll_interval_seconds)
                operation = client.operations.get(operation)

            response = getattr(operation, "response", None) or getattr(operation, "result", None)
            if not response:
                logger.warning("[VEO] operation completed but response missing.")
                return None

            videos = getattr(response, "generated_videos", None) or []
            if not videos:
                logger.warning("[VEO] generated_videos is empty.")
                return None

            video_ref = getattr(videos[0], "video", None)
            if not video_ref:
                logger.warning("[VEO] video reference missing in generated_videos.")
                return None

            client.files.download(file=video_ref)
            video_ref.save(str(out_path))

            logger.info(f"[VEO] saved video preview: {out_path}")
            return str(out_path)
        except Exception as e:
            logger.warning(f"[VEO] failed to generate video preview: {e}")
            return None
