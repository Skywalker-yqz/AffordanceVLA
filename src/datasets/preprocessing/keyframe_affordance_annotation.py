#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Step 2b — per-keyframe affordance annotation via a VLM.

Given the original instruction, the keyframe sequence, the current keyframe
index/meaning, the sub-instruction from Step 2a, and the keyframe image,
a VLM is asked to emit:
    - detection_category: one concise noun phrase identifying the part the
      gripper interacts with (used as the Which2Act target).
    - affordance_instruction: a where-to referring expression for the
      interaction (used as the Where2Act query).

Two backends are supported. The default is RexOmni; Qwen3-VL is available as
a side path. Endpoints / API URLs are not hard-coded — supply them via
environment variables (see ``EnvVars`` below).

Prompt templates are intentionally left empty: fill in ``SYSTEM_PROMPT`` and
``USER_PROMPT_TEMPLATE`` before running.
"""

from __future__ import annotations

import base64
import importlib
import io
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image

# Backend selection: "rexomni" (default) | "qwen3vl"
ANNOTATION_BACKEND: str = "rexomni"

# Import path of an optional external VLM client when the bundled
# RexOmni / Qwen3-VL backends are insufficient. Same contract as the
# Step 2a client: ``call_with_image(system_prompt, user_prompt, image) -> str``.
VLM_CLIENT_MODULE: Optional[str] = None

# Prompt B — per-keyframe affordance annotation. Fill in before use.
SYSTEM_PROMPT: str = ""
USER_PROMPT_TEMPLATE: str = ""


class EnvVars:
    """Names of the environment variables consulted by this module."""
    REXOMNI_URL = "REXOMNI_VLM_URL"
    REXOMNI_TOKEN = "REXOMNI_VLM_TOKEN"
    QWEN3VL_URL = "QWEN3_VL_API_URL"
    QWEN3VL_TOKEN = "QWEN3_VL_API_TOKEN"


@dataclass
class AnnotationResult:
    """Structured output of Step 2b for one keyframe."""
    detection_category: str
    affordance_instruction: str


def _encode_image(image: Union[str, Image.Image, np.ndarray]) -> str:
    """Encode an image to a base64 string (JPEG)."""
    if isinstance(image, str):
        with open(image, "rb") as f:
            data = f.read()
    elif isinstance(image, Image.Image):
        buf = io.BytesIO()
        image.convert("RGB").save(buf, format="JPEG")
        data = buf.getvalue()
    elif isinstance(image, np.ndarray):
        buf = io.BytesIO()
        Image.fromarray(image).convert("RGB").save(buf, format="JPEG")
        data = buf.getvalue()
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")
    return base64.b64encode(data).decode("utf-8")


def _build_user_prompt(
    original_instruction: str,
    keyframe_sequence: List[Tuple[int, str]],
    current_index: int,
    current_role: str,
    sub_instruction: str,
) -> str:
    """Render Prompt B's user message."""
    if not USER_PROMPT_TEMPLATE:
        raise RuntimeError(
            "Prompt B is empty. Set SYSTEM_PROMPT and USER_PROMPT_TEMPLATE "
            "in keyframe_affordance_annotation.py before annotating."
        )
    return USER_PROMPT_TEMPLATE.format(
        original_instruction=original_instruction,
        keyframe_sequence=json.dumps(keyframe_sequence, ensure_ascii=False),
        current_index=current_index,
        current_role=current_role,
        sub_instruction=sub_instruction,
    )


def _call_vlm(system_prompt: str, user_prompt: str, image_b64: str) -> str:
    """Dispatch the VLM call to the configured backend."""
    if VLM_CLIENT_MODULE:
        client = importlib.import_module(VLM_CLIENT_MODULE)
        return client.call_with_image(system_prompt, user_prompt, image_b64)

    backend = ANNOTATION_BACKEND.lower()
    if backend == "rexomni":
        url = os.environ.get(EnvVars.REXOMNI_URL)
        token = os.environ.get(EnvVars.REXOMNI_TOKEN)
    elif backend == "qwen3vl":
        url = os.environ.get(EnvVars.QWEN3VL_URL)
        token = os.environ.get(EnvVars.QWEN3VL_TOKEN)
    else:
        raise ValueError(
            f"Unknown ANNOTATION_BACKEND: {ANNOTATION_BACKEND}. "
            f"Use 'rexomni' or 'qwen3vl', or provide VLM_CLIENT_MODULE."
        )

    if not url or not token:
        raise RuntimeError(
            f"Missing endpoint for backend '{backend}'. "
            f"Set the environment variables required by EnvVars."
        )

    import requests

    payload = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                ],
            },
        ],
        "temperature": 0.0,
    }
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    resp = requests.post(url, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


def annotate_keyframe(
    image: Union[str, Image.Image, np.ndarray],
    original_instruction: str,
    keyframe_sequence: List[Tuple[int, str]],
    current_index: int,
    current_role: str,
    sub_instruction: str,
) -> AnnotationResult:
    """Run Prompt B for a single keyframe and return its parsed result.

    Args:
        image: Keyframe image (path, PIL.Image or HxWx3 RGB array).
        original_instruction: The long-horizon task instruction.
        keyframe_sequence: ``[(index, role), ...]`` for the whole trajectory.
        current_index: Index of the keyframe being annotated.
        current_role: Role label of the current keyframe.
        sub_instruction: Sub-instruction produced by Step 2a.

    Returns:
        AnnotationResult with the detection category and affordance instruction.
    """
    if not SYSTEM_PROMPT:
        raise RuntimeError(
            "Prompt B is empty. Set SYSTEM_PROMPT and USER_PROMPT_TEMPLATE "
            "in keyframe_affordance_annotation.py before annotating."
        )

    user_prompt = _build_user_prompt(
        original_instruction=original_instruction,
        keyframe_sequence=keyframe_sequence,
        current_index=current_index,
        current_role=current_role,
        sub_instruction=sub_instruction,
    )
    image_b64 = _encode_image(image)

    raw = _call_vlm(SYSTEM_PROMPT, user_prompt, image_b64)
    parsed: Dict[str, Any] = json.loads(raw)

    return AnnotationResult(
        detection_category=parsed["detection_category"],
        affordance_instruction=parsed["affordance_instruction"],
    )
