#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Step 2a — instruction decomposition via a text LLM.

Given a long-horizon instruction and an ordered keyframe-meaning sequence
``[(index, role), ...]``, an external text LLM is asked to emit one atomic
sub-instruction per keyframe, aligned to its semantic role.

This module only defines the I/O wrapper. The LLM client itself lives in a
separate module — set ``LLM_CLIENT_MODULE`` below to its import path. The
client must expose a callable ``call(system_prompt, user_prompt) -> str``.

Prompt templates are intentionally left empty here: fill in
``SYSTEM_PROMPT`` and ``USER_PROMPT_TEMPLATE`` before running.
"""

from __future__ import annotations

import importlib
import json
from typing import Any, Dict, List, Optional, Tuple

# Import path of the external text-LLM client (e.g. "src.datasets.preprocessing.claude_client").
# Leave as None until the client module is created.
LLM_CLIENT_MODULE: Optional[str] = None

# Prompt A — instruction decomposition. Fill in before use.
SYSTEM_PROMPT: str = ""
USER_PROMPT_TEMPLATE: str = ""


def _load_client():
    """Import and return the external LLM client module."""
    if not LLM_CLIENT_MODULE:
        raise RuntimeError(
            "LLM_CLIENT_MODULE is not set. Point it at the module that "
            "implements `call(system_prompt, user_prompt) -> str`."
        )
    return importlib.import_module(LLM_CLIENT_MODULE)


def decompose_instruction(
    original_instruction: str,
    keyframe_sequence: List[Tuple[int, str]],
) -> List[Dict[str, Any]]:
    """Decompose a long-horizon instruction into per-keyframe sub-instructions.

    Args:
        original_instruction: The long-horizon task instruction.
        keyframe_sequence: Ordered list of ``(keyframe_index, role)`` tuples,
            where ``role`` is the keyframe's semantic label
            (``Start`` / ``Pre-Action`` / ``Gripper`` / ``Stop`` / ``Apex`` / ``End``).

    Returns:
        ``[{"keyframe_index": int, "sub_instruction": str}, ...]`` aligned to
        ``keyframe_sequence``.
    """
    if not SYSTEM_PROMPT or not USER_PROMPT_TEMPLATE:
        raise RuntimeError(
            "Prompt A is empty. Set SYSTEM_PROMPT and USER_PROMPT_TEMPLATE "
            "in instruction_decomposition.py before calling decompose_instruction()."
        )

    user_prompt = USER_PROMPT_TEMPLATE.format(
        original_instruction=original_instruction,
        keyframe_sequence=json.dumps(keyframe_sequence, ensure_ascii=False),
    )

    client = _load_client()
    raw = client.call(SYSTEM_PROMPT, user_prompt)
    parsed = json.loads(raw)

    if not isinstance(parsed, list):
        raise ValueError(f"Expected a JSON list from the LLM, got: {type(parsed)}")
    return parsed
