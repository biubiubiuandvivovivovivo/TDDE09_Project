"""Utility helpers for building and evaluating simple SciQ baselines.

This module extracts reusable logic from the exploratory notebook and fixes
common evaluation pitfalls:
- deterministic shuffling without global random side-effects
- robust answer letter extraction from model outputs
- clearer prompt builders shared by LLM-only and RAG baselines
"""

from __future__ import annotations

import random
import re
from typing import List, Sequence, Tuple

CHOICE_LETTERS = "ABCD"


def make_mcq(example: dict, seed: int = 0) -> Tuple[str, List[str], str]:
    """Build a shuffled 4-choice question from a SciQ example.

    Args:
        example: A SciQ row containing question, correct_answer, and distractors.
        seed: Random seed used only for this sample shuffle.

    Returns:
        (question, choices, correct_letter)
    """
    rnd = random.Random(seed)
    question = example["question"]

    choices = [
        example["distractor1"],
        example["distractor2"],
        example["distractor3"],
        example["correct_answer"],
    ]
    rnd.shuffle(choices)

    correct_idx = choices.index(example["correct_answer"])
    correct_letter = CHOICE_LETTERS[correct_idx]
    return question, choices, correct_letter


def format_prompt(question: str, choices: Sequence[str]) -> str:
    """Create a strict 4-option MCQ prompt."""
    if len(choices) != 4:
        raise ValueError(f"Expected 4 choices, got {len(choices)}")

    return (
        "Answer the multiple-choice question.\n"
        "Reply with ONLY A, B, C, or D.\n\n"
        f"Question: {question}\n"
        f"A) {choices[0]}\n"
        f"B) {choices[1]}\n"
        f"C) {choices[2]}\n"
        f"D) {choices[3]}\n\n"
        "Answer:"
    )


def format_rag_prompt(context_passages: Sequence[str], question: str, choices: Sequence[str]) -> str:
    """Create an MCQ prompt augmented with retrieved context passages."""
    context = "\n\n".join(f"[Context {i + 1}]\n{p}" for i, p in enumerate(context_passages))
    return (
        "Use the context to answer the multiple-choice question.\n"
        "Reply with ONLY A, B, C, or D. No explanation.\n\n"
        f"{context}\n\n"
        f"Question: {question}\n"
        f"A) {choices[0]}\n"
        f"B) {choices[1]}\n"
        f"C) {choices[2]}\n"
        f"D) {choices[3]}\n\n"
        "Answer:"
    )


def extract_letter(text: str) -> str | None:
    """Extract the first standalone answer letter (A-D) from model output.

    This avoids false positives from words like "Because" (contains "B").
    """
    if not text:
        return None

    upper = text.upper()
    match = re.search(r"\b([ABCD])\b", upper)
    return match.group(1) if match else None
