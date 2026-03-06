# TDDE09 Project

This repository contains baseline utilities for SciQ-style multiple-choice QA
and retrieval-augmented prompting.

## Improvements included

- Reusable helper module for:
  - deterministic MCQ construction
  - strict prompt formatting for LLM-only and RAG settings
  - robust answer-letter extraction (`A/B/C/D`) from generated text
- Lightweight unit tests for core logic.

## Run tests

```bash
python -m unittest discover -s tests -v
```

## Existing notebook

The exploratory workflow remains in `test.ipynb`. The new `src/sciq_baselines.py`
module is intended as a cleaner, testable foundation for future experiments.
