"""
config.py - Configuration for the WELS Translation Pipeline

This file stores all the settings for the pipeline in one place.
When you want to change which model you're using, which languages
you're targeting, or how the evaluation works, you change it here
instead of digging through multiple files.

HOW TO USE YOUR API KEY:
  Before running any pipeline scripts, set your API key in the terminal:

  On Windows (PowerShell):
    $env:ANTHROPIC_API_KEY = "your-key-here"

  On Mac/Linux:
    export ANTHROPIC_API_KEY="your-key-here"

  NEVER paste your API key directly into this file or any code file.
  If you accidentally commit it to GitHub, anyone can use it and
  rack up charges on your account.
"""

import os

# ---------------------------------------------------------------------------
# API SETTINGS
# ---------------------------------------------------------------------------

EVAL_MODEL = "claude-sonnet-4-6"

TRANSLATE_MODEL = "claude-opus-4-6"

BACK_TRANSLATE_MODEL = "claude-sonnet-4-6"

# Maximum tokens the model can generate in one response.
MAX_TOKENS = 4096

# ---------------------------------------------------------------------------
# LANGUAGE SETTINGS
# ---------------------------------------------------------------------------
SOURCE_LANGUAGE = "German"

# "Resource level" = how much training data AI models have seen in this language.
TARGET_LANGUAGES = {
    "high":   "Spanish",
    "medium": "Hungarian",
    "low":    "Hmong",
}

DEFAULT_TARGET = "Hungarian"

# ---------------------------------------------------------------------------
# EVALUATION RUBRIC WEIGHTS
# ---------------------------------------------------------------------------
# Each dimension gets a weight that reflects how important it is.
# The weights should add up to 1.0 (i.e., 100%).
#
# Current ranking:
#   1. Doctrinal accuracy (non-negotiable)
#   2. Consistent theological terminology (glossary alignment)
#   3. Clarity and understandability for native speakers
#   4. Natural readability / style
EVAL_WEIGHTS = {
    "doctrinal_accuracy":    0.40,  # Most important - non-negotiable
    "terminology_consistency": 0.25,  # Key theological terms must be consistent
    "clarity":               0.20,  # Must be understandable to native speakers
    "naturalness":           0.15,  # Should read smoothly (but lowest priority)
}

# ---------------------------------------------------------------------------
# FILE PATHS
# ---------------------------------------------------------------------------
# Where chunks are stored (the source texts).
CHUNKS_FILE = os.path.join(os.path.dirname(__file__), "..", "chunks.json")

# Where Benjamin's existing translations are stored.
PASS2_DIR = os.path.join(os.path.dirname(__file__), "..", "pass2")
FINAL_DIFFER_DIR = os.path.join(os.path.dirname(__file__), "..", "final_differ")

# Where our evaluation results will be saved.
EVAL_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "eval", "results")
