# DCAE Kodak Script Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a standalone `models/DCAE/test_kodak.py` script that mirrors the structure of `test_era5.py` but evaluates DCAE on Kodak images.

**Architecture:** Keep the Kodak runner independent from the shared compression pipeline so it can mirror DCAE's model-local evaluation style. Reuse the DCAE checkpoint discovery and image compress/decompress flow, but replace ERA5-specific normalization and channel grouping logic with Kodak RGB image loading and padding/cropping.

**Tech Stack:** Python, PyTorch, NumPy, PIL, argparse, unittest/pytest

---

### Task 1: Add test coverage for Kodak script helpers

**Files:**
- Create: `tests/test_dcae_kodak.py`

- [ ] Add tests for checkpoint discovery and Kodak image enumeration helpers.
- [ ] Run the new test file and verify it fails before implementation.

### Task 2: Implement standalone Kodak evaluation script

**Files:**
- Create: `models/DCAE/test_kodak.py`

- [ ] Implement Kodak image discovery, padding/cropping, checkpoint loading, and real compress/decompress evaluation.
- [ ] Match `test_era5.py` output style with per-sample results written to `summary.json`.

### Task 3: Verify the new script

**Files:**
- Test: `tests/test_dcae_kodak.py`

- [ ] Run the targeted tests and confirm they pass.
- [ ] Provide the exact Kodak invocation command for manual use.
