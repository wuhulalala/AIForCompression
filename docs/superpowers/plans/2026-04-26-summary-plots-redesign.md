# Summary Plots Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make summary plots show the CAESAR `eb` sweep clearly while keeping only one `CAESAR-V` point and making image/video distinctions more explicit.

**Architecture:** Extend `utils/plot_results_summary.py` with small CAESAR-specific metadata helpers and use them during plotting, instead of changing the source result format. Keep the existing CSV and data collection flow, but annotate CAESAR-D points and tighten the subset/style logic used by the generated figures.

**Tech Stack:** Python, `unittest`, Matplotlib

---

### Task 1: Lock in behavior with tests

**Files:**
- Modify: `tests/test_plot_results_summary.py`

- [ ] **Step 1: Add a failing test for CAESAR error-bound parsing**

```python
    def test_caesar_error_bound_label_parses_from_source_name(self):
        module = load_module()

        self.assertIsNone(module.caesar_error_bound_label("CAESAR"))
        self.assertEqual("1e-3", module.caesar_error_bound_label("CAESAR_eb_1em3"))
        self.assertEqual("1.5e-3", module.caesar_error_bound_label("CAESAR_eb_1p5em3"))
```

- [ ] **Step 2: Run the targeted test and verify it fails**

Run: `python -m unittest tests.test_plot_results_summary.PlotResultsSummaryTest.test_caesar_error_bound_label_parses_from_source_name`
Expected: FAIL with `AttributeError` because `caesar_error_bound_label` does not exist yet

- [ ] **Step 3: Add a failing test for CAESAR plot annotations**

```python
    def test_caesar_annotation_text_only_marks_caesar_d_error_bounds(self):
        module = load_module()

        caesar_v = module.normalize_record(
            "CAESAR_eb_1em3",
            {"arch": "CAESAR", "model_id": "caesar_v", "psnr": 87.0, "compression_ratio": 177.0},
        )
        caesar_d = module.normalize_record(
            "CAESAR_eb_1em3",
            {"arch": "CAESAR", "model_id": "caesar_d", "psnr": 69.0, "compression_ratio": 338.0},
        )

        self.assertIsNone(module.caesar_annotation_text(caesar_v))
        self.assertEqual("eb=1e-3", module.caesar_annotation_text(caesar_d))
```

- [ ] **Step 4: Run the targeted test and verify it fails**

Run: `python -m unittest tests.test_plot_results_summary.PlotResultsSummaryTest.test_caesar_annotation_text_only_marks_caesar_d_error_bounds`
Expected: FAIL with `AttributeError` because `caesar_annotation_text` does not exist yet

### Task 2: Implement CAESAR metadata helpers and plotting behavior

**Files:**
- Modify: `utils/plot_results_summary.py`

- [ ] **Step 1: Add helper functions for CAESAR error-bound labels**

```python
def caesar_error_bound_label(source_name):
    ...


def caesar_annotation_text(row):
    ...
```

- [ ] **Step 2: Use the helper during plotting so CAESAR-D points show `eb`**

```python
        annotation = caesar_annotation_text(item)
        if annotation:
            ax.annotate(...)
```

- [ ] **Step 3: Make family-level styling more explicit without changing data collection**

```python
    family_palettes = {
        "image": [...],
        "video": [...],
    }
```

- [ ] **Step 4: Keep only one CAESAR-V point and preserve the CAESAR-D sweep**

```python
def prepare_rows_for_plot(rows):
    ...
```

### Task 3: Verify and regenerate outputs

**Files:**
- Modify: `logs/results/summary_plots/*.png`

- [ ] **Step 1: Run the plot summary unit tests**

Run: `python -m unittest tests.test_plot_results_summary`
Expected: PASS

- [ ] **Step 2: Regenerate summary plots**

Run: `python utils/plot_results_summary.py`
Expected: `Wrote ... rows and plots to .../logs/results/summary_plots`

- [ ] **Step 3: Check git diff for the intended files only**

Run: `git status --short`
Expected: changes in the plotting script, its tests, the plan/spec docs, and regenerated plot assets
