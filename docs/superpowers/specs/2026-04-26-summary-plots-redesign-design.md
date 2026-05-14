# Summary Plots Redesign

## Goal

Improve `utils/plot_results_summary.py` so the generated summary plots clearly highlight:

- scientific compression vs generic compression
- image compression vs video compression
- CAESAR `V` and `D` modes as separate series

## Decisions

- Split CAESAR into `CAESAR-V` and `CAESAR-D`.
- Keep only one `CAESAR-V` point because its compression ratio is effectively not tunable in the current results.
- Keep all `CAESAR-D` points and connect them in compression-ratio order so the error-bound trend stays visible.
- Generate multiple trade-off figures instead of forcing all models into one crowded figure.
- Keep the x-axis in log scale, but add a small set of explicit reference ticks such as `500` and `1000` when they fall inside the plotted range.

## Output

The plotting script should emit:

- a scientific-vs-generic comparison figure
- an image-vs-video comparison figure
- a CAESAR-only comparison figure
- metric-specific figures that use the cleaned plotting rows

## Constraints

- Preserve existing CSV generation.
- Preserve parameter plots.
- Avoid overwriting unrelated user changes in the repository.
