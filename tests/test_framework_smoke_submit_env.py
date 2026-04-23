import re
from pathlib import Path


def test_framework_smoke_job_leaves_slurm_cuda_visible_devices_owned_by_cluster():
    script = Path(__file__).resolve().parents[1] / "scripts" / "run_framework_smoke_model.sh"
    text = script.read_text(encoding="utf-8")

    assert "#SBATCH --export=NONE" in text
    assert "unset CUDA_VISIBLE_DEVICES" not in text
    assert "export CUDA_VISIBLE_DEVICES" not in text
    assert re.search(r"(^|[^A-Za-z0-9_])CUDA_VISIBLE_DEVICES=", text) is None


def test_framework_smoke_job_has_configurable_model_job_limit():
    script = Path(__file__).resolve().parents[1] / "scripts" / "run_framework_smoke_model.sh"
    text = script.read_text(encoding="utf-8")

    assert 'MAX_MODEL_JOBS="${2:-1}"' in text
    assert 'if [[ "$MAX_MODEL_JOBS" == "all" ]]; then' in text
    assert '--max_model_jobs "$MAX_MODEL_JOBS"' in text


def test_framework_smoke_job_has_configurable_sample_limit():
    script = Path(__file__).resolve().parents[1] / "scripts" / "run_framework_smoke_model.sh"
    text = script.read_text(encoding="utf-8")

    assert 'MAX_SAMPLES="${3:-1}"' in text
    assert 'if [[ "$MAX_SAMPLES" == "all" ]]; then' in text
    assert 'common=(--project_root "$ROOT" --max_model_jobs "$MAX_MODEL_JOBS" --max_samples "$MAX_SAMPLES")' in text
    assert re.search(r"--max_samples 1(?![0-9])", text) is None
