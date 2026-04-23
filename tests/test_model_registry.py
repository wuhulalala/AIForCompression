import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from compression_pipeline.model_registry import image_model_jobs


def test_lictcm_large_checkpoint_uses_large_model_id_and_loader_default():
    jobs = list(image_model_jobs(Path(__file__).resolve().parents[1], {"LIC_TCM"}))
    large_jobs = [job for job in jobs if job.checkpoint and job.checkpoint.endswith("mse_lambda_0.05.pth.tar")]

    assert len(large_jobs) == 1
    assert large_jobs[0].model_id == "LICTCM_mse_lambda_0.05_large"
