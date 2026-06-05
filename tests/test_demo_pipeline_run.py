from pathlib import Path

from image_analysis.generate_demo_data import generate_demo_data
from image_analysis.pipeline import run_pipeline


def test_demo_pipeline_runs(tmp_path):
    data_dir = tmp_path / "data"
    output_dir = tmp_path / "outputs"

    # Generate tiny synthetic dataset
    generate_demo_data(
        output_dir=data_dir,
        num_images=2,
    )

    # Run the pipeline
    run_pipeline(
        data_dir=data_dir,
        output_dir=output_dir,
    )

    # Verify expected outputs exist
    assert output_dir.exists()
    assert any(output_dir.iterdir())
