import subprocess

# Run the training pipeline with full months of data
cmd = """python run_full_pipeline.py train \
    --train-years 2023 \
    --val '2024-04-01:2024-04-30' \
    --test '2024-05-01:2024-05-31' \
    --decay 0.0008"""

subprocess.run(cmd, shell=True, check=True) 