# EC estimation

A simple mOU pipeline for effective connectivity estimation. The notebook selects tau by maximizing the off-diagonal correlation between EC and empirical lag-one covariance (Q1), and compares EC estimates across tau values. Local pyMOU source and its license are included.

Open `EC_pipeline_Q1.ipynb`, set the input path and subject identifiers, and run the cells. Install dependencies from `requirements.txt`. Results are saved locally and excluded from Git.
