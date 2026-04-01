#!/bin/bash

# Ensure DVC is initialized
dvc init 2>/dev/null

echo "Configuring AWS S3 backend..."
# Setup AWS S3 remote storage
dvc remote add -d s3_remote s3://mlops-2022bcs0050-assignment/mlops -f

echo "Creating Dataset Version 1 (500 rows)..."
head -n 501 data/winequality-red.csv > data/dataset_v1.csv
dvc add data/dataset_v1.csv
git add data/dataset_v1.csv.dvc data/.gitignore
git commit -m "Add dataset v1"

echo "Creating Dataset Version 2 (Full dataset)..."
cp data/winequality-red.csv data/dataset_v2.csv
dvc add data/dataset_v2.csv
git add data/dataset_v2.csv.dvc data/.gitignore
git commit -m "Add dataset v2"

echo "Setup Complete!"
echo "Make sure your AWS environment variables are set up, then run: dvc push"
``