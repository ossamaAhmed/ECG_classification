now=$(date +"%Y%m%d_%H%M%S")
JOB_NAME="CNN__deep_1D_TF_$now"
OUTPUT_PATH=gs://aml-project3/output_deep/
INPUT_PATH=gs://aml-project3/data/
gcloud ml-engine jobs submit training $JOB_NAME --package-path train --module-name train.training_deep --staging-bucket gs://aml-project3 --job-dir gs://aml-project3/output_deep --region us-east1 --config config.yaml --runtime-version=1.6 -- --data_dir="${INPUT_PATH}" --output_dir="${OUTPUT_PATH}"