now=$(date +"%Y%m%d_%H%M%S")
JOB_NAME="first_try_$now"
OUTPUT_PATH=gs://aml-project3/output/
INPUT_PATH=gs://aml-project3/data/
gcloud ml-engine jobs submit training $JOB_NAME --package-path train --module-name train.training --staging-bucket gs://aml-project3 --job-dir gs://aml-project3/output --region us-east1 --config config.yaml --runtime-version=1.6 -- --data_dir="${INPUT_PATH}" --output_dir="${OUTPUT_PATH}"