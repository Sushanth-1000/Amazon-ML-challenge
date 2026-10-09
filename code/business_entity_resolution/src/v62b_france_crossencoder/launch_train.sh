# relay v6.2b groups GCS -> S3 and start the spot training job (E20 recipe)
set -e
S3=s3://<YOUR_SAGEMAKER_BUCKET>/mlc-v62b; G=gs://mlc-gcp1-v6/v61
gcloud storage cp $G/v62b/ce_groups.parquet ce_groups.parquet -q
MSYS_NO_PATHCONV=1 aws s3 cp ce_groups.parquet $S3/train/ce_groups.parquet --only-show-errors
JOB=v62b-ce-train-$(date +%H%M); echo $JOB > train_job.txt
cat > tjob.json <<J
{"TrainingJobName": "$JOB", "RoleArn": "arn:aws:iam::<AWS_ACCOUNT_ID>:role/service-role/AmazonSageMakerAdminIAMExecutionRole",
 "AlgorithmSpecification": {"TrainingImage": "763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.9.0-gpu-py312-cu130-ubuntu22.04-sagemaker", "TrainingInputMode": "File"},
 "HyperParameters": {"sagemaker_program": "entry.py", "sagemaker_submit_directory": "$S3/code/train_sourcedir.tar.gz", "sagemaker_region": "us-east-1"},
 "InputDataConfig": [{"ChannelName": "train", "DataSource": {"S3DataSource": {"S3DataType": "S3Prefix", "S3Uri": "$S3/train/", "S3DataDistributionType": "FullyReplicated"}}}],
 "OutputDataConfig": {"S3OutputPath": "$S3/output/"}, "CheckpointConfig": {"S3Uri": "$S3/ckpt/", "LocalPath": "/opt/ml/checkpoints"},
 "ResourceConfig": {"InstanceType": "ml.g5.12xlarge", "InstanceCount": 1, "VolumeSizeInGB": 100}, "EnableManagedSpotTraining": true,
 "StoppingCondition": {"MaxRuntimeInSeconds": 14400, "MaxWaitTimeInSeconds": 21600}, "Tags": [{"Key": "project", "Value": "mlc-v6"}]}
J
MSYS_NO_PATHCONV=1 aws sagemaker create-training-job --region us-east-1 --cli-input-json file://tjob.json --query TrainingJobArn --output text
