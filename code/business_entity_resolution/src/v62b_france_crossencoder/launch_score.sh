# relay v6.2b scoring pairs (train stage-2 slice + v6.2a test top-3) in 7 shards and score them in parallel on AWS
set -e
r() { for i in 1 2 3 4 5 6 7 8; do "$@" && return 0; sleep 60; done; return 1; }  # retry: slow/unstable network
S3=s3://<YOUR_SAGEMAKER_BUCKET>/mlc-v62b; G=gs://mlc-gcp1-v6/v61
TJ=$(cat train_job.txt)
bash prestage.sh || exit 1   # idempotent: skips inputs already in S3
: > score_jobs.txt
for k in 0 1 2 3 4 5 7; do
  JOB=v62b-score-s$k-$(date +%H%M); echo $JOB >> score_jobs.txt
  cat > sjob.json <<J
{"TrainingJobName": "$JOB", "RoleArn": "arn:aws:iam::<AWS_ACCOUNT_ID>:role/service-role/AmazonSageMakerAdminIAMExecutionRole",
 "AlgorithmSpecification": {"TrainingImage": "763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.9.0-gpu-py312-cu130-ubuntu22.04-sagemaker", "TrainingInputMode": "File"},
 "HyperParameters": {"sagemaker_program": "entry.py", "sagemaker_submit_directory": "$S3/code/score_sourcedir.tar.gz", "sagemaker_region": "us-east-1"},
 "InputDataConfig": [{"ChannelName": "pairs", "DataSource": {"S3DataSource": {"S3DataType": "S3Prefix", "S3Uri": "$S3/score/s$k/", "S3DataDistributionType": "FullyReplicated"}}},
                     {"ChannelName": "model", "DataSource": {"S3DataSource": {"S3DataType": "S3Prefix", "S3Uri": "$S3/ckpt/m/", "S3DataDistributionType": "FullyReplicated"}}}],
 "OutputDataConfig": {"S3OutputPath": "$S3/scored/"}, "ResourceConfig": {"InstanceType": "ml.g5.12xlarge", "InstanceCount": 1, "VolumeSizeInGB": 100},
 "EnableManagedSpotTraining": true, "StoppingCondition": {"MaxRuntimeInSeconds": 7200, "MaxWaitTimeInSeconds": 14400}, "Tags": [{"Key": "project", "Value": "mlc-v6"}]}
J
  mk() { MSYS_NO_PATHCONV=1 aws sagemaker create-training-job --region us-east-1 --cli-input-json file://sjob.json --query TrainingJobArn --output text || MSYS_NO_PATHCONV=1 aws sagemaker describe-training-job --region us-east-1 --training-job-name $JOB --query TrainingJobArn --output text; }; r mk
done
