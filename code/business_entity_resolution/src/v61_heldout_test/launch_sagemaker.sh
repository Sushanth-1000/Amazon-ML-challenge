export AWS_DEFAULT_REGION=us-east-1
B=s3://<YOUR_SAGEMAKER_BUCKET>
aws_() { MSYS_NO_PATHCONV=1 aws "$@"; }
for i in 1 2 3; do aws_ s3 cp sourcedir.tar.gz $B/mlc-cmp/code/sourcedir.tar.gz --only-show-errors && break; sleep 20; done
JOB=cmp-v61-v4-$(date +%H%M); echo $JOB > job.txt
cat > job.json <<J
{"TrainingJobName": "$JOB", "RoleArn": "arn:aws:iam::<AWS_ACCOUNT_ID>:role/service-role/AmazonSageMakerAdminIAMExecutionRole",
 "AlgorithmSpecification": {"TrainingImage": "763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.9.0-cpu-py312-ubuntu22.04-sagemaker", "TrainingInputMode": "File"},
 "HyperParameters": {"sagemaker_program": "entry.py", "sagemaker_submit_directory": "$B/mlc-cmp/code/sourcedir.tar.gz", "sagemaker_region": "us-east-1"},
 "InputDataConfig": [
  {"ChannelName": "data", "DataSource": {"S3DataSource": {"S3DataType": "S3Prefix", "S3Uri": "$B/mlc-v6data/data/", "S3DataDistributionType": "FullyReplicated"}}},
  {"ChannelName": "v61", "DataSource": {"S3DataSource": {"S3DataType": "S3Prefix", "S3Uri": "$B/mlc-cmp/v61/", "S3DataDistributionType": "FullyReplicated"}}}],
 "OutputDataConfig": {"S3OutputPath": "$B/mlc-cmp/output/"},
 "ResourceConfig": {"InstanceType": "ml.r6i.12xlarge", "InstanceCount": 1, "VolumeSizeInGB": 100},
 "StoppingCondition": {"MaxRuntimeInSeconds": 7200}, "Tags": [{"Key": "project", "Value": "mlc-v6"}]}
J
for i in 1 2 3 4 5; do aws_ sagemaker create-training-job --cli-input-json file://job.json --query TrainingJobArn --output text && { echo LAUNCHED $JOB; exit 0; }; sleep 30; done
echo LAUNCH_FAILED
