# copy the v6.2b scoring inputs GCS -> laptop -> S3 ahead of time; skips anything already in S3 at full size; retries
export AWS_DEFAULT_REGION=us-east-1
S3=s3://<YOUR_SAGEMAKER_BUCKET>/mlc-v62b; G=gs://mlc-gcp1-v6/v61
# one copier at a time: wait while another live prestage holds the lock (stale locks are taken over)
until mkdir prestage.lock 2>/dev/null; do
  p=$(cat prestage.lock/pid 2>/dev/null); [ -n "$p" ] && kill -0 "$p" 2>/dev/null || { rm -rf prestage.lock; continue; }
  sleep 30
done
echo $$ > prestage.lock/pid; trap 'rm -rf prestage.lock' EXIT
r() { for i in $(seq 1 15); do "$@" && return 0; sleep 60; done; return 1; }
one() {  # $1 = GCS name, $2 = S3 key
  want=$(gcloud storage ls -l $G/v62b/$1 2>/dev/null | awk 'NR==1{print $1}')
  have=$(MSYS_NO_PATHCONV=1 aws s3 ls $S3/$2 2>/dev/null | awk '{print $3}')
  [ -n "$want" ] && [ "$want" = "$have" ] && { echo "$(date +%H:%M) skip $1 (in S3)" >> prestage.log; return 0; }
  r gcloud storage cp $G/v62b/$1 ./$1 -q && r env MSYS_NO_PATHCONV=1 aws s3 cp ./$1 $S3/$2 --only-show-errors \
    && echo "$(date +%H:%M) ok $1" >> prestage.log || { echo "$(date +%H:%M) FAILED $1" >> prestage.log; return 1; }
}
one ce_score_train.parquet score/s7/ce_score_train.parquet
for k in 0 1 2 3 4 5; do one shard_test_$k.parquet score/s$k/ce_score_test.parquet; done
echo "$(date +%H:%M) PRESTAGE_DONE" >> prestage.log
