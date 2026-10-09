# laptop watcher: launches the v6.2b AWS jobs as soon as their inputs exist (no dependence on scheduled checks)
G=gs://mlc-gcp1-v6/v61; LOG=aws_watch.log
log() { echo "$(date +%H:%M:%S) $*" >> $LOG; }
log "watcher (resume) started"
S3T=s3://<YOUR_SAGEMAKER_BUCKET>/mlc-v62b/train/ce_groups.parquet
until MSYS_NO_PATHCONV=1 aws s3 ls $S3T >/dev/null 2>&1; do sleep 60; done
JOB=v62b-ce-train-$(date +%H%M); echo $JOB > train_job.txt; sed -i "s/v62b-ce-train-[0-9]*/$JOB/" tjob.json
log "groups in S3 -> create training job $JOB"
ok=0; for i in 1 2 3 4 5; do MSYS_NO_PATHCONV=1 aws sagemaker create-training-job --region us-east-1 --cli-input-json file://tjob.json --query TrainingJobArn --output text >> $LOG 2>&1 && { ok=1; break; }; sleep 30; done
[ $ok = 1 ] || { log "LAUNCH_TRAIN_FAILED"; exit 1; }
TJ=$(cat train_job.txt)
while :; do
  st=$(MSYS_NO_PATHCONV=1 aws sagemaker describe-training-job --region us-east-1 --training-job-name $TJ --query TrainingJobStatus --output text 2>/dev/null)
  [ "$st" = Completed ] && break
  [ "$st" = Failed ] || [ "$st" = Stopped ] && { log "TRAIN_$st"; exit 1; }
  sleep 120
done
log "training completed"
until gcloud storage ls $G/v62b_pairs_done.txt >/dev/null 2>&1; do sleep 60; done
log "pairs ready -> launch scoring"; bash launch_score.sh >> $LOG 2>&1 || { log "LAUNCH_SCORE_FAILED"; exit 1; }
S3=s3://<YOUR_SAGEMAKER_BUCKET>/mlc-v62b
while :; do
  n=0; bad=0
  for j in $(cat score_jobs.txt); do
    st=$(MSYS_NO_PATHCONV=1 aws sagemaker describe-training-job --region us-east-1 --training-job-name $j --query TrainingJobStatus --output text 2>/dev/null)
    [ "$st" = Completed ] && n=$((n+1)); [ "$st" = Failed ] && bad=1
  done
  [ $bad = 1 ] && { log "SCORE_FAILED"; exit 1; }
  [ $n = 7 ] && break; sleep 120
done
log "all 7 scoring jobs completed -> collect"
mkdir -p scored; : > scored/list.txt
for j in $(cat score_jobs.txt); do
  MSYS_NO_PATHCONV=1 aws s3 cp $S3/scored/$j/output/model.tar.gz scored/$j.tar.gz --only-show-errors && tar -xzf scored/$j.tar.gz -C scored && for f in train_ce test_ce; do [ -f scored/$f.parquet ] && mv scored/$f.parquet scored/${j}_$f.parquet; done
done
ls scored/*_ce.parquet >> $LOG
for f in scored/*_ce.parquet; do gcloud storage cp $f $G/v62b/scored/ -q; done
echo done > scored/done.txt && gcloud storage cp scored/done.txt $G/v62b_scored_done.txt -q
log "AWS_WATCH_DONE"
