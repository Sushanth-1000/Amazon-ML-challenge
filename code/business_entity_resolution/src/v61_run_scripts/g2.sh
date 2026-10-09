# v6.1 GPU step 2: cosines for train pairs, French-style copy pairs, test pairs (as soon as each pair list exists)
mkdir -p ~/v61cos
until [ -f ~/g1_done ]; do sleep 30; done
gcloud storage cp gs://mlc-gcp1-v6/code/v61/v61_gpu_cos.py ~/v61/ -q
for w in train copies test; do
  f=$([ $w = copies ] && echo train_copyqs.npy || echo ${w}_qs.npy)
  until gcloud storage ls gs://mlc-gcp1-v6/v61/$f >/dev/null 2>&1; do sleep 30; done
  gcloud storage cp gs://mlc-gcp1-v6/v61/$f ~/v61cos/ -q
  python3 ~/v61/v61_gpu_cos.py --emb ~/v61emb --work ~/v61cos --what $w >> ~/v61_g2.log 2>&1 || { echo "G2 $w FAILED" >> ~/v61_g2.log; exit 1; }
  o=$([ $w = copies ] && echo train_cos_fr.npy || echo ${w}_cos.npy)
  gcloud storage cp ~/v61cos/$o gs://mlc-gcp1-v6/v61/ -q && echo "pushed $o" >> ~/v61_g2.log
done
echo G2_DONE >> ~/v61_g2.log
