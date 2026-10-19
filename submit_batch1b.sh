#!/bin/bash
# Batch 1b: both models on the chest leads V1, V2, V3 and bipolar V2-V1 (3 seeds
# each), plus seed 2 of the 12-lead reference. One job per configuration.
# Skips configurations that are already queued or finished, so it is safe to
# run again whenever the submit limit stops it.
# Usage: bash submit_batch1b.sh a40   (or l40)
P=${1:-a40}
cd ~/stage1 && mkdir -p logs
queued=$(squeue -h -u $USER -O "Name:200" | awk '{print $1}')

submit() {  # name lead_tag leads_arg model seeds
  local name=$1 tag=$2 leads=$3 model=$4 seeds=$5 missing=""
  for s in $seeds; do
    [[ -f runs/superclass_${model}_${tag}_500hz_2.5s_seed${s}/results.json ]] || missing+="$s "
  done
  if [[ -z $missing ]]; then echo "done:    $name"; return 0; fi
  if grep -qx "$name" <<< "$queued"; then echo "queued:  $name"; return 0; fi
  sbatch -p "$P" --qos="$P" --job-name="$name" train_seeds.sbatch "$missing" \
         --task superclass --leads "$leads" --model "$model" --win_sec 2.5 --skip_done
}

for model in ${MODELS:-resnet1d_wang resnet1d_wang_s5}; do
  for leads in V1 V2 V3 V2-V1; do
    submit "b1b_${model}_${leads}" "$leads" "$leads" "$model" "0 1 2" \
      || { echo "Submit limit reached. Run this script again later."; squeue -u $USER; exit 0; }
  done
  submit "b1b_${model}_12lead" 12lead all "$model" "2" \
    || { echo "Submit limit reached. Run this script again later."; squeue -u $USER; exit 0; }
done
squeue -u $USER
