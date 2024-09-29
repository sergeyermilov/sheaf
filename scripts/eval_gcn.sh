#!/bin/bash

#CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 bash scripts/sheaf_eval_gcn.sh cuda FACEBOOK 40 40 1024 8
#CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 bash scripts/sheaf_eval_gcn.sh cuda YAHOO 30 40 16000 6 3 "[16000,64000,128000]"
#CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 bash scripts/sheaf_eval_gcn.sh cuda MOVIELENS1M 30 40 10000 6 3 "[10000,40000,80000]"

DEVICE=$1
DATASET=$2
EPOCHS=$3
LATENT_DIMS=$4
BATCH_SIZE=$5
SAMPLES=$6

NUM_HOPS=$7 # 2
HOP_MAX_EDGES=$8 #[$BATCH_SIZE,$BATCH_SIZE*4,$BATCH_SIZE*8]

ENABLE_SUBSAMPLING_CMD=""

[[ $NUM_HOPS && $HOP_MAX_EDGES ]] && ENABLE_SUBSAMPLING_CMD=",'enable_subsampling':true,'num_k_hops':$NUM_HOPS,'hop_max_edges':$HOP_MAX_EDGES"

echo "DEVICE = $DEVICE"
echo "DATASET = $DATASET"
echo "EPOCHS = $EPOCHS"
echo "LATENT_DIMS = $LATENT_DIMS"
echo "BATCH_SIZE = $BATCH_SIZE"

ARTIFACT_DIR="./GCN_${DATASET}_${EPOCHS}_$(date +%s)"

#LAYER_TYPES=("['hetero_global']" "['homo_global']" "['homo_simple_ffn']" "['hetero_simple_ffn']" "['hetero_global','hetero_simple_ffn']" "['homo_global','homo_simple_ffn']")
# LAYER_TYPES=("['hetero_global']" "['hetero_simple_ffn']" "['hetero_global','hetero_simple_ffn']")

for SEED in $(seq 1 $SAMPLES); do
#  python -m src.train --model TopKPopularity --dataset $DATASET --device $DEVICE --epochs $EPOCHS --artifact-dir $ARTIFACT_DIR
#  python -m src.train --model EASE --model-params "{'lambda_reg':250}" --dataset $DATASET --device $DEVICE --epochs $EPOCHS --artifact-dir $ARTIFACT_DIR
  python -m src.train --model GAT --model-params "{'latent_dim':$LATENT_DIMS}" --dataset-params "{'random_state':$SEED, 'batch_size':$BATCH_SIZE $ENABLE_SUBSAMPLING_CMD}" --dataset $DATASET --device $DEVICE --epochs $EPOCHS --artifact-dir $ARTIFACT_DIR
  python -m src.train --model SheafGCN --model-params "{'latent_dim':$LATENT_DIMS}" --dataset-params "{'random_state':$SEED, 'batch_size':$BATCH_SIZE $ENABLE_SUBSAMPLING_CMD}" --dataset $DATASET --device $DEVICE --epochs $EPOCHS --artifact-dir $ARTIFACT_DIR
  python -m src.train --model LightGCN --model-params "{'latent_dim':$LATENT_DIMS}" --dataset-params "{'random_state':$SEED, 'batch_size':$BATCH_SIZE $ENABLE_SUBSAMPLING_CMD}" --dataset $DATASET --device $DEVICE --epochs $EPOCHS --artifact-dir $ARTIFACT_DIR
done;


if [ ! -d "$ARTIFACT_DIR" ]; then
  echo "Artifact directory $ARTIFACT_DIR does not exists!"
  exit 1
fi

for ENTRY in "$ARTIFACT_DIR/"*
do
  TRIM_LENGTH=$[${#ARTIFACT_DIR}+1]
  ARTIFACT_ID=$(echo $ENTRY | sed "s/^.\{${TRIM_LENGTH}\}//")

  python -m src.evaluate --artifact-dir $ARTIFACT_DIR --artifact-id $ARTIFACT_ID --device $DEVICE
done