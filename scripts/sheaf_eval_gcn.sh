#!/bin/bash

DEVICE=$1
DATASET=$2
EPOCHS=$3
LATENT_DIMS=$4
BATCH_SIZE=$5
DEPTH=$6
SAMPLES=$7

NUM_HOPS=$8 # 2
HOP_MAX_EDGES=$9 #[$BATCH_SIZE,$BATCH_SIZE*4,$BATCH_SIZE*8]

ENABLE_SUBSAMPLING_CMD=""

[[ $NUM_HOPS && $HOP_MAX_EDGES ]] && ENABLE_SUBSAMPLING_CMD=",'enable_subsampling':true,'num_k_hops':$NUM_HOPS,'hop_max_edges':$HOP_MAX_EDGES"

echo "DEVICE = $DEVICE"
echo "DATASET = $DATASET"
echo "EPOCHS = $EPOCHS"
echo "LATENT_DIMS = $LATENT_DIMS"
echo "BATCH_SIZE = $BATCH_SIZE"
echo "DEPTH = $DEPTH"
echo "SAMPLES = $SAMPLES"

ARTIFACT_DIR="./SHEAF_${DATASET}_${EPOCHS}_$(date +%s)"

#LAYER_TYPES=("['hetero_global']" "['homo_global']" "['homo_simple_ffn']" "['hetero_simple_ffn']" "['hetero_global','hetero_simple_ffn']" "['homo_global','homo_simple_ffn']")
LAYER_TYPES=("['hetero_simple_ffn']" "['hetero_global']" "['hetero_global','hetero_simple_ffn']")
#LAYER_TYPES=("['hetero_global']")

for SEED in $(seq 1 $SAMPLES); do
  for LAYER in "${LAYER_TYPES[@]}"; do
    python -m src.train \
      --model ExtendableSheafGCN \
      --dataset-params "{'random_state':$SEED, 'batch_size':$BATCH_SIZE $ENABLE_SUBSAMPLING_CMD}" \
      --model-params "{'latent_dim':$LATENT_DIMS,'operator_ffn_depth':$DEPTH,'layer_types':$LAYER,'epochs_per_operator':20}" \
      --dataset $DATASET \
      --device $DEVICE \
      --epochs $EPOCHS \
      --checkpoint \
      --monitor-lr \
      --artifact-dir $ARTIFACT_DIR
  done;
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