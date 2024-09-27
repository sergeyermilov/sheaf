#!/bin/bash

DEVICE=$1
DATASET=$2
EPOCHS=$3
LATENT_DIMS=$4
BATCH_SIZE=$5

echo "DEVICE = $DEVICE"
echo "DATASET = $DATASET"
echo "EPOCHS = $EPOCHS"
echo "LATENT_DIMS = $LATENT_DIMS"
echo "BATCH_SIZE = $BATCH_SIZE"

ARTIFACT_DIR="./SHEAF_${DATASET}_${EPOCHS}_$(date +%s)"

#python -m src.train --model TopKPopularity --dataset $DATASET --device $DEVICE --epochs $EPOCHS --artifact-dir $ARTIFACT_DIR
#python -m src.train --model EASE --params "{'lambda_reg':250}" --dataset $DATASET --device $DEVICE --epochs $EPOCHS --artifact-dir $ARTIFACT_DIR
#python -m src.train --model GAT --params "{'latent_dim':$LATENT_DIMS}" --dataset $DATASET --device $DEVICE --epochs $EPOCHS --artifact-dir $ARTIFACT_DIR
#python -m src.train --model SheafGCN --params "{'latent_dim':$LATENT_DIMS}" --dataset $DATASET --device $DEVICE --epochs $EPOCHS --artifact-dir $ARTIFACT_DIR
#python -m src.train --model LightGCN --params "{'latent_dim':$LATENT_DIMS}" --dataset $DATASET --device $DEVICE --epochs $EPOCHS --artifact-dir $ARTIFACT_DIR
#python -m src.train --model ESheafGCN --params "{'latent_dim':$LATENT_DIMS}" --dataset $DATASET --device $DEVICE --epochs $EPOCHS --artifact-dir $ARTIFACT_DIR

LAYER_TYPES=("['hetero_global']", "['homo_global']", "['homo_simple_ffn']", "['hetero_simple_ffn']", "['hetero_global','hetero_simple_ffn']", "['homo_global','homo_simple_ffn']")

for SEED in {1..8}; do
  for LAYER in LAYER_TYPES; do
    python -m src.train \
      --model ExtendableSheafGCN \
      --dataset-params "{'random_state':$SEED, 'batch_size':$BATCH_SIZE}" \
      --model-params "{'latent_dim':$LATENT_DIMS,'layer_types':$LAYER}" \
      --dataset $DATASET \
      --device $DEVICE \
      --epochs $EPOCHS \
      --artifact-dir $ARTIFACT_DIR
  done;
done;

#python -m src.train --batch-size $BATCH_SIZE --model ExtendableSheafGCN --params "{'sample_share':$SAMPLE_SHARE, 'latent_dim':$LATENT_DIMS,'layer_types':['global']}" --dataset $DATASET --device $DEVICE --epochs $EPOCHS --artifact-dir $ARTIFACT_DIR
#python -m src.train --batch-size $BATCH_SIZE --model ExtendableSheafGCN --params "{'sample_share':$SAMPLE_SHARE, 'latent_dim':$LATENT_DIMS,'layer_types':['single']}" --dataset $DATASET --device $DEVICE --epochs $EPOCHS --artifact-dir $ARTIFACT_DIR
#python -m src.train --batch-size $BATCH_SIZE --model ExtendableSheafGCN --params "{'sample_share':$SAMPLE_SHARE, 'latent_dim':$LATENT_DIMS,'layer_types':['paired']}" --dataset $DATASET --device $DEVICE --epochs $EPOCHS --artifact-dir $ARTIFACT_DIR
#python -m src.train --batch-size $BATCH_SIZE --model ExtendableSheafGCN --params "{'sample_share':$SAMPLE_SHARE, 'latent_dim':$LATENT_DIMS,'layer_types':['global','single']}" --dataset $DATASET --device $DEVICE --epochs $EPOCHS --artifact-dir $ARTIFACT_DIR
#python -m src.train --batch-size $BATCH_SIZE --model ExtendableSheafGCN --params "{'sample_share':$SAMPLE_SHARE, 'latent_dim':$LATENT_DIMS,'layer_types':['global','paired']}" --dataset $DATASET --device $DEVICE --epochs $EPOCHS --artifact-dir $ARTIFACT_DIR
#python -m src.train --batch-size $BATCH_SIZE --model ExtendableSheafGCN --params "{'sample_share':$SAMPLE_SHARE, 'latent_dim':$LATENT_DIMS,'layer_types':['single','paired']}" --dataset $DATASET --device $DEVICE --epochs $EPOCHS --artifact-dir $ARTIFACT_DIR
#python -m src.train --batch-size $BATCH_SIZE --model ExtendableSheafGCN --params "{'sample_share':$SAMPLE_SHARE, 'latent_dim':$LATENT_DIMS,'layer_types':['global','single','paired']}" --dataset $DATASET --device $DEVICE --epochs $EPOCHS --artifact-dir $ARTIFACT_DIR
#
#
#if [ ! -d "$ARTIFACT_DIR" ]; then
#  echo "Artifact directory $ARTIFACT_DIR does not exists!"
#  exit 1
#fi
#
#for ENTRY in "$ARTIFACT_DIR/"*
#do
#  TRIM_LENGTH=$[${#ARTIFACT_DIR}+1]
#  ARTIFACT_ID=$(echo $ENTRY | sed "s/^.\{${TRIM_LENGTH}\}//")
#
#  python -m src.evaluate --artifact-dir $ARTIFACT_DIR --artifact-id $ARTIFACT_ID --device $DEVICE
#done