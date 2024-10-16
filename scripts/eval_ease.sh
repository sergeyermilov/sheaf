#!/bin/bash

DEVICE=$1
DATASET=$2
EPOCHS=$3
LATENT_DIMS=$4
BATCH_SIZE=$5
SAMPLES=$6

echo "DEVICE = $DEVICE"
echo "DATASET = $DATASET"
echo "EPOCHS = $EPOCHS"
echo "LATENT_DIMS = $LATENT_DIMS"
echo "BATCH_SIZE = $BATCH_SIZE"

ARTIFACT_DIR="./EASE_${DATASET}_${EPOCHS}_$(date +%s)"

for SEED in $(seq 1 $SAMPLES); do
  python -m src.train --model EASE --model-params "{'lambda_reg':250}" --dataset-params "{'random_state':$SEED, 'batch_size':$BATCH_SIZE}" --dataset $DATASET --device $DEVICE --epochs $EPOCHS --artifact-dir $ARTIFACT_DIR
done ;


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