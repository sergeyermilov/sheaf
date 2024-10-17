#!/bin/bash

ARTIFACT_DIR="./artifact"
ARTIFACT_ID=$1
CHECKPOINTS=$2
for ENTRY in "$ARTIFACT_DIR/$ARTIFACT_ID/$CHECKPOINTS/"*
do
  echo $ENTRY
  filename="$(basename "$ENTRY")"
  echo $filename
  python -m src.evaluate --artifact-dir $ARTIFACT_DIR --artifact-id $ARTIFACT_ID --device cpu --model-name "checkpoints/$filename"
done