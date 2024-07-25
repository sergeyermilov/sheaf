#!/bin/bash

DEVICE=$1
DATASET=$2
EPOCHS=$3

ARTIFACT_DIR="./SHEAF_${DATASET}_${EPOCHS}"

python -m src.train --model ESheafGCN --dataset $DATASET --device $DEVICE --epochs $EPOCHS --artifact_dir $ARTIFACT_DIR
python -m src.train --model ExtendableSheafGCN --params "{'latent_dim':30,'layer_types':['global']}" --dataset $DATASET --device $DEVICE --epochs $EPOCHS --artifact_dir $ARTIFACT_DIR
python -m src.train --model ExtendableSheafGCN --params "{'latent_dim':30,'layer_types':['single']}" --dataset $DATASET --device $DEVICE --epochs $EPOCHS --artifact_dir $ARTIFACT_DIR
python -m src.train --model ExtendableSheafGCN --params "{'latent_dim':30,'layer_types':['paired']}" --dataset $DATASET --device $DEVICE --epochs $EPOCHS --artifact_dir $ARTIFACT_DIR
python -m src.train --model ExtendableSheafGCN --params "{'latent_dim':30,'layer_types':['global,single']}" --dataset $DATASET --device $DEVICE --epochs $EPOCHS --artifact_dir $ARTIFACT_DIR
python -m src.train --model ExtendableSheafGCN --params "{'latent_dim':30,'layer_types':['global,paired']}" --dataset $DATASET --device $DEVICE --epochs $EPOCHS --artifact_dir $ARTIFACT_DIR
python -m src.train --model ExtendableSheafGCN --params "{'latent_dim':30,'layer_types':['single,paired']}" --dataset $DATASET --device $DEVICE --epochs $EPOCHS --artifact_dir $ARTIFACT_DIR
python -m src.train --model ExtendableSheafGCN --params "{'latent_dim':30,'layer_types':['global,single,paired']}" --dataset $DATASET --device $DEVICE --epochs $EPOCHS --artifact_dir $ARTIFACT_DIR


for ENTRY in ./*
do
  ARTIFACT_ID=$(echo $ENTRY | sed 's/^.\{2\}//')
  python -m src.evaluate --artifact_dir $ARTIFACT_DIR --artifact_id $ARTIFACT_ID --device $DEVICE
done