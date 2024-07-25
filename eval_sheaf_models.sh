#!/bin/bash

DEVICE=$1
DATASET=$2
EPOCHS=$3

echo "DEVICE = $DEVICE"
echo "DATASET = $DATASET"
echo "EPOCHS = $EPOCHS"

ARTIFACT_DIR="./SHEAF_${DATASET}_${EPOCHS}"

python -m src.train --model GAT --params "{'latent_dim':30}" --dataset $DATASET --device $DEVICE --epochs $EPOCHS --artifact-dir $ARTIFACT_DIR
python -m src.train --model LightGCN --params "{'latent_dim':30}" --dataset $DATASET --device $DEVICE --epochs $EPOCHS --artifact-dir $ARTIFACT_DIR
python -m src.train --model ESheafGCN --params "{'latent_dim':30}" --dataset $DATASET --device $DEVICE --epochs $EPOCHS --artifact-dir $ARTIFACT_DIR
python -m src.train --model ExtendableSheafGCN --params "{'latent_dim':30,'layer_types':['global']}" --dataset $DATASET --device $DEVICE --epochs $EPOCHS --artifact-dir $ARTIFACT_DIR
python -m src.train --model ExtendableSheafGCN --params "{'latent_dim':30,'layer_types':['single']}" --dataset $DATASET --device $DEVICE --epochs $EPOCHS --artifact-dir $ARTIFACT_DIR
python -m src.train --model ExtendableSheafGCN --params "{'latent_dim':30,'layer_types':['paired']}" --dataset $DATASET --device $DEVICE --epochs $EPOCHS --artifact-dir $ARTIFACT_DIR
python -m src.train --model ExtendableSheafGCN --params "{'latent_dim':30,'layer_types':['global,single']}" --dataset $DATASET --device $DEVICE --epochs $EPOCHS --artifact-dir $ARTIFACT_DIR
python -m src.train --model ExtendableSheafGCN --params "{'latent_dim':30,'layer_types':['global,paired']}" --dataset $DATASET --device $DEVICE --epochs $EPOCHS --artifact-dir $ARTIFACT_DIR
python -m src.train --model ExtendableSheafGCN --params "{'latent_dim':30,'layer_types':['single,paired']}" --dataset $DATASET --device $DEVICE --epochs $EPOCHS --artifact-dir $ARTIFACT_DIR
python -m src.train --model ExtendableSheafGCN --params "{'latent_dim':30,'layer_types':['global,single,paired']}" --dataset $DATASET --device $DEVICE --epochs $EPOCHS --artifact-dir $ARTIFACT_DIR


[ ! -d "$ARTIFACT_DIR" ] && echo "Artifact directory $ARTIFACT_DIR does not exists!" && exit 1

for ENTRY in "$ARTIFACT_DIR/"*
do
  TRIM_LENGTH=$[${#ARTIFACT_DIR}+1]
  ARTIFACT_ID=$(echo $ENTRY | sed "s/^.\{${TRIM_LENGTH}\}//")

  python -m src.evaluate --artifact-dir $ARTIFACT_DIR --artifact-id $ARTIFACT_ID --device $DEVICE
done