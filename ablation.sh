LATENT_DIMS=40
DATASET=FACEBOOK
DEVICE=cuda
EPOCHS=4
MODEL=ExtendableSheafGCN


ARTIFACT_DIR="./ABLATION_${MODEL}_${DATASET}_${EPOCHS}"

#for SEED in {1..10}; do
#  echo "Compute for seed ${SEED}"
#  python -m src.train --model ExtendableSheafGCN --seed $SEED --params "{'latent_dim':$LATENT_DIMS, 'losses':[]}" --dataset $DATASET --device $DEVICE --epochs $EPOCHS --artifact-dir $ARTIFACT_DIR
#  python -m src.train --model ExtendableSheafGCN --seed $SEED --params "{'latent_dim':$LATENT_DIMS, 'losses':['orth']}" --dataset $DATASET --device $DEVICE --epochs $EPOCHS --artifact-dir $ARTIFACT_DIR
#  python -m src.train --model ExtendableSheafGCN --seed $SEED --params "{'latent_dim':$LATENT_DIMS, 'losses':['cons']}" --dataset $DATASET --device $DEVICE --epochs $EPOCHS --artifact-dir $ARTIFACT_DIR
#  python -m src.train --model ExtendableSheafGCN --seed $SEED --params "{'latent_dim':$LATENT_DIMS, 'losses':['orth','cons']}" --dataset $DATASET --device $DEVICE --epochs $EPOCHS --artifact-dir $ARTIFACT_DIR
#done

for SEED in {1..10}; do
  echo "Compute for seed ${SEED}"
  python -m src.train --model $MODEL --seed $SEED --params "{'latent_dim':$LATENT_DIMS,'composition_type': 'mult', 'losses':['orth','cons'], 'layer_types':['global']}" --dataset $DATASET --device $DEVICE --epochs $EPOCHS --artifact-dir $ARTIFACT_DIR
  python -m src.train --model $MODEL --seed $SEED --params "{'latent_dim':$LATENT_DIMS,'composition_type': 'mult', 'losses':['orth','cons'], 'layer_types':['single']}" --dataset $DATASET --device $DEVICE --epochs $EPOCHS --artifact-dir $ARTIFACT_DIR
  python -m src.train --model $MODEL --seed $SEED --params "{'latent_dim':$LATENT_DIMS,'composition_type': 'mult', 'losses':['orth','cons'], 'layer_types':['paired']}" --dataset $DATASET --device $DEVICE --epochs $EPOCHS --artifact-dir $ARTIFACT_DIR
  python -m src.train --model $MODEL --seed $SEED --params "{'latent_dim':$LATENT_DIMS,'composition_type': 'mult', 'losses':['orth','cons'], 'layer_types':['global','single']}" --dataset $DATASET --device $DEVICE --epochs $EPOCHS --artifact-dir $ARTIFACT_DIR
  python -m src.train --model $MODEL --seed $SEED --params "{'latent_dim':$LATENT_DIMS,'composition_type': 'mult', 'losses':['orth','cons'], 'layer_types':['global','paired']}" --dataset $DATASET --device $DEVICE --epochs $EPOCHS --artifact-dir $ARTIFACT_DIR
  python -m src.train --model $MODEL --seed $SEED --params "{'latent_dim':$LATENT_DIMS,'composition_type': 'mult', 'losses':['orth','cons'], 'layer_types':['single','paired']}" --dataset $DATASET --device $DEVICE --epochs $EPOCHS --artifact-dir $ARTIFACT_DIR
  python -m src.train --model $MODEL --seed $SEED --params "{'latent_dim':$LATENT_DIMS,'composition_type': 'mult', 'losses':['orth','cons'], 'layer_types':['global','single','paired']}" --dataset $DATASET --device $DEVICE --epochs $EPOCHS --artifact-dir $ARTIFACT_DIR
done

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
