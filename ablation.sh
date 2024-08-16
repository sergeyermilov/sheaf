LATENT_DIMS=40
DATASET=FACEBOOK
DEVICE=cuda
EPOCHS=120
MODEL=ExtendableSheafGCN
COMPOSITION=add


ARTIFACT_DIR="./ABLATION_${MODEL}_${DATASET}_${COMPOSITION}_${EPOCHS}_$(date +%s)"
#ARTIFACT_DIR="./ABLATION_ExtendableSheafGCN_FACEBOOK_add_60_1722880731"

for SEED in {1..8}; do
  echo "Compute for seed ${SEED}"
  python -m src.train --model ExtendableSheafGCN --seed $SEED --params "{'grad_clip':0.5, 'latent_dim':$LATENT_DIMS, 'losses':['bpr', 'diff']}" --dataset $DATASET --device $DEVICE --epochs $EPOCHS --artifact-dir $ARTIFACT_DIR
  python -m src.train --model ExtendableSheafGCN --seed $SEED --params "{'grad_clip':0.5, 'latent_dim':$LATENT_DIMS, 'losses':['bpr', 'diff', 'orth']}" --dataset $DATASET --device $DEVICE --epochs $EPOCHS --artifact-dir $ARTIFACT_DIR
  python -m src.train --model ExtendableSheafGCN --seed $SEED --params "{'grad_clip':0.5, 'latent_dim':$LATENT_DIMS, 'losses':['bpr', 'diff', 'cons']}" --dataset $DATASET --device $DEVICE --epochs $EPOCHS --artifact-dir $ARTIFACT_DIR
  python -m src.train --model ExtendableSheafGCN --seed $SEED --params "{'grad_clip':0.5, 'latent_dim':$LATENT_DIMS, 'losses':['bpr', 'diff', 'orth','cons']}" --dataset $DATASET --device $DEVICE --epochs $EPOCHS --artifact-dir $ARTIFACT_DIR
done

#for SEED in {1..8}; do
#  echo "Compute for seed ${SEED}"
#  #python -m src.train --model $MODEL --seed $SEED --params "{'operator_ffn_depth': 3, 'latent_dim':$LATENT_DIMS,'composition_type': '$COMPOSITION', 'losses':['orth','cons'], 'layer_types':['single_distinct']}" --dataset $DATASET --device $DEVICE --epochs $EPOCHS --artifact-dir $ARTIFACT_DIR
#  #python -m src.train --model $MODEL --seed $SEED --params "{'latent_dim':$LATENT_DIMS,'composition_type': '$COMPOSITION', 'losses':['orth','cons'], 'layer_types':['global']}" --dataset $DATASET --device $DEVICE --epochs $EPOCHS --artifact-dir $ARTIFACT_DIR
#  python -m src.train --model $MODEL --seed $SEED --params "{'latent_dim':$LATENT_DIMS,'composition_type': '$COMPOSITION', 'losses':['orth','cons'], 'layer_types':['paired']}" --dataset $DATASET --device $DEVICE --epochs $EPOCHS --artifact-dir $ARTIFACT_DIR
#  python -m src.train --model $MODEL --seed $SEED --params "{'operator_ffn_depth': 3, 'operator_train_mode': 'cons', 'latent_dim':$LATENT_DIMS, 'composition_type': '$COMPOSITION', 'losses':['orth','cons'], 'layer_types':['global','single_distinct']}" --dataset $DATASET --device $DEVICE --epochs $EPOCHS --artifact-dir $ARTIFACT_DIR
#  python -m src.train --model $MODEL --seed $SEED --params "{'latent_dim':$LATENT_DIMS,'composition_type': '$COMPOSITION', 'losses':['orth','cons'], 'layer_types':['global','paired']}" --dataset $DATASET --device $DEVICE --epochs $EPOCHS --artifact-dir $ARTIFACT_DIR
#  python -m src.train --model $MODEL --seed $SEED --params "{'latent_dim':$LATENT_DIMS,'composition_type': '$COMPOSITION', 'losses':['orth','cons'], 'layer_types':['single','paired']}" --dataset $DATASET --device $DEVICE --epochs $EPOCHS --artifact-dir $ARTIFACT_DIR
#  python -m src.train --model $MODEL --seed $SEED --params "{'latent_dim':$LATENT_DIMS,'composition_type': '$COMPOSITION', 'losses':['orth','cons'], 'layer_types':['global','single','paired']}" --dataset $DATASET --device $DEVICE --epochs $EPOCHS --artifact-dir $ARTIFACT_DIR
#done

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
