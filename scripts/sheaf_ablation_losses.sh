LATENT_DIMS=40
DATASET=FACEBOOK
DEVICE=cuda
EPOCHS=60
MODEL=ExtendableSheafGCN
COMPOSITION=add


ARTIFACT_DIR="./ABLATION_${MODEL}_${DATASET}_${COMPOSITION}_${EPOCHS}_$(date +%s)"

for SEED in {1..8}; do
  echo "Compute for seed ${SEED}"
  python -m src.train --model ExtendableSheafGCN --seed $SEED --params "{'grad_clip':0.5, 'latent_dim':$LATENT_DIMS, 'operator_train_mode': 'sim', 'losses':['bpr', 'diff']}" --dataset $DATASET --device $DEVICE --epochs $EPOCHS --artifact-dir $ARTIFACT_DIR
  python -m src.train --model ExtendableSheafGCN --seed $SEED --params "{'grad_clip':0.5, 'latent_dim':$LATENT_DIMS, 'operator_train_mode': 'sim', 'losses':['bpr', 'diff', 'orth']}" --dataset $DATASET --device $DEVICE --epochs $EPOCHS --artifact-dir $ARTIFACT_DIR
  python -m src.train --model ExtendableSheafGCN --seed $SEED --params "{'grad_clip':0.5, 'latent_dim':$LATENT_DIMS, 'operator_train_mode': 'sim', 'losses':['bpr', 'diff', 'cons']}" --dataset $DATASET --device $DEVICE --epochs $EPOCHS --artifact-dir $ARTIFACT_DIR
  python -m src.train --model ExtendableSheafGCN --seed $SEED --params "{'grad_clip':0.5, 'latent_dim':$LATENT_DIMS, 'operator_train_mode': 'sim', 'losses':['bpr', 'diff', 'orth','cons']}" --dataset $DATASET --device $DEVICE --epochs $EPOCHS --artifact-dir $ARTIFACT_DIR
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
