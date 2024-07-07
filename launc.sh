python -m src.train \
    --model=ESheafGCN_CF \
    --dataset=MOVIELENS_CF \
    --epochs=50 \
    --device="cuda"


python -m src.train \
    --model=ESheafGCN \
    --dataset=MOVIELENS \
    --epochs=50 \
    --device="cuda"