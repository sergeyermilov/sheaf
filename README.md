## Instalation
Depending on the system you have, you can either use
```pip install -r requirements_cpu.txt``` or ```pip install -r requirements.txt```.

If you decided to install CPU-only version, usage of --device = cuda property would be prohibited.

## Example of usage

Train model ExtendableSheafGCN on small FACEBOOK dataset
```CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=1 python -m src.train --model ExtendableSheafGCN --model-params "{'latent_dim':30,'layer_types':['homo_global']}" --dataset-params "{'batch_size': 512}" --epochs 1 --dataset FACEBOOK --denoise --device cpu```
Train model ExtendableSheafGCN on medium YAHOO dataset
```CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=1 python -m src.train --model ExtendableSheafGCN --model-params "{'latent_dim':30,'layer_types':['homo_global']}" --dataset-params "{'batch_size': 64, 'enable_subsampling':true,'num_k_hops':3,'hop_max_edges':[256,1024,4096]}" --epochs 1 --dataset YAHOO --denoise --device cpu```
Evaluate model ExtendableSheafGCN on medium YAHOO dataset
```CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=1 python -m src.evaluate --artifact-id c8098d8480aa --device cpu```
