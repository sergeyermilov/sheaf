## Instalation
Depending on the system you have, you can either use
```pip install -r requirements_cpu.txt``` or ```pip install -r requirements.txt```.

If you decided to install CPU-only version, usage of --device = cuda property would be prohibited.

## Example of usage

```CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=1 python -m src.train --model ESheafGCN --dataset FACEBOOK --epochs 5```

```CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=1 python -m src.evaluate --model ESheafGCN --dataset FACEBOOK --epochs 5```
