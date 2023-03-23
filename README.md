# RoTP-CLIP

## How to run?

```
python -u experiments/clip/TP_training.py \
--cnt-prompt 5 \
--lamb 10.0 \
--dataset cifar10 \
| tee logs/10.0_5.log
```

```
python -u experiments/clip/TP_standard_training.py \
--cnt-prompt 5 \
--dataset cifar10 \
| tee logs/standard_training_5.log
```