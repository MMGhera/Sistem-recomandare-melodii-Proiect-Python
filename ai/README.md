# Train

## Models
- `big_sample_rate`: 22500 sample rate (the best)
- `small_sample_rate`: 12000 sample rate

## Test model

```sh
python3 test.py samples/Clocks\ -\ Coldplay.m4a checkpoints/big_sample_rate/final.pt
```

## Visualize training data

```sh
tensorboard --logdir logs/{training_run}
```
