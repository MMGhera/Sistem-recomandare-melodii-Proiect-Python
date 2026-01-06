# Train

## Model

We use a custom trained CNN (convolutional neural network) in which we feed
~10s samples of songs to predict probabilities that a given instrument is
present in a song.

Trained on [MTG-Jamendo dataset](https://github.com/MTG/mtg-jamendo-dataset/).

Melspectrogram generation config, the number of frames per sample that must be
fed into the model and calculated optimal thresholds for output probabilities
per instrument are embedded inside the final model file.

### Variants
- `big_sample_rate`: 22050 sample rate (the best)
- `small_sample_rate`: 12000 sample rate

## File structure
- `samples/`: song samples to test the model
- `logs/`: data gathered during training
- `checkpoints/`: intermediate models saved during training and final models
- `cleanup_data.py`: script to cleanup the dataset of mp3s that are not
required for instrument tagging
- `train.py`: script containing the training loop
- `test.py`: example functions for interacting with the model
- `model.py`: CNN model definition
- `dataset.py`: MTG-Jamendo dataset loading

## Test model

```sh
python3 test.py samples/Clocks\ -\ Coldplay.m4a checkpoints/big_sample_rate/final.pt
```

## Visualize training data

```sh
tensorboard --logdir logs/{training_run}
```
