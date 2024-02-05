# IQT with Connectom Data

## Installation Instructions

To be able to run IQT with Connectom Python 3.11 is recommended, but Python 3.10 can be used as well.
We recommend a Conda installation.

To install the dependencies, run

```
$ pip install -r requirements.txt
```

To utilise TensorFlow with GPU support, a Linux distribution or a-WSL2 compatible Windows distribution is required.

## Training

To train the model, run

```
$ python IQT/train.py
```

The training parameters can be configured inside the Python script. CLI parameters will be added in the near future. (WIP)

## Testing

We have provided a notebook to run our neural network output, labeled Test.ipynb under Notebooks.
See aforementioned notebook for further details.
