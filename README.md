# Kaggle Clouds Segmentation Challenge



* [Kaggle Clouds Segmentation Challenge](#kaggle-clouds-segmentation-challenge)
	* [Requirements](#requirements)
	* [Features](#features)
	* [Folder Structure](#folder-structure)
	* [Usage](#usage)
	* [Customization](#customization)
		* [Custom CLI options](#custom-cli-options)
		* [Data Loader](#data-loader)
		* [Trainer](#trainer)
		* [Model](#model)
		* [Loss](#loss)
		* [metrics](#metrics)
		* [Additional logging](#additional-logging)
		* [Validation data](#validation-data)
		* [Checkpoints](#checkpoints)
    * [Tensorboard Visualization](#tensorboard-visualization)
	* [TODOs](#todos)
	* [License](#license)
	* [Acknowledgements](#acknowledgements)

<!-- /code_chunk_output -->

## Requirements
* Python >= 3.5 (3.6 recommended)
* PyTorch >= 1.1 (1.2 recommended)
* tqdm (Optional for `test.py`)
* tensorboard >= 1.14 (see [Tensorboard Visualization](#tensorboard-visualization))
* torchvision
* numpy
* tqdm
* tensorboard>=1.14
* catalyst
* matplotlib
* albumentations
* segmentation-models-pytorch
* seaborn
* scikit-learn

## Folder Structure
  ```
  kaggle-clouds-segmentation-challenge/
  │
  ├── train.py - main script to start training
  ├── test.py - evaluation of trained model
  ├── data_loader/ - anything about data loading goes here
  │   └── data_loaders.py
  |
  ├──saved_models/ - contains the saved model weights after the training
  ├── data/ - default directory for storing input data
  │
  ├── model/ - models, losses, and metrics
  │   ├── model.py
  |
  ├── saved/
  │   ├── models/ - trained models are saved here
  │  
  │
  └── utils/ - small utility functions
      ├── utils.py

  ```

## Usage
```py

python train.py -bs <batch_size:int> -num_epochs <num_epochs:int>

Default values:
* batch_size : 16
* num_epochs: 20

```



<!-- ### Resuming from checkpoints
You can resume from a previously saved checkpoint by:

  ```
  python train.py --resume path/to/checkpoint
  ``` -->



<!-- ### Data Loader

### Testing
You can test trained model by running `test.py` passing path to the trained checkpoint by `--resume` argument.

### Validation data


### Checkpoints

**Note**: checkpoints contain:
  ```python
  {
    'arch': arch,
    'epoch': epoch,
    'state_dict': self.model.state_dict(),
    'optimizer': self.optimizer.state_dict(),
    'monitor_best': self.mnt_best,
    'config': self.config
  }
  ```

### Tensorboard Visualization
This template supports Tensorboard visualization by using either  `torch.utils.tensorboard` or [TensorboardX](https://github.com/lanpa/tensorboardX).

1. **Install**

    If you are using pytorch 1.1 or higher, install tensorboard by 'pip install tensorboard>=1.14.0'.

    Otherwise, you should install tensorboardx. Follow installation guide in [TensorboardX](https://github.com/lanpa/tensorboardX).

2. **Run training** 

    Make sure that `tensorboard` option in the config file is turned on.

    ```
     "tensorboard" : true
    ```

3. **Open Tensorboard server** 

    Type `tensorboard --logdir saved/log/` at the project root, then server will open at `http://localhost:6006`

By default, values of loss and metrics specified in config file, input images, and histogram of model parameters will be logged.
If you need more visualizations, use `add_scalar('tag', data)`, `add_image('tag', image)`, etc in the `trainer._train_epoch` method.
`add_something()` methods in this template are basically wrappers for those of `tensorboardX.SummaryWriter` and `torch.utils.tensorboard.SummaryWriter` modules. 

**Note**: You don't have to specify current steps, since `WriterTensorboard` class defined at `logger/visualization.py` will track current steps. -->



## TODOs

- [ ] Resume Checkpoints
- [ ] Implement Callbacks
- [ ] Automate writing to submission files
- [ ] Enable TensorBoard Logging 

## License
This project is licensed under the MIT License. See  LICENSE for more details

## Acknowledgements
