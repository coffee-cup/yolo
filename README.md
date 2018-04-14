# Yolo9000 Implementation

- [Yolo9000 Paper](https://arxiv.org/pdf/1612.08242.pdf)
- [YoloV1 Paper](https://arxiv.org/pdf/1506.02640.pdf)

[The final report can be found here](https://gitlab.csc.uvic.ca/courses/201801/csc486b/final-project/group-a/term-project/blob/master/report.pdf).

## Dataset

We trained on the [VOC2012 dataset](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/). However the VOC2007 or any data in the same format can be used.

Download and extract the VOC2012 Train/Validation data from [here](https://pjreddie.com/projects/pascal-voc-dataset-mirror/). 

## Installing

Create a Python 3 environment and install the dependences

```sh
pip install -r requirements.txt
```

## Training

If you have a `VOCdevkit` folder in the same directory as `run.py`, you should not need to change any config vars. Just run 

```sh
python run.py
```

This should do a few things

- If not found a TFRecords file will be created for both training and validation. In subsequent runs this file will be used
- The training process with start. By default for 1000 iterations with a batch size of 32. This is definitly not enough iterations and will require ~11GB of memory for the images

The network, evaluation, and training can be configured via command line args. Run

```sh
python run.py --help
```

to see the available commands.

The command that started the current long running training process is

```sh
python run.py --batch_size=32 --max_iter=45000 --learning_rate=3e-5
```

## Evaluation

All that shows up in Tensorboard is the current loss and preprocessed images with ground truth bounding boxes drawn.
To evaluate how the network is doing we output an image every `config.val_freq` to the `eval_images/` directory.
On the left of the image is the ground truth bounding boxes. On the right is the predicted bounding boxes. An example of this is

![Eval Image Example](https://i.imgur.com/wPvVNYy.jpg)

## Training Progress

The network is currently still training. Progress has been made and the network is finally making bounding box predictions.
Although fairly terrible ones. The loss after 27k iterations (31 hours) is...

![Loss](https://i.imgur.com/8iI6hS2.png)

The big spike in loss occurred right when the network started predicting bounding boxes with a confidence greater than 30%.
These predictions were not correct but the loss spiked because we started including how wrong these predictions were.
Before the spike we were only taking into account class and confidence predictions.

**Training progress will be updated here, even after the project deadline.**

[Current training progress](http://ec2-34-217-209-5.us-west-2.compute.amazonaws.com:6006)