# Yolo9000 Implementation

- [Yolo9000 Paper](https://arxiv.org/pdf/1612.08242.pdf)
- [YoloV1 Paper](https://arxiv.org/pdf/1506.02640.pdf)

[The final report can be found here](https://github.com/coffee-cup/yolo/blob/master/report.pdf).

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

## Thanks

I would like to thank all of the Github repositories and blogs found by searching "Yolo" and "Yolo Tensorflow". The source was not entirely
copied line by line, but the many found were used to guide our implementation. I wish this implementation may help others with their project one day.

- [keras-yolo2](https://github.com/experiencor/keras-yolo2)
- [yolo-tf](https://github.com/ruiminshen/yolo-tf)
- [yolov2](https://github.com/datlife/yolov2)
- [yolo v2 tutorial](https://mlblr.com/includes/mlai/index.html#yolov2)
