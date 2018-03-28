# Yolo9000 Implementation

- [Yolo9000 Paper](https://arxiv.org/pdf/1612.08242.pdf)
- [YoloV1 Paper](https://arxiv.org/pdf/1506.02640.pdf)

## Dataset

Download and extract the VOC 2012 Train/Validation data from [here](https://pjreddie.com/projects/pascal-voc-dataset-mirror/).

## Installing

Create a Python 3 environment and install the dependences

```sh
pip install -r requirements.txt
```

## Running

If you have a `VOCdevkit` folder in the same directory as `run.py`, you should not need to change any config vars. Just run 

`python run.py`

This should do a few things

- If not found a TFRecords file will be created. In subsequent runs this file will be used
- The training process will be run
- Other stuff hopefully, still in progress...
