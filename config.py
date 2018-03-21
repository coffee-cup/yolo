import argparse

# ----------------------------------------
# Global variables within this script
arg_lists = []
parser = argparse.ArgumentParser()


# ----------------------------------------
# Some nice macros to be used for arparse
def str2bool(v):
    return v.lower() in ("true", "1")


def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg


# ----------------------------------------
# Arguments for creating the dataset from pascal voc data
dataset_arg = add_argument_group("Dataset")

dataset_arg.add_argument(
    "--data_dir",
    type=str,
    default="VOCdevkit",
    help="Root directory to raw PACAL VOC dataset.")

dataset_arg.add_argument(
    "--set",
    type=str,
    default="trainval",
    choices=["train", "val", "trainval", "test"],
    help="Type of data to convert")

dataset_arg.add_argument(
    "--annotations_dir",
    type=str,
    default="Annotations",
    help="(Relative) path to annotations directory.")

dataset_arg.add_argument(
    "--year",
    type=str,
    default="VOC2012",
    choices=["VOC2007", "VOC2012", "merged"],
    help="Desired challenge year")

dataset_arg.add_argument(
    "--output_path", type=str, default="data/pascal.record")

# ----------------------------------------
# Arguments for training
train_arg = add_argument_group("Training")

train_arg.add_argument(
    "--record_file",
    type=str,
    default="./pascal.record",
    help="File for the Pascal VOC data")

train_arg.add_argument(
    "--learning_rate",
    type=float,
    default=1e-3,
    help="Learning rate (gradient step size)")

train_arg.add_argument(
    "--batch_size", type=int, default=100, help="Size of each training batch")

train_arg.add_argument(
    "--max_iter", type=int, default=5000, help="Number of iterations to train")

train_arg.add_argument(
    "--log_dir",
    type=str,
    default="./logs",
    help="Directory to save logs and current model")

train_arg.add_argument(
    "--save_dir",
    type=str,
    default="./save",
    help="Directory to save the best model")

train_arg.add_argument(
    "--val_freq", type=int, default=500, help="Validation interval")

train_arg.add_argument(
    "--report_freq", type=int, default=50, help="Summary interval")

# ----------------------------------------
# Arguments for model
model_arg = add_argument_group("Model")

model_arg.add_argument(
    "--reg_lambda", type=float, default=1e-4, help="Regularization strength")

model_arg.add_argument(
    "--num_conv_base",
    type=int,
    default=8,
    help="Number of neurons in the first conv layer")

model_arg.add_argument(
    "--num_unit",
    type=int,
    default=64,
    help="Number of neurons in the hidden layer")

model_arg.add_argument(
    "--num_hidden", type=int, default=0, help="Number of hidden layers")

model_arg.add_argument(
    "--num_class",
    type=int,
    default=10,
    help="Number of classes in the dataset")

model_arg.add_argument(
    "--activ_type",
    type=str,
    default="relu",
    choices=["relu", "tanh"],
    help="Activation type")


def get_config():
    config, unparsed = parser.parse_known_args()

    return config, unparsed


def print_usage():
    parser.print_usage()
