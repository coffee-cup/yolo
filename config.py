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
    "--year",
    type=str,
    default="VOC2012",
    choices=["VOC2007", "VOC2012", "merged"],
    help="Desired challenge year")

dataset_arg.add_argument("--output_path", type=str, default="pascal.record")

# ----------------------------------------
# Arguments for training
train_arg = add_argument_group("Training")

train_arg.add_argument(
    "--record_file",
    type=str,
    default="./pascal_{}.record",
    help=
    "File for the Pascal VOC data. Expects {} to be in the name so it can be formatted with the split type"
)

train_arg.add_argument(
    "--learning_rate",
    type=float,
    default=1e-5,
    help="Learning rate (gradient step size)")

train_arg.add_argument(
    "--batch_size", type=int, default=32, help="Size of each training batch")

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
    "--val_freq", type=int, default=100, help="Validation interval")

train_arg.add_argument(
    "--report_freq", type=int, default=50, help="Summary interval")

train_arg.add_argument(
    "--print_boxes",
    type=str2bool,
    default=False,
    help="Whether or not to print the predicted validation bounding boxes")

train_arg.add_argument(
    '--allow_restore',
    type=str2bool,
    default=False,
    help='Whether or not to allow restoring model from checkpoint')

train_arg.add_argument(
    '--debug',
    type=str2bool,
    default=False,
    help='Whether or not to print train stats as it runs.')


def get_config():
    config, unparsed = parser.parse_known_args()

    return config, unparsed


def print_usage():
    parser.print_usage()
