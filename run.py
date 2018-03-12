import os

import numpy as np
import tensorflow as tf
from tqdm import trange

from config import get_config, print_usage
from utils.cifar10 import load_data

data_dir = "./cifar-10-batches-py"


class MyNetwork(object):
    """Network class """

    def __init__(self, x_shp, config):

        self.config = config

        # Get shape
        self.x_shp = x_shp

        # Build the network
        self._build_placeholder()
        self._build_preprocessing()
        self._build_model()
        self._build_loss()
        self._build_optim()
        self._build_eval()
        self._build_summary()
        self._build_writer()

    def _build_placeholder(self):
        """Build placeholders."""

        # TODO: Get shape for placeholder
        x_in_shp = (None, self.x_shp[1], self.x_shp[2], self.x_shp[3])

        # Create Placeholders for inputs
        self.x_in = tf.placeholder(tf.float32, shape=x_in_shp)
        self.y_in = tf.placeholder(tf.int64, shape=(None, ))

    def _build_preprocessing(self):
        """Build preprocessing related graph."""

        with tf.variable_scope("Normalization", reuse=tf.AUTO_REUSE):
            # TODO: we will make `n_mean`, `n_range`, `n_mean_in` and
            # `n_range_in` as scalar this time! This is how we often use in
            # CNNs, as we KNOW that these are image pixels, and all pixels
            # should be treated equally!

            # Create placeholders for saving mean, range to a TF variable for
            # easy save/load. Create these variables as well.
            self.n_mean_in = tf.placeholder(tf.float32, shape=())
            self.n_range_in = tf.placeholder(tf.float32, shape=())
            # Make the normalization as a TensorFlow variable. This is to make
            # sure we save it in the graph
            self.n_mean = tf.get_variable(
                "n_mean", shape=(), trainable=False)
            self.n_range = tf.get_variable(
                "n_range", shape=(), trainable=False)
            # Assign op to store this value to TF variable
            self.n_assign_op = tf.group(
                tf.assign(self.n_mean, self.n_mean_in),
                tf.assign(self.n_range, self.n_range_in),
            )

    def _build_model(self):
        """We're probably going to put darknet in here..."""

        # Initializer and activations
        # if self.config.activ_type == "relu":
        #     activ = tf.nn.relu
        #     kernel_initializer = tf.keras.initializers.he_normal()
        # elif self.config.activ_type == "tanh":
        #     activ = tf.nn.tanh
        #     kernel_initializer = tf.glorot_normal_initializer()

        # Build the network (use tf.layers)
        
        """ We will probably put Darknet here"""

        # with tf.variable_scope("Network", reuse=tf.AUTO_REUSE):
        #     # Normalize using the above training-time statistics
        #     cur_in = (self.x_in - self.n_mean) / self.n_range
        #     # TODO: Convolutional layer 0. Make output shape become 32 > 28 >
        #     # 14 as we do convolution and pooling. We will also use the
        #     # argument from the configuration to determine the number of
        #     # filters for the inital conv layer. Have `num_unit` of filters,
        #     # use the kernel_initializer above.
        #     num_unit = self.config.num_conv_base
        #     cur_in = tf.layers.conv2d(cur_in, num_unit, [5, 5], kernel_initializer=kernel_initializer)
        #     # Activation
        #     cur_in = activ(cur_in)
        #     # TODO: use `tf.layers.max_pooling2d` to see how it should run. If
        #     # you want to try different pooling strategies, add it as another
        #     # config option. Be sure to have the max_pooling implemented.
        #     cur_in = tf.layers.max_pooling2d(cur_in, [2, 2], 2)
        #     # TODO: double the number of filters we will use after pooling
        #     num_unit *= 2
        #     # TODO: Convolutional layer 1. Make output shape become 14 > 12 > 6
        #     # as we do convolution and pooling. Have `num_unit` of filters,
        #     # use the kernel_initializer above.
        #     cur_in = tf.layers.conv2d(cur_in, num_unit, [3, 3], kernel_initializer=kernel_initializer)
        #     # Activation
        #     cur_in = activ(cur_in)
        #     # TODO: max pooling
        #     cur_in = tf.layers.max_pooling2d(cur_in, [2, 2], 2)
        #     # TODO: double the number of filters we will use after pooling
        #     num_unit *= 2
        #     # TODO: Convolutional layer 2. Make output shape become 6 > 4 > 2
        #     # as we do convolution and pooling. Have `num_unit` of filters,
        #     # use the kernel_initializer above.
        #     cur_in = tf.layers.conv2d(cur_in, num_unit, [3, 3], kernel_initializer=kernel_initializer)
        #     # Activation
        #     cur_in = activ(cur_in)
        #     # TODO: max pooling
        #     cur_in = tf.layers.max_pooling2d(cur_in, [2, 2], 2)
        #     # TODO: Flatten to put into FC layer with `tf.layers.flatten`
        #     cur_in = tf.layers.flatten(cur_in)
        #     # Hidden layers
        #     num_unit = self.config.num_unit
        #     for _i in range(self.config.num_hidden):
        #         cur_in = tf.layers.dense(
        #             cur_in, num_unit, kernel_initializer=kernel_initializer)
        #         if self.config.activ_type == "relu":
        #             cur_in = tf.nn.relu(cur_in)
        #         elif self.config.activ_type == "tanh":
        #             cur_in = tf.nn.tanh(cur_in)
        #     # Output layer
        #     self.logits = tf.layers.dense(
        #         cur_in, self.config.num_class,
        #         kernel_initializer=kernel_initializer)

        #     # Get list of all weights in this scope. They are called "kernel"
        #     # in tf.layers.dense.
        #     self.kernels_list = [
        #         _v for _v in tf.trainable_variables() if "kernel" in _v.name]

    def _build_loss(self):
        """Build our cross entropy loss."""

        with tf.variable_scope("Loss", reuse=tf.AUTO_REUSE):

            # Create cross entropy loss
            self.loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=self.y_in, logits=self.logits)
            )

            # Create l2 regularizer loss and add
            l2_loss = tf.add_n([
                tf.reduce_sum(_v**2) for _v in self.kernels_list])
            self.loss += self.config.reg_lambda * l2_loss

            # Record summary for loss
            tf.summary.scalar("loss", self.loss)

    def _build_optim(self):
        """Build optimizer related ops and vars."""

        with tf.variable_scope("Optim", reuse=tf.AUTO_REUSE):
            self.global_step = tf.get_variable(
                "global_step", shape=(),
                initializer=tf.zeros_initializer(),
                dtype=tf.int64,
                trainable=False)
            optimizer = tf.train.AdamOptimizer(
                learning_rate=self.config.learning_rate)
            self.optim = optimizer.minimize(
                self.loss, global_step=self.global_step)

    def _build_eval(self):
        """Build the evaluation related ops"""

        with tf.variable_scope("Eval", tf.AUTO_REUSE):

            # Compute the accuracy of the model. When comparing labels
            # elemwise, use tf.equal instead of `==`. `==` will evaluate if
            # your Ops are identical Ops.
            self.pred = tf.argmax(self.logits, axis=1)
            self.acc = tf.reduce_mean(
                tf.to_float(tf.equal(self.pred, self.y_in))
            )

            # Record summary for accuracy
            tf.summary.scalar("accuracy", self.acc)

            # TODO: We also want to save best validation accuracy. So we do
            # something similar to what we did before with n_mean. Note that
            # these will also be a scalar variable
            self.best_va_acc_in = tf.placeholder(tf.float32)
            self.best_va_acc = tf.get_variable("best_va_acc", dtype=tf.float32, shape=(), trainable=False)
            # TODO: Assign op to store this value to TF variable
            self.acc_assign_op = tf.assign(self.best_va_acc, self.best_va_acc_in)

    def _build_summary(self):
        """Build summary ops."""

        # Merge all summary op
        self.summary_op = tf.summary.merge_all()

    def _build_writer(self):
        """Build the writers and savers"""

        # Create summary writers (one for train, one for validation)
        self.summary_tr = tf.summary.FileWriter(
            os.path.join(self.config.log_dir, "train"))
        self.summary_va = tf.summary.FileWriter(
            os.path.join(self.config.log_dir, "valid"))
        # Create savers (one for current, one for best)
        self.saver_cur = tf.train.Saver()
        self.saver_best = tf.train.Saver()
        # Save file for the current model
        self.save_file_cur = os.path.join(
            self.config.log_dir, "model")
        # Save file for the best model
        self.save_file_best = os.path.join(
            self.config.save_dir, "model")

    def train(self, x_tr, y_tr, x_va, y_va):
        """Training function.

        Parameters
        ----------
        x_tr : ndarray
            Training data.

        y_tr : ndarray
            Training labels.

        x_va : ndarray
            Validation data.

        y_va : ndarray
            Validation labels.

        """

        # ----------------------------------------
        # Preprocess data

        # We will simply use the data_mean for x_tr_mean, and 128 for the range
        # as we are dealing with image and CNNs now
        x_tr_mean = x_tr.mean()
        x_tr_range = 128.0

        # Report data statistic
        print("Training data before: mean {}, std {}, min {}, max {}".format(
            x_tr_mean, x_tr.std(), x_tr.min(), x_tr.max()
        ))

        # ----------------------------------------
        # Run TensorFlow Session
        with tf.Session() as sess:
            # Init
            print("Initializing...")
            sess.run(tf.global_variables_initializer())

            # Assign normalization variables from statistics of the train data
            sess.run(self.n_assign_op, feed_dict={
                self.n_mean_in: x_tr_mean,
                self.n_range_in: x_tr_range,
            })

            # TODO: Check if previous train exists
            b_resume = tf.train.latest_checkpoint(self.config.log_dir)
            if b_resume:
                # TODO: Restore network
                print("Restoring from {}...".format(
                    self.config.log_dir))
                self.saver_best.restore(sess, b_resume)
                # TODO: restore number of steps so far
                step = tf.train.load_variable(b_resume, "global_step")
                # TODO: restore best acc
                best_acc = tf.train.load_variable(b_resume, "best_acc")
            else:
                print("Starting from scratch...")
                step = 0
                best_acc = 0

            print("Training...")
            batch_size = config.batch_size
            max_iter = config.max_iter
            # For each epoch
            for step in trange(step, max_iter):

                # Get a random training batch. Notice that we are now going to
                # forget about the `epoch` thing. Theoretically, they should do
                # almost the same.
                ind_cur = np.random.choice(
                    len(x_tr), batch_size, replace=False)
                x_b = np.array([x_tr[_i] for _i in ind_cur])
                y_b = np.array([y_tr[_i] for _i in ind_cur])

                # TODO: Write summary every N iterations as well as the first
                # iteration. Use `self.config.report_freq`. Make sure that we
                # write at the first iteration, and every kN iterations where k
                # is an interger. HINT: we write the summary after we do the
                # optimization.
                b_write_summary = step % (self.config.report_freq - 1) == 0
                if b_write_summary:
                    fetches = {
                        "optim": self.optim,
                        "summary": self.summary_op,
                        "global_step": self.global_step,
                    }
                else:
                    fetches = {
                        "optim": self.optim,
                    }

                # Run the operations necessary for training
                res = sess.run(
                    fetches=fetches,
                    feed_dict={
                        self.x_in: x_b,
                        self.y_in: y_b,
                    },
                )

                # TODO: Write Training Summary if we fetched it (don't write
                # meta graph). See that we actually don't need the above
                # `b_write_summary` actually :-). I know that we can check this
                # with b_write_summary, but let's check `res` to do this as an
                # exercise.
                if "summary" in res:
                    self.summary_tr.add_summary(
                        res["summary"], global_step=res["global_step"],
                    )
                    self.summary_tr.flush()

                    # Also save current model to resume when we write the
                    # summary.
                    self.saver_cur.save(
                        sess, self.save_file_cur,
                        global_step=self.global_step,
                        write_meta_graph=False,
                    )

                # TODO: Validate every N iterations and at the first
                # iteration. Use `self.config.val_freq`. Make sure that we
                # validate at the correct iterations. HINT: should be similar
                # to above.
                b_validate = step % (self.config.val_freq - 1) == 0
                if b_validate:
                    res = sess.run(
                        fetches={
                            "acc": self.acc,
                            "summary": self.summary_op,
                            "global_step": self.global_step,
                        },
                        feed_dict={
                            self.x_in: x_va,
                            self.y_in: y_va
                        })
                    # Write Validation Summary
                    self.summary_va.add_summary(
                        res["summary"], global_step=res["global_step"],
                    )
                    self.summary_va.flush()

                    # If best validation accuracy, update W_best, b_best, and
                    # best accuracy. We will only return the best W and b
                    if res["acc"] > best_acc:
                        best_acc = res["acc"]
                        # TODO: Write best acc to TF variable
                        sess.run(
                            fetches={
                                "best_acc": self.acc_assign_op
                            },
                            feed_dict={
                                self.best_va_acc_in: best_acc
                            }
                        )
                        # Save the best model
                        self.saver_best.save(
                            sess, self.save_file_best,
                            write_meta_graph=False,
                        )

    def test(self, x_te, y_te):
        """Test routine"""

        with tf.Session() as sess:
            # Load the best model
            latest_checkpoint = tf.train.latest_checkpoint(
                self.config.save_dir)
            if tf.train.latest_checkpoint(self.config.save_dir) is not None:
                print("Restoring from {}...".format(
                    self.config.save_dir))
                self.saver_best.restore(
                    sess,
                    latest_checkpoint
                )

            # Test on the test data
            res = sess.run(
                fetches={
                    "acc": self.acc,
                },
                feed_dict={
                    self.x_in: x_te,
                    self.y_in: y_te,
                },
            )

            # Report (print) test result
            print("Test accuracy with the best model is {}".format(
                res["acc"]))


def main(config):
    """The main function."""

    # ----------------------------------------
    # Load cifar10 train data
    print("Reading training data...")
    data_trva, y_trva = load_data(config.data_dir, "train")

    # ----------------------------------------
    # Load cifar10 test data
    print("Reading test data...")
    data_te, y_te = load_data(config.data_dir, "test")

    # ----------------------------------------
    # Extract features
    # We now simply use raw images
    print("Using raw images...")
    x_trva = data_trva.astype(float)
    x_te = data_te.astype(float)

    # Randomly shuffle data and labels. IMPORANT: make sure the data and label
    # is shuffled with the same random indices so that they don't get mixed up!
    idx_shuffle = np.random.permutation(len(x_trva))
    x_trva = x_trva[idx_shuffle]
    y_trva = y_trva[idx_shuffle]

    # Change type to float32 and int64 since we are going to use that for
    # TensorFlow.
    x_trva = x_trva.astype("float32")
    y_trva = y_trva.astype("int64")

    # ----------------------------------------
    # Simply select the last 20% of the training data as validation dataset.
    num_tr = int(len(x_trva) * 0.8)

    x_tr = x_trva[:num_tr]
    x_va = x_trva[num_tr:]
    y_tr = y_trva[:num_tr]
    y_va = y_trva[num_tr:]

    # ----------------------------------------
    # Init network class
    mynet = MyNetwork(x_tr.shape, config)

    # ----------------------------------------
    # Train
    # Run training
    mynet.train(x_tr, y_tr, x_va, y_va)

    # ----------------------------------------
    # Test
    mynet.test(x_te, y_te)


if __name__ == "__main__":

    # ----------------------------------------
    # Parse configuration
    config, unparsed = get_config()
    # If we have unparsed arguments, print usage and exit
    if len(unparsed) > 0:
        print_usage()
        exit(1)

    main(config)


