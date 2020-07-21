#! /usr/bin/env python
# -*- coding: utf-8 -*-
from argparse import ArgumentParser
import tensorflow as tf
import os
import sys
import time
import numpy as np
import datetime
import data_helpers
from text_cnn import TextCNN
from multi_class_data_loader import MultiClassDataLoader
from tensorflow.keras.preprocessing.text import Tokenizer
    
def main(args):
    
    # Parameters
    # ==================================================
    # Data
    tf.flags.DEFINE_string("train_corpus_path", args.train_corpus_path, "train txt file path")
    tf.flags.DEFINE_string("dev_corpus_path", args.dev_corpus_path, "dev txt file path")
    tf.flags.DEFINE_string("test_corpus_path", args.test_corpus_path, "test txt file path")
    tf.flags.DEFINE_string("class_data_path", args.class_data_path, "Data source for the class list")

    # Tokenizer
    tf.flags.DEFINE_string("tokenizer_type", args.tokenizer_type, "mecab/sp/kobert")
    tf.flags.DEFINE_integer("vocab_size", args.vocab_size, "Number of vocabulary")

    # Model Hyperparameters
    tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
    tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
    tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
    tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
    tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularizaion lambda (default: 0.0)")

    # Training parameters
    tf.flags.DEFINE_integer("batch_size", args.batch_size, "Batch Size (default: 64)")
    tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
    tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
    tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
    tf.flags.DEFINE_integer("patience", args.patience, "early-stops after this patience (default: 7)")
    # Misc Parameters
    tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
    tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

    FLAGS = tf.flags.FLAGS
    FLAGS(sys.argv)
    print("\nParameters:")
    for attr, value in sorted(FLAGS.__flags.items()):
        print("{}={}".format(attr.upper(), value))
    print("")

    # Tokenizer Preparation
    max_len = args.max_len
    vocab_size = args.vocab_size  # default 30000
    oov_token = '<UNK>'
    data_loader = MultiClassDataLoader(tf.flags, Tokenizer(vocab_size, oov_token=oov_token), max_len=max_len)

    # Data Preparation
    # ==================================================

    # Load data
    print("Loading data...")
    x_train, y_train, x_dev, y_dev, x_test, y_test = data_loader.prepare_data()
    vocab_processor = data_loader.vocab_processor

    print("Vocabulary in Total: {:d}".format(len(data_loader.vocab_processor.word_index)))
    print("Vocabulary Size: {:d}".format(vocab_size))
    print("Train/Valid/Test split: {:d}/{:d}/{:d}".format(len(y_train), len(y_dev), len(y_test)))


    # Training
    # ==================================================

    with tf.device('/device:GPU:0'), tf.Graph().as_default():
        session_conf = tf.compat.v1.ConfigProto(
          allow_soft_placement=FLAGS.allow_soft_placement,
          log_device_placement=FLAGS.log_device_placement)
        sess = tf.compat.v1.Session(config=session_conf)
        with sess.as_default():
            cnn = TextCNN(
                sequence_length=x_train.shape[1],
                num_classes=y_train.shape[1],
                vocab_size = vocab_size,
                embedding_size=FLAGS.embedding_dim,
                filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                num_filters=FLAGS.num_filters,
                l2_reg_lambda=FLAGS.l2_reg_lambda)

            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.compat.v1.train.AdamOptimizer(1e-3)
            grads_and_vars = optimizer.compute_gradients(cnn.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            # Keep track of gradient values and sparsity (optional)
            grad_summaries = []
            for g, v in grads_and_vars:         
                if g is not None:
                    grad_hist_summary = tf.compat.v1.summary.histogram("{}/grad/hist".format(v.name), g)
                    sparsity_summary = tf.compat.v1.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)
            grad_summaries_merged = tf.compat.v1.summary.merge(grad_summaries)

            # Output directory for models and summaries
            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
            print("Writing to {}\n".format(out_dir))

            # Summaries for loss and accuracy
            loss_summary = tf.compat.v1.summary.scalar("loss", cnn.loss)
            acc_summary = tf.compat.v1.summary.scalar("accuracy", cnn.accuracy)

            # Train Summaries
            train_summary_op = tf.compat.v1.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.compat.v1.summary.FileWriter(train_summary_dir, sess.graph)

            # Dev summaries
            dev_summary_op = tf.compat.v1.summary.merge([loss_summary, acc_summary])
            dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
            dev_summary_writer = tf.compat.v1.summary.FileWriter(dev_summary_dir, sess.graph)

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables())

            # Write vocabulary
            import pickle
            with open(os.path.join(out_dir, "vocab.pickle"), 'wb') as handle:
                pickle.dump(vocab_processor, handle, protocol=pickle.HIGHEST_PROTOCOL)

            # Initialize all variables
            sess.run(tf.compat.v1.global_variables_initializer())

            def train_step(x_batch, y_batch):
                """
                A single training step
                """
                feed_dict = {
                  cnn.input_x: x_batch,
                  cnn.input_y: y_batch,
                  cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
                }
                _, step, summaries, loss, accuracy = sess.run(
                    [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()

                if step % int(FLAGS.evaluate_every/2) == 0:
                    print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))

                train_summary_writer.add_summary(summaries, step)

            def dev_step(x_batch, y_batch, writer=None):
                """
                Evaluates model on a dev set
                """
                feed_dict = {
                  cnn.input_x: x_batch,
                  cnn.input_y: y_batch,
                  cnn.dropout_keep_prob: 1.0
                }
                step, summaries, loss, accuracy = sess.run(
                    [global_step, dev_summary_op, cnn.loss, cnn.accuracy],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                if writer:
                    writer.add_summary(summaries, step)
                return loss

            # Generate batches
            batches = data_helpers.batch_iter(
                list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)

            # Training loop. For each batch...
            val_loss_min = np.Inf
            patience = FLAGS.patience
            early_stopping_counter = 0
            train_start = datetime.datetime.now()
            train_end = datetime.datetime.now()

            for batch in batches:
                x_batch, y_batch = zip(*batch)
                train_step(x_batch, y_batch)
                current_step = tf.compat.v1.train.global_step(sess, global_step)
                if current_step % FLAGS.evaluate_every == 0:
                    print("\nEvaluation:")
                    curr_loss = dev_step(x_dev, y_dev, writer=dev_summary_writer)
                    if curr_loss <= val_loss_min:
                        val_loss_min = curr_loss
                        early_stopping_counter = 0
                        train_end = datetime.datetime.now()
                        path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                        print("Saved model checkpoint to {}\n".format(path))
                        print("\nTest Evaluation:")
                        test_loss = dev_step(x_test, y_test, writer=dev_summary_writer)
                        print('test loss:', test_loss)
                    elif curr_loss > val_loss_min:
                        early_stopping_counter += 1
                        print('early stopping counter:', str(early_stopping_counter))
                        if early_stopping_counter == patience:
                            total_train = (train_end - train_start).total_seconds()
                            print('early stopping')
                            print('total train time:', str(total_train),'s')
                            break
                    print("")
                
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--train_corpus_path', type=str, required=True, help="train txt file path")
    parser.add_argument('--dev_corpus_path', type=str, required=True, help="dev txt file path")
    parser.add_argument('--test_corpus_path', type=str, required=True, help="test txt file path")
    parser.add_argument('--class_data_path', type=str, required=True, help="Data source for the class list")
    parser.add_argument('--tokenizer_type', type=str, help="mecab/sp/kobert")

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--patience", type=int, default=7,
                        help="Number of epochs before stopping once your loss stops improving")
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--vocab_size", type=int, default=30000)
    
    args = parser.parse_args()
    main(args)