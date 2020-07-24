import tensorflow as tf
import numpy as np


class TextCNN_Classifier(object):

    def __init__(self, checkpoint_dir, checkpoint_path, TextProcessor):
        """
        TextProcessor: TokenizerProcessor or SentencepieceProcessor
        """
        self.__checkpoint_dir = checkpoint_dir
        self.__checkpoint_path = checkpoint_path
        self.text_processor = TextProcessor

    def predict_proba(self, texts):
        graph = tf.Graph()
        with graph.as_default():
            session_conf = tf.ConfigProto(
                allow_soft_placement=True,
                log_device_placement=False)
            sess = tf.Session(config=session_conf)

            with sess.as_default():
                saver = tf.train.import_meta_graph('{}.meta'.format(self.__checkpoint_path))
                saver.restore(sess, tf.train.latest_checkpoint(self.__checkpoint_dir))

                # Get the placeholders from the graph by name
                input_x = graph.get_operation_by_name("input_x").outputs[0]
                dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

                # Tensors we want to evaluate
                softmax = graph.get_operation_by_name('output/softmax').outputs[0]

                # Transform data
                x_setnence = list(self.text_processor.transform(texts))

                # Predict
                proba = sess.run(softmax, {input_x: x_setnence, dropout_keep_prob: 1.0})
                return proba

    def predict_label(self, texts):
        return np.argmax(self.predict_proba(texts=texts))