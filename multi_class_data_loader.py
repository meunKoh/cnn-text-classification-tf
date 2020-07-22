import numpy as np
from vocab_processor import VocabProcessor


class MultiClassDataLoader(object):
    """
    Handles multi-class training data.  It takes predefined sets of "train_data_file" and "dev_data_file"
    of the following record format.
        <text>,<class label>
    Class labels are given as "class_data_file", which is a list of class labels.
    """

    def __init__(self, flags):
        self.__flags = flags
        self.__train_data_file = None
        self.__dev_data_file = None
        self.__test_data_file = None
        self.__class_data_file = None
        self.__classes_cache = None
        self.__tokenizer_type = None
        self.__vocab_size = None
        self.__max_len = None

    def prepare_data(self):
        self.__resolve_params()
        x_train, y_train = self.__load_data_and_labels(self.__train_data_file)
        x_dev, y_dev = self.__load_data_and_labels(self.__dev_data_file)
        x_test, y_test = self.__load_data_and_labels(self.__test_data_file)

        vocab_processor = VocabProcessor(self.__tokenizer_type, self.__vocab_size, self.__max_len)
        return vocab_processor.process_texts(x_train, y_train, x_dev, y_dev, x_test, y_test)

    def class_labels(self, class_indexes):
        return [self.__classes()[idx] for idx in class_indexes]

    def class_count(self):
        return self.__classes().__len__()

    def load_dev_data_and_labels(self):
        self.__resolve_params()
        x_dev, y_dev = self.__load_data_and_labels(self.__dev_data_file)
        return [x_dev, y_dev]

    def load_data_and_labels(self):
        self.__resolve_params()
        x_train, y_train = self.__load_data_and_labels(self.__train_data_file)
        x_dev, y_dev = self.__load_data_and_labels(self.__dev_data_file)
        x_all = x_train + x_dev
        y_all = np.concatenate([y_train, y_dev], 0)
        return [x_all, y_all]

    @staticmethod
    def clean_data(string):
        if ":" not in string:
            string = string.strip().lower()
        return string

    def __load_data_and_labels(self, data_file):
        x_text = []
        y = []
        with open(data_file, 'r', encoding='utf-8') as tsvin:
            classes = self.__classes()
            one_hot_vectors = np.eye(len(classes), dtype=int)
            class_vectors = {}
            for i, cls in enumerate(classes):
                class_vectors[cls] = one_hot_vectors[i]

            tsvin = tsvin.readlines()
            for row in tsvin:
                row = row.replace('\n', '')
                idx = row.rfind(',')
                data = self.clean_data(row[:idx])
                x_text.append(data)
                y.append(class_vectors[row[idx + 1:]])
        return [x_text, np.array(y)]

    def __classes(self):
        self.__resolve_params()
        if self.__classes_cache is None:
            with open(self.__class_data_file, 'r', encoding='utf-8') as catin:
                classes = list(catin.readlines())
                self.__classes_cache = [s.strip() for s in classes]
        return self.__classes_cache

    def __resolve_params(self):
        if self.__class_data_file is None:
            self.__train_data_file = self.__flags.FLAGS.train_corpus_path
            self.__dev_data_file = self.__flags.FLAGS.dev_corpus_path
            self.__test_data_file = self.__flags.FLAGS.test_corpus_path
            self.__class_data_file = self.__flags.FLAGS.class_data_path
        if self.__tokenizer_type is None:
            self.__tokenizer_type = self.__flags.FLAGS.tokenizer_type
            self.__max_len = self.__flags.FLAGS.max_len
            self.__vocab_size = self.__flags.FLAGS.vocab_size
