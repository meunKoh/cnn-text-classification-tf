import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from konlpy.tag import Mecab


class MecabTokenizerProcessor(object):

    def __init__(self, max_len, vocab_size=30000, train_tokenizer=False, tokenizer=None, train_data=None):
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.tokenizer = tokenizer
        if train_tokenizer is True:
            self.fit_tokenizer(train_data=train_data)
            print('fitted tokenizer words count:', len(self.tokenizer.word_counts))

    def fit_tokenizer(self, train_data):
        """train_data : train-data to apply mecab and fit tokenizer"""
        self.tokenizer = Tokenizer(num_words=self.vocab_size + 1, oov_token='[UNK]')
        self.tokenizer.fit_on_texts(self.__load_train_data(train_data=train_data))

    def __load_train_data(self, train_data):
        x_train = []
        with open(train_data, 'r', encoding='utf-8') as tsvin:
            tsvin = tsvin.readlines()
            for row in tsvin:
                row = row.replace('\n', '')
                idx = row.rfind(',')
                data = row[:idx]
                data = self.load_data(data)
                x_train.append(data)
        return x_train

    def load_data(self, data):
        """data : list of string"""
        mecab = Mecab()
        data = mecab.morphs(data)
        return ' '.join(data)

    def process_datasets(self, datasets):
        return [self.transform(dataset) for dataset in datasets]

    def transform(self, data):
        """data : list of string"""
        data_list = []
        for text in data:
            text = self.load_data(text)
            data_list.append(text)
        return np.array(pad_sequences(self.tokenizer.texts_to_sequences(data_list)
                                      , maxlen=self.max_len
                                      , padding='post'))