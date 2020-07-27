import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

"""
클래스 분리 : TokenizerProcessor, SentencepieceProcessor, KobertProcessor
"""


class TokenizerProcessor(object):

    def __init__(self, max_len, vocab_size=30000, tokenizer=None):
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.tokenizer = tokenizer

    def fit_tokenizer(self, train_data):
        """data_train : train-data to fit tokenizer"""
        from tensorflow.keras.preprocessing.text import Tokenizer
        self.tokenizer = Tokenizer(num_words=self.vocab_size+1, oov_token='[UNK]')
        self.tokenizer.fit_on_texts(self.__load_train_data(train_data=train_data))

    def __load_train_data(self, train_data):
        x_train = []
        with open(train_data, 'r', encoding='utf-8') as tsvin:
            tsvin = tsvin.readlines()
            for row in tsvin:
                row = row.replace('\n', '')
                idx = row.rfind(',')
                data = row[:idx]
                x_train.append(data)
        return x_train

    def process_datasets(self, datasets):
        return [self.transform(dataset) for dataset in datasets]

    def transform(self, data):
        """data : list of string"""
        return np.array(pad_sequences(self.tokenizer.texts_to_sequences(data)
                                      , maxlen=self.max_len
                                      , padding='post'))


class SentencepieceProcessor(object):

    def __init__(self, max_len, tokenizer_name=None, vocab_size=30000):
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.__tokenizer_name = tokenizer_name
        self.__tokenizer = self.load_tokenizer()

    def load_tokenizer(self):
        if self.__tokenizer_name is None:
            return
        else:
            import sentencepiece as spm
            tokenizer_path = {
                'mesp30k': './tokenizers/mecab+sp-up-good3-30k.model',
                'sp30k': './tokenizers/sp-up-good3-30k.model'
            }
            sp_model_path = tokenizer_path[self.__tokenizer_name]
            sp_tokenizer = spm.SentencePieceProcessor()
            sp_tokenizer.load(sp_model_path)
            return sp_tokenizer

    def get_tokenizer(self):
        return self.__tokenizer

    def process_datasets(self, datasets):
        return [self.transform(dataset) for dataset in datasets]

    def transform(self, data):
        """data : list of string"""
        return np.array(pad_sequences([self.__tokenizer.encode_as_ids(x) for x in data]
                                      , maxlen=self.max_len
                                      , padding='post'
                                      , value=1))


class TextProcessor(object):

    def __init__(self, tokenizer_type, max_len, vocab_size=30000):
        tokenizer_map = {
            'mecab': self.__process_with_Tokenizer,
            'sp': self.__process_with_Sentencepiece,
            'mesp': self.__process_with_Sentencepiece,
            'kobert': self.__process_with_kobert
        }

        self.__tokenizer_type = tokenizer_type
        self.__tokenizer = None
        self.__process_with_tokenizer = tokenizer_map[self.__tokenizer_type]  # data 1개
        # self.__process_with_tokenizer = tokenizer_map[self.__tokenizer_type]
        self.__vocab_size = vocab_size
        self.__max_len = max_len

    def process_texts(self, x_train, y_train, x_dev, y_dev, x_test, y_test):
        return self.__process_with_tokenizer(x_train, y_train, x_dev, y_dev, x_test, y_test)

    def process_with_kobert(self, x_train, y_train, x_dev, y_dev, x_test, y_test):
        import gluonnlp as nlp
        from kobert.utils import get_tokenizer
        from kobert.pytorch_kobert import get_pytorch_kobert_model

        # Build vocabulary
        bertmodel, vocab = get_pytorch_kobert_model()
        tokenizer = get_tokenizer()
        kobert_tokenizer = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)
        transform = nlp.data.BERTSentenceTransform(
            kobert_tokenizer, max_seq_length=self.__max_len, pad=True, pair=False)

        # transform
        x_train = self.__process_with_kobert(x_train, transform)
        x_dev = self.__process_with_kobert(x_dev, transform)
        x_test = self.__process_with_kobert(x_test, transform)
        return [x_train, y_train, x_dev, y_dev, x_test, y_test]

    def __process_with_kobert(self, data, transform_fn):
        data = [transform_fn([sentence])[0] for sentence in data]
        return np.array([array for array in data])

    def clean_data(self, string):
        if ":" not in string:
            string = string.strip().lower()
        return string
