import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import sentencepiece as spm
import gluonnlp as nlp
from kobert.utils import get_tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model


class VocabProcessor(object):

    def __init__(self, tokenizer_type, vocab_size, max_len):
        tokenizer_map = {
            'mecab': self.process_with_Tokenizer,
            'sp': self.process_with_Sentencepiece,
            'kobert': self.process_with_kobert
        }

        self.__tokenizer_type = tokenizer_type
        self.__process_with_tokenizer = tokenizer_map[self.__tokenizer_type]
        self.__vocab_size = vocab_size
        self.__max_len = max_len

    def process_texts(self, x_train, y_train, x_dev, y_dev, x_test, y_test):
        return self.__process_with_tokenizer(self, x_train, y_train, x_dev, y_dev, x_test, y_test)

    def process_with_Tokenizer(self, x_train, y_train, x_dev, y_dev, x_test, y_test):
        # Build vocabulary
        tokenizer = Tokenizer(num_words=self.__vocab_size+1, oov_token='[UNK]')
        tokenizer.fit_on_texts(x_train)

        # transform
        x_train = np.array(pad_sequences(tokenizer.texts_to_sequences(x_train)
                                         , maxlen=self.__max_len
                                         , padding='post'))
        x_dev = np.array(pad_sequences(tokenizer.texts_to_sequences(x_dev)
                                       , maxlen=self.__max_len
                                       , padding='post'))
        x_test = np.array(pad_sequences(tokenizer.texts_to_sequences(x_test)
                                        , maxlen=self.__max_len
                                        , padding='post'))
        return [x_train, y_train, x_dev, y_dev, x_test, y_test]

    def process_with_Sentencepiece(self, x_train, y_train, x_dev, y_dev, x_test, y_test):
        # Load tokenizer
        sp_model_path = './tokenizers/sp-up-good3-30k.model'
        sp_tokenizer = spm.SentencePieceProcessor()
        sp_tokenizer.load(sp_model_path)

        # transform
        x_train = np.array(pad_sequences([sp_tokenizer.encode_as_ids(data) for data in x_train]
                                         , maxlen=self.__max_len
                                         , padding='post'
                                         , value=1))
        x_dev = np.array(pad_sequences([sp_tokenizer.encode_as_ids(data) for data in x_dev]
                                       , maxlen=self.__max_len
                                       , padding='post'
                                       , value=1))
        x_test = np.array(pad_sequences([sp_tokenizer.encode_as_ids(data) for data in x_test]
                                        , maxlen=self.__max_len
                                        , padding='post'
                                        , value=1))
        return [x_train, y_train, x_dev, y_dev, x_test, y_test]

    def process_with_kobert(self, x_train, y_train, x_dev, y_dev, x_test, y_test):
        # Build vocabulary
        bertmodel, vocab = get_pytorch_kobert_model()
        tokenizer = get_tokenizer()
        kobert_tokenizer = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)
        transform = nlp.data.BERTSentenceTransform(
            kobert_tokenizer, max_seq_length=self.__max_len, pad=True, pair=False)

        x_train = [transform([sentence])[0] for sentence in x_train]
        x_train = np.array([array for array in x_train])

        x_dev = [transform([sentence])[0] for sentence in x_dev]
        x_dev = np.array([array for array in x_dev])

        x_test = [transform([sentence])[0] for sentence in x_test]
        x_test = np.array([array for array in x_test])
        return [x_train, y_train, x_dev, y_dev, x_test, y_test]

    def clean_data(self, string):
        if ":" not in string:
            string = string.strip().lower()
        return string
