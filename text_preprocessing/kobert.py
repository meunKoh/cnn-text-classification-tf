import gluonnlp as nlp
from kobert.utils import get_tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model


class KobertProcessor(object):

    def __init__(self, max_len, vocab_size=8002):
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.tokenizer = self.load_tokenizer()

    def load_tokenizer(self):
        # Build vocabulary
        bertmodel, vocab = get_pytorch_kobert_model()
        tokenizer = get_tokenizer()
        kobert_tokenizer = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)
        transform = nlp.data.BERTSentenceTransform(
            kobert_tokenizer, max_seq_length=self.max_len, pad=True, pair=False)
        return transform

    def process_datasets(self, datasets):
        return [self.transform(dataset) for dataset in datasets]

    def transform(self, data):
        data = [self.tokenizer([sentence])[0] for sentence in data]
        return np.array([array for array in data])