#! /usr/bin/env python
# -*- coding: utf-8 -*-


class WordDataProcessor(object):
    """
    def vocab_processor(_, *texts):
        max_document_length = 0
        for text in texts:
            max_doc_len = max([len(line.split(" ")) for line in text])
            if max_doc_len > max_document_length:
                max_document_length = max_doc_len
        return learn.preprocessing.VocabularyProcessor(max_document_length)

    def restore_vocab_processor(_, vocab_path):
        return learn.preprocessing.VocabularyProcessor.restore(vocab_path)
    """
    def clean_data(_, string):
        if ":" not in string:
            string = string.strip().lower()
        return string
