# -*- coding: utf-8 -*-
import sys
import re
import nltk
import csv
import sklearn.feature_extraction.text as txt
from nltk.corpus import stopwords

def clear_text(raw_text):
    lowered_case = raw_text.lower()
    letters_only = re.sub(r"http\S+", "", lowered_case)
    letters_only = re.sub("[\d,.?\-_!;:<>/\\\\\"\\'@…]", "", letters_only)
    stopwords_portuguese = stopwords.words('portuguese')
    words = letters_only.split()
    words = [w for w in words if w not in stopwords_portuguese]
    return (" ".join(words))

def get_info_from(filename):
    corpus = []
    labels = []
    with open(filename, mode="r") as input:
        reader = csv.DictReader(input, delimiter=",")
        for row in reader:
            corpus.append(row['tweet'])
            labels.append(row['review'])
    return corpus, labels

def clear_corpus(corpus):
    clean_corpus = []
    for document in corpus:
        clean_corpus.append(clear_text(document))
    return clean_corpus

def vectorize_frequency(corpus):
    vectorizer = txt.CountVectorizer(analyzer="word", \
            tokenizer=None, \
            preprocessor=None, \
            stop_words=None, \
            max_features=5000)
    corpus_features = vectorizer.fit_transform(corpus)
    return vectorizer, corpus_features.toarray()

def vectorize_binary(corpus):
    vectorizer = txt.CountVectorizer(analyzer="word", \
            tokenizer=None, \
            preprocessor=None, \
            stop_words=None, \
            binary=True, \
            max_features=5000)
    corpus_features = vectorizer.fit_transform(corpus)
    return vectorizer, corpus_features.toarray()

def vectorize_tf_idf(corpus):
    vectorizer = txt.TfidfVectorizer(analyzer="word", \
            tokenizer=None, \
            preprocessor=None, \
            stop_words=None, \
            max_features=5000)
    corpus_features =  vectorizer.fit_transform(corpus)
    return vectorizer, corpus_features.toarray()

def write_to_csv(corpus_features, labels, filename):
    fieldnames = ['x' + str(i) for i in range(corpus_features.shape[1])]
    fieldnames.append('label')
    with open(filename, mode="w") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames, delimiter=",")
        writer.writeheader()
        for features, label in zip(corpus_features, labels):
            row = {}
            for i in range(corpus_features.shape[1]):
                row['x' + str(i)] = features[i]
            row['label'] = label
            writer.writerow(row)

def main(filename):
    if filename is None:
        raise RuntimeError("Por favor, forneca um nome de arquivo")
        sys.exit(1)
    corpus, labels = get_info_from(filename)
    clean_corpus = clear_corpus(corpus)
    vectorizer_tf, tfs = vectorize_tf_idf(clean_corpus)
    vectorizer_freq, freqs = vectorize_frequency(clean_corpus)
    vectorizer_binary, binaries = vectorize_binary(clean_corpus)
    print("gerando csv com as frequencias")
    write_to_csv(freqs, labels, "output_freq.csv")
    print("gerando csv com tf-idf")
    write_to_csv(tfs, labels, "output_tf.csv")
    print("gerando csv com vetor binario")
    write_to_csv(binaries, labels, "output_binary.csv")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Uso: python bow.py <nome-do-arquivo>")
        sys.exit(1)
    main(sys.argv[1])
