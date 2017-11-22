import sys
import numpy as np
import sklearn.model_selection as model_selection
from sklearn.metrics import classification_report
from classifiers import logistic_regression, svm
from extractor import bow

def show_usage():
    '''
        Shows usage of the program
    '''
    print("Usage: python experiments_scikit.py <algorithm> <vectorizing_option>", \
            "algorithm: svm or logreg for logistic regression", \
            "vectorizing_option: bin, freq or tfidf", sep="\n")

def vectorize_data(corpus, option):
    '''
        Clears the corpus and vectorizes it according to the option parameter
        
        Parameters:
            corpus: corpus with texts to be vectorized
            option: option for the vectorization. the possible are:
                    bin: for a binary vector s.t. x_t = 1 if term t is present in the document x
                    freq: frequency vector with x_t = #amount of times t is present in document x
                    tfidf: vector with x_t = tf-idf value of the vector

        Returns:
            a pair vectorizer, vectorized_corpus given by the bow module
    '''
    corpus = bow.clear_corpus(corpus)
    if option == "bin":
        return bow.vectorize_binary(corpus)
    if option == "freq":
        return bow.vectorize_frequency(corpus)
    if option =="tfidf":
        return bow.vectorize_tf_idf(corpus)
    raise Exception("Opção inválida!")
    show_usage()
    sys.exit(-1)

def remove_positives(corpus, labels):
    indexes = []
    acc = 0
    for idx, lbl in enumerate(labels):
        if lbl == '1':
            acc += 1
            indexes.append(idx)
    print("removidos: {}".format(acc))
    new_corpus = [c for idx, c in enumerate(corpus) if idx not in indexes]
    new_labels = [lbl for idx, lbl in enumerate(labels) if idx not in indexes]
    return new_corpus, new_labels

def classifier_for(option):
    if option == "svm":
        classifier = svm.SVMLearn()
        gamma = np.logspace(-4, 4, 6)
        param_grid = [{'kernelType': [''], 'C': [10**x for x in range(-4, 3)]}, \
                {'kernelType': ['rbf'], 'gamma': gamma, 'C': [10**x for x in range(-4, 3)]}, \
                {'kernelType': ['poly'], 'degree': [3, 4, 5], 'coef': [1, 5, 10], \
                    'C': [10**x for x in range(-4, 3)]}]
    elif option == "logreg":
        lamb = np.logspace(-5, 4, 15)
        classifier = logistic_regression.LogRegClassifier()
        param_grid = [{'lamb': [1e-5, 10]}]
    else:
        raise Exception("Opção inválida!")
        show_usage()
        sys.exit(-1)
    return model_selection.GridSearchCV(classifier, param_grid, n_jobs=3)

def labels_for(classifier, labels):
    labels = np.array(labels, dtype=np.int)
    if classifier == "logreg":
        return labels + 1
    if classifier == "svm":
        new_labels = np.zeros(labels.shape[0])
        for idx, lbl in enumerate(labels):
            if lbl == 0:
                new_labels[idx] = 1
            else:
                new_labels[idx] = lbl
        return new_labels

def main(argv):
    grid_cv = classifier_for(argv[1])
    X, t = bow.get_info_from("output_clam.csv")
    print("tamanhos: X: {} e t: {}".format(len(X), len(t)))
    X, t = remove_positives(X, t)
    print("tamanhos: X: {} e t: {}".format(len(X), len(t)))
    t = labels_for(argv[1], t)
    X_train, X_test, t_train, t_test = model_selection.train_test_split(X, t)
    print("Vectorizing data")
    print()
    vectorizer, X_train_vectorized = vectorize_data(X_train, argv[2])
    print("Fitting data")
    print("Shapes: X_train {} t_train {}".format(X_train_vectorized.shape, t_train.shape))
    grid_cv.fit(X_train_vectorized, t_train)
    print("Best parameters set found on development set:")
    print()
    print(grid_cv.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = grid_cv.cv_results_['mean_test_score']
    stds = grid_cv.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, grid_cv.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print("Cleaning and vectorizing training data:")
    X_test_vectorized = vectorizer.transform(bow.clear_corpus(X_test)).toarray()
    print("Classification report:")
    print()
    t_pred = grid_cv.predict(X_test_vectorized)
    print(classification_report(t_test, t_pred))
    print()

if __name__ == "__main__":
    if len(sys.argv) != 3:
        show_usage()
        sys.exit(-1)
    main(sys.argv)
