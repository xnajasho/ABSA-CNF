from preprocessor import Preprocessor

import pycrfsuite
import nltk
import numpy as np

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


RESTAURANT_TRAIN_DIRECTORY = "data/train_data/Restaurants_Train_v2.xml"
RESTAURANT_TEST_DIRECTORY = "data/test_data/Restaurants_Test_Truth.xml"

LAPTOP_TRAIN_DIRECTORY = "data/train_data/Laptop_Train_v2.xml"
LAPTOP_TEST_DIRECTORY = "data/test_data/Laptops_Test_Truth.xml"

class CNFBaselineModel:

    def __init__(self, train_directory=RESTAURANT_TRAIN_DIRECTORY, test_directory=RESTAURANT_TEST_DIRECTORY):
        self.preprocessed = Preprocessor(train_directory, test_directory)
        self.train_data = self.preprocessed.train_data
        self.test_data =  self.preprocessed.test_data

    def word2features(self, sentence, i):
        current_word = sentence[i][0]
        current_pos = sentence[i][1]

        # Features relevant to the CURRENT token in sentence
        features = [
            'bias',
            'word.lower=' + current_word.lower(),
            'word[-3:]=' + current_word[-3:],
            'word[-2:]=' + current_word[-2:],
            'word.istitle=%s' % current_word.istitle(),
            'word.isdigit=%s' % current_word.isdigit(),
            'word.isupper=%s' % current_word.isupper(),
            'postag=' + current_pos,
            'postag[:2]=' + current_pos[:2],
        ]

        # Features for words that are not at the beginning of a sentence
        if i > 0:
            prev_word = sentence[i - 1][0]
            previous_pos = sentence[i - 1][1]
            features.extend([
                '-1:word.lower=' + prev_word.lower(),
                '-1:word.istitle=%s' % prev_word.istitle(),
                '-1:word.isdigit=%s' % prev_word.isdigit(),
                '-1:word.isupper=%s' % prev_word.isupper(),
                '-1:postag=' + previous_pos,
                '-1:postag[:2]=' + previous_pos[:2],
            ])
        else:
            features.append('BOS')

        # Features for words that are not at the end of a sentence
        if i < len(sentence) - 1:
            next_word = sentence[i + 1][0]
            next_pos = sentence[i + 1][1]
            features.extend([
                '+1:word.lower=' + next_word.lower(),
                '+1:word.istitle=%s' % next_word.istitle(),
                '+1:word.isdigit=%s' % next_word.isdigit(),
                '+1:word.isupper=%s' % next_word.isupper(),
                '+1:postag=' + next_pos,
                '+1:postag[:2]=' + next_pos[:2],
            ])
        else:
            features.append('EOS')

        return features

    def extract_features(self, sentence):
        return [self.word2features(sentence, i) for i in range(len(sentence))]

    def get_label(self, sentence):
        return [label for (token, pos, label) in sentence]

    def train_model(self):
        X_train = [self.extract_features(sentence) for sentence in self.train_data]
        y_train = [self.get_label(sentence) for sentence in self.train_data]

        trainer = pycrfsuite.Trainer(verbose=False)
        for xseq, yseq in zip(X_train, y_train):
            trainer.append(xseq, yseq)

        trainer.set_params({
            # coefficient for L1 penalty
            'c1': 0.1,

            # coefficient for L2 penalty
            'c2': 0.01,

            # maximum number of iterations
            'max_iterations': 200,

            # whether to include transitions that
            # are possible, but not observed
            'feature.possible_transitions': True
        })
        trainer.train('crf.model')
        print("Finished training model")

    def predict(self):
        X_test = [self.extract_features(sentence) for sentence in self.test_data]
        y_test = [self.get_label(sentence) for sentence in self.test_data]

        tagger = pycrfsuite.Tagger()
        tagger.open('crf.model')

        y_pred = [tagger.tag(xseq) for xseq in X_test]
        labels = {"B": 0, 'I': 1, 'O': 2}  # row indexes for position of labels in the classification matrix

        predictions = np.array([labels[tag] for row in y_pred for tag in row])
        truths = np.array([labels[tag] for row in y_test for tag in row])
        print(classification_report(truths, predictions, target_names=['B', 'I', 'O']))

        new_y_test = list(map(lambda x: list(map(self.change_BIO, x)), y_test))
        new_y_pred = list(map(lambda x: list(map(self.change_BIO, x)), y_pred))

        print(self.get_metrics(new_y_test, new_y_pred, b=1))

    def change_BIO(self, label):
        if label == 'O':
            return 0
        elif label == 'B':
            return 1
        else:
            return 2

    def get_term_pos(self, labels):
        start, end = 0, 0
        tag_on = False
        terms = []
        labels = np.append(labels, [0])
        for i, label in enumerate(labels):
            if label == 1 and not tag_on:
                tag_on = True
                start = i
            if tag_on and labels[i + 1] != 2:
                tag_on = False
                end = i
                terms.append((start, end))
        return terms

    def get_metrics(self, test_y, pred_y, b=1):
        common, relevant, retrieved = 0., 0., 0.
        for i in range(len(test_y)):
            cor = self.get_term_pos(test_y[i])
            pre = self.get_term_pos(pred_y[i])
            common += len([a for a in pre if a in cor])
            retrieved += len(pre)
            relevant += len(cor)
        p = common / retrieved if retrieved > 0 else 0.
        r = common / relevant
        f1 = (1 + (b ** 2)) * p * r / ((p * b ** 2) + r) if p > 0 and r > 0 else 0.
        return p, r, f1, common, retrieved, relevant


if __name__ == "__main__":
    model = CNFBaselineModel()
    model.train_model()
    model.predict()