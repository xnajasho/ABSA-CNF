import string

from preprocessor import Preprocessor

import pycrfsuite
import nltk
import numpy as np
import spacy
import time
import copy
import stanza

from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


RESTAURANT_TRAIN_DIRECTORY = "data/train_data/Restaurants_Train_v2.xml"
RESTAURANT_TEST_DIRECTORY = "data/test_data/Restaurants_Test_Truth.xml"

LAPTOP_TRAIN_DIRECTORY = "data/train_data/Laptop_Train_v2.xml"
LAPTOP_TEST_DIRECTORY = "data/test_data/Laptops_Test_Truth.xml"

# nltk.download('wordnet')
# nltk.download('omw-1.4')
# nltk.download('vader_lexicon')
#stanza.download('en')
class CNFModel:

    def __init__(self, train_directory=RESTAURANT_TRAIN_DIRECTORY, test_directory=RESTAURANT_TEST_DIRECTORY):
        self.preprocessed = Preprocessor(train_directory, test_directory)
        self.train_data = self.preprocessed.train_data
        self.test_data =  self.preprocessed.test_data

        self.test_full_sentence = self.preprocessed.test_full_sentence
        self.train_full_sentence = self.preprocessed.train_full_sentence


    # sentence = [(w1, pos, bio_label), (w2, pos, bio_label),..., (wn, pos, bio_label)]
    def extract_features(self, sentence, index, corpus_used):

        if corpus_used == "Test":
            original_sentence = self.test_full_sentence[index]
        else:
            original_sentence = self.train_full_sentence[index]

        #nlp = spacy.load("en_core_web_sm")
        nlp = stanza.Pipeline('en', processors='tokenize,mwt,pos,lemma,depparse')
        #list_tokens_NER = self.get_tokens_NER(original_sentence, parser=nlp)

        list_tokens_dependency = self.get_tokens_dependency(original_sentence,
                                                            pos_sentence=copy.deepcopy(sentence), parser=nlp)
        #list_tokens_ner_dep = self.get_tokens_NER_DEP(original_sentence, parser=nlp)


        sentiment_analyzer = SentimentIntensityAnalyzer()

        all_features = []

        print("Joined POS sentence = " + ' '.join([tup[0] for tup in sentence]))
        print("Original Sentence = " + original_sentence)
        print(list_tokens_dependency)

        for i in range(len(sentence)):
            current_word = sentence[i][0]
            current_pos = sentence[i][1]
            #current_word_ner = list_tokens_NER[i]
            current_word_dep = list_tokens_dependency[i]

            if current_word != current_word_dep[0]:
                print((current_word, current_word_dep[0]))

            polarity_score = sentiment_analyzer.polarity_scores(current_word)

            lemmatizer = WordNetLemmatizer()
            stemmer = PorterStemmer()

            # Features relevant to the CURRENT token in sentence
            features = [
                'bias',
                'word.lower=' + current_word.lower(),
                'word[-3:]=' + current_word[-3:],
                'word[-2:]=' + current_word[-2:],
                'word.istitle=%s' % current_word.istitle(),
                'word.isdigit=%s' % current_word.isdigit(),
                'word.isupper=%s' % current_word.isupper(),
                'word.lemmatized=' + lemmatizer.lemmatize(current_word),
                'word.stemmed=' + stemmer.stem(current_word),
                'word.isStopword=%s' % self.isStopword(current_word),
                'word.positivityscore=%s' % polarity_score['pos'],
                'word.negativityscore=%s' % polarity_score['neg'],
                #'word.nerlabel=' + current_word_ner_dep[1],
                #'word.deplabel=' + current_word_ner_dep[2],
                'word.is_iobj=' + ('1' if current_word_dep[1] == 'iobj' else '0'),
                'word.is_dobj=' + ('1' if current_word_dep[1] == 'dobj' else '0'),
                'word.is_nsubj=' + ('1' if current_word_dep[1] == 'nsubj' else '0'),
                #'word.is_hyp=' + ('1' if current_word_dep[1] == 'hyp' else '0'),
                'pos.isSuperlative=%s' % self.isSuperlative(current_pos),
                'pos.isComparative=%s' % self.isComparative(current_pos),
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

            all_features.append(features)

        print("------------------------------------------------------------------------------------------------------")
        return all_features

    '''
        Helper function to get the generate individual token dependencies in a sentence
        Input sentence: A full english text sentence (not tokenized) 
                        e.g "Autonomous cars shift insurance liability toward manufacturers"
        Output: [(Autonomous, amod), (cars, nsubj), (shift, ROOT), (insurance, compound), (liability, dobj), 
                 (toward, prep), (manufacturers, pobj)]
    '''
    def get_tokens_dependency(self, full_sentence, pos_sentence, parser):
        doc = parser(full_sentence)
        output = []

        for sentences in doc.to_dict():

            joined_hyphenated_word = ""

            for token_info in sentences:
                dep_generated_token = token_info['text']
                dep_generated_dep = token_info['deprel']

                if dep_generated_token not in string.punctuation:

                    if (dep_generated_token == pos_sentence[0][0]):
                        output.append((dep_generated_token, dep_generated_dep))
                        pos_sentence.pop(0)
                        continue

                    hyphenated_word = pos_sentence[0][0]
                    joined_hyphenated_word += dep_generated_token
                    if hyphenated_word[len(hyphenated_word) - len(dep_generated_token):len(hyphenated_word)] == dep_generated_token:
                        output.append((joined_hyphenated_word, "hyp"))
                        pos_sentence.pop(0)
                        joined_hyphenated_word = ""
        return output

        # joined_hyphenated_word = ""
        # output = []
        #
        # for i in range(len(doc)):
        #     dep_generated_token = doc[i].text
        #
        #     if dep_generated_token not in string.punctuation:
        #         if (dep_generated_token == pos_sentence[0][0]):
        #             output.append((dep_generated_token, doc[i].dep_))
        #             pos_sentence.pop(0)
        #             continue
        #
        #         hyphenated_word = pos_sentence[0][0]
        #         joined_hyphenated_word += dep_generated_token
        #         if hyphenated_word[len(hyphenated_word) - len(dep_generated_token):len(hyphenated_word)] == dep_generated_token:
        #             output.append((joined_hyphenated_word, "hyp"))
        #             pos_sentence.pop(0)
        #             joined_hyphenated_word = ""

        return output

    '''
        Helper function to get the named entity recognition for a sentence
        Input sentence: A full english text sentence (not tokenized) e.g "San Francisco"
        Output: [('San', 'B-GPE'), ('Francisco', 'I-GPE')]
    '''
    def get_tokens_NER(self, full_sentence, parser):
        doc = parser(full_sentence)

        lst = []
        for i in range(len(doc)):

            # Entity BIO labels of token (might not be same as Aspect Labelled BIO tag)
            # 'B' if token = start of labelled entity, 'I' if token = inside an entity, 'O' if token != entity
            token_NER_IOB = doc[i].ent_iob_

            if token_NER_IOB == 'O':
                token_ner_info = (doc[i].text, token_NER_IOB)  # e.g ('nice', 'O')

            # token has an entity type, add according to its IOB label (if token is inside a multi-word entity)
            else:
                token_ner_info = (doc[i].text, token_NER_IOB + "-" + doc[i].ent_type_)  # e.g ('San', 'B-GPE'), ('Francisco', 'I-GPE')

            lst.append(token_ner_info)
        return lst

    '''
        Ignore this function for the time being
    '''
    def get_tokens_NER_DEP(self, full_sentence, parser):
        doc = parser(full_sentence)

        lst = []

        dep_list = ['dobj', 'xobj', 'iobj']

        for i in range(len(doc)):
            if doc[i].text not in string.punctuation:
                token_NER_IOB = doc[i].ent_iob_
                token_dep = doc[i].dep_
                # if token_dep not in dep_list:
                #     token_dep = "NULL"

                if token_NER_IOB == 'O':
                    token_ner_dep_info = (doc[i].text, token_NER_IOB, token_dep)

                else:
                    token_ner_dep_info = (doc[i].text, token_NER_IOB + "-" + doc[i].ent_type_, token_dep)

                lst.append(token_ner_dep_info)
        return lst

    def get_label(self, sentence):
        return [label for (token, pos, label) in sentence]

    def isSuperlative(self, pos):
        superlatives = ['JJS', 'RBS']
        if pos in superlatives:
            return True
        return False

    def isComparative(self, pos):
        comparatives = ['JJR', 'RBR']
        if pos in comparatives:
            return True
        return False

    def isStopword(self, token):
        if token in set(stopwords.words('english')):
            return True
        return False

    def train_model(self):
        print("Training Model...")
        X_train = [self.extract_features(self.train_data[i], i, corpus_used="Train") for i in range(len(self.train_data))]
        y_train = [self.get_label(sentence) for sentence in self.train_data]

        print('Generated Training Features + Labels...')
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
        print("Finished training model...")

    def predict(self):
        print("Predicting Model...")
        X_test = [self.extract_features(self.test_data[i], i, corpus_used="Test") for i in range(len(self.test_data))]
        y_test = [self.get_label(sentence) for sentence in self.test_data]

        tagger = pycrfsuite.Tagger()
        tagger.open('crf.model')

        y_pred = [tagger.tag(xseq) for xseq in X_test]

        print('Generated Predicted Features + Labels...')
        labels = {"B": 0, 'I': 1, 'O': 2}  # row indexes for position of labels in the classification matrix

        predictions = np.array([labels[tag] for row in y_pred for tag in row])
        truths = np.array([labels[tag] for row in y_test for tag in row])
        print(classification_report(truths, predictions, target_names=['B', 'I', 'O'])) # printing classification report


        new_y_test = list(map(lambda x: list(map(self.change_BIO, x)), y_test))
        new_y_pred = list(map(lambda x: list(map(self.change_BIO, x)), y_pred))

        print(self.get_metrics(new_y_test, new_y_pred, b=1)) ## printing new metric to calculate F1


    '''
        Helper Function to Change Bio label to numerical values. "O" = 0, "B" = 1, "I" = 2
    '''
    def change_BIO(self, label):
        if label == 'O':
            return 0
        elif label == 'B':
            return 1
        else:
            return 2

    '''
        Helper Function for get_metrics() to calculate new F1 metric measure
    '''
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


    '''
        Function to calculate new metric to evaluate our model instead of classification report.
    '''
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
    model = CNFModel()
    start_time = time.clock()
    model.train_model()
    model.predict()
    print("Time Taken = ", time.clock() - start_time)