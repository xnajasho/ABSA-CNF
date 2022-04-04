import codecs
import re
import nltk
import string

from bs4 import BeautifulSoup as bs
from bs4.element import Tag

from contractions import CONTRACTION_MAP

RESTAURANT_TRAIN_DIRECTORY = "data/train_data/Restaurants_Train_v2.xml"
RESTAURANT_TEST_DIRECTORY = "data/test_data/Restaurants_Test_Truth.xml"

LAPTOP_TRAIN_DIRECTORY = "data/train_data/Laptop_Train_v2.xml"
LAPTOP_TEST_DIRECTORY = "data/test_data/Laptops_Test_Truth.xml"

class Preprocessor:

    def __init__(self, train_file_directory, test_file_directory):

        with codecs.open(train_file_directory, "r", "utf-8") as train_file:
            self.train_soup = bs(train_file, "html5lib")

        with codecs.open(test_file_directory, "r", "utf-8") as test_file:
            self.test_soup = bs(test_file, "html5lib")


        # contains data of all training sentences, each sentence broken down into individual (word, pos, bio_tag)
        self.train_data = self.build_corpus(self.train_soup)

        self.train_full_sentence = self.get_original_sentences(self.train_soup)

        # contains data of all test sentences
        self.test_data = self.build_corpus(self.test_soup)

        self.test_full_sentence = self.get_original_sentences(self.test_soup)

    def find_all_occurrences(self, aspectTerms, text):
        lst = []
        for term in aspectTerms:
            currLst = [(m.start(), term) for m in re.finditer(term, text)]
            lst = currLst + lst
            text = re.sub(term, "ASPECT", text)
        lst.sort(key=lambda a: a[0])
        return [item[1:][0] for item in lst], text

    def expand_contractions(self, text, contraction_mapping=CONTRACTION_MAP):
        contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())),
                                          flags=re.IGNORECASE | re.DOTALL)

        def expand_match(contraction):
            match = contraction.group(0)
            first_char = match[0]
            expanded_contraction = contraction_mapping.get(match) \
                if contraction_mapping.get(match) \
                else contraction_mapping.get(match.lower())
            expanded_contraction = first_char + expanded_contraction[1:]
            return expanded_contraction

        expanded_text = contractions_pattern.sub(expand_match, text)
        expanded_text = re.sub("'", "", expanded_text)
        return expanded_text

    def build_corpus(self, soup_used):
        docs = []

        for elem in soup_used.find_all("sentence"):
            texts = []
            aspectTerms = []
            updated_tokens = []

            for c in elem.find_all("aspectterms"):
                for aspectTerm in c.find_all("aspectterm"):
                    # add aspect terms into the list
                    aspectTerms.append(aspectTerm['term'])

            for text in elem.find("text"):

                aspectTermsLst, modified_text = self.find_all_occurrences(aspectTerms, text)

                tokens = nltk.word_tokenize(self.expand_contractions(modified_text))

                for token in tokens:
                    word = token.strip(string.punctuation)
                    if len(word) != 0:
                        updated_tokens.append(word)

                for i in range(len(updated_tokens)):
                    if updated_tokens[i] == "ASPECT":
                        updated_tokens[i] = aspectTermsLst[0]
                        aspectTermsLst.pop(0)

                for token in updated_tokens:
                    # Remove punctuations except for special cases such as 17-inch, "sales" team
                    label = ""
                    # if in aspectTerm then label = A
                    # else label = N

                    if token in aspectTerms:
                        token_list = token.split()
                        label = "B"
                        texts.append((token_list[0], label))
                        if len(token_list) > 1:
                            label = "I"
                            token_list.pop(0)
                            for t in token_list:
                                texts.append((t, label))
                    else:
                        label = "O"
                        texts.append((token, label))

            docs.append(texts)

        corpus = []
        for i, doc in enumerate(docs):

            # Obtain the list of tokens in the document
            tokens = [t for t, label in doc]

            # Perform POS tagging
            tagged = nltk.pos_tag(tokens)

            # Take the word, POS tag, and its label
            corpus.append([(w, pos, label) for (w, label), (word, pos) in zip(doc, tagged)])

        return corpus

    def get_original_sentences(self, soup_used):
        corpus = []

        for elem in soup_used.find_all("sentence"):
            for text in elem.find("text"):
                corpus.append(self.expand_contractions(str(text)))

        return corpus

## To test out preprocessor class
if __name__ == "__main__":
    pp = Preprocessor(RESTAURANT_TRAIN_DIRECTORY, RESTAURANT_TEST_DIRECTORY)
    print(pp.train_full_sentence)