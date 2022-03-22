import string

from bs4 import BeautifulSoup as bs
from bs4.element import Tag
import codecs
import re
import nltk
nltk.download('punkt')

# Read data file and parse the XML
with codecs.open("data/train_data/Laptop_Train_v2.xml", "r", "utf-8") as infile:
    soup = bs(infile, "html5lib")


def findall_occurences(aspectTerms, text):
    lst = []
    for term in aspectTerms:
        currLst = [(m.start(), term) for m in re.finditer(term, text)]
        lst = currLst + lst
        text = re.sub(term, "ASPECT", text)
    lst.sort(key=lambda a: a[0])
    return [item[1:][0] for item in lst], text


docs = []
for elem in soup.find_all("sentence"):
    texts = []
    aspectTerms = []
    updated_tokens = []

    for c in elem.find_all("aspectterms"):
        for aspectTerm in c.find_all("aspectterm"):
            # add aspect terms into the list
            aspectTerms.append(aspectTerm['term'])

    for text in elem.find("text"):
        aspectTermsLst, modified_text = findall_occurences(aspectTerms, text)
        tokens = nltk.word_tokenize(modified_text)
        for token in tokens:
            updated_tokens.append(token.strip(string.punctuation))

        # remove empty strings


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
                label = "A"
            else:
                label = "N"
            texts.append((token, label))
    docs.append(texts)

data = []
for i, doc in enumerate(docs):

    # Obtain the list of tokens in the document
    tokens = [t for t, label in doc]

    # Perform POS tagging
    tagged = nltk.pos_tag(tokens)

    # Take the word, POS tag, and its label
    data.append([(w, pos, label) for (w, label), (word, pos) in zip(doc, tagged)])

print(data[0])
