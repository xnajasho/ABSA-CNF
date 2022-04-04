import nltk
import spacy
import re
from nltk import word_tokenize

import stanza
#stanza.download('en')

from contractions import CONTRACTION_MAP

def test():
    nlp = spacy.load("en_core_web_sm")
    sentence = "I stumbled upon this second floor walk-up two Fridays ago when I was with two friends in town from L.A. Being serious sushi lovers, we sat at the sushi bar to be closer to the action."
    doc = nlp(expand_contractions(sentence))
    tokenized = nltk.word_tokenize(sentence)
    final = []

    joined_word = ""

    for i in range(len(doc)):
        dep_token = doc[i].text
        print((dep_token, doc[i].dep_))
        if (dep_token == tokenized[0]):
            final.append((dep_token, doc[i].dep_))
            tokenized.pop(0)
            continue

        hyphenated_word = tokenized[0]
        if hyphenated_word[len(hyphenated_word) - len(dep_token):len(hyphenated_word)] == dep_token:
            joined_word += dep_token
            final.append((joined_word, "hyp"))
            tokenized.pop(0)
            joined_word = ""
        else:
            joined_word += dep_token

    print(final)
    print(nltk.word_tokenize(sentence))

def test2():
    nlp = stanza.Pipeline('en', processors = 'tokenize,mwt,pos,lemma,depparse')
    sentence = "He cannot help, they are bought up so fast."
    doc = nlp(sentence)
    lst = []
    for sentences in doc.to_dict():
        for dict in sentences:
            lst.append((dict['text'], dict['deprel']))
    print(lst)
    print(nltk.word_tokenize(sentence))

def expand_contractions(text, contraction_mapping=CONTRACTION_MAP):
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


if __name__ == "__main__":
    test2()