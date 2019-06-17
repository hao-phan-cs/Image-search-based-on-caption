import spacy
import re

nlp = spacy.load('en_core_web_sm')

def tokenize(text):
    tokens = re.findall(r"[a-z]+", text)
    return ' '.join(tokens)

def generate_term(text):
    text = text.lower()
    tokens = tokenize(text)
    tokens = nlp(tokens)
    result = []
    for token in tokens:
        if not token.is_punct and token.text not in ['a', 'an', 'the']:
            if token.lemma_ != " ":
                result.append(token.lemma_)
    return result

if __name__ == "__main__":
    word = 'IM NOT EXACTLY SURE WHAT THIS IS A PICTURE OF'
    # word = tokenize(word)
    print(word)
    word_list = generate_term(word)
    print(word_list)
