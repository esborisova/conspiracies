import nltk
from nltk.tokenize import word_tokenize


def lemmatize(doc: str, nlp) -> str:
    """
    Lemmatizes a sequence of words.
    Args:
        doc (str): A sequence to be lemmatized.
        nlp: A spaCy pipeline.
    Returns:
        str: Lemmatized word sequence.
    """

    docs = nlp(doc)
    lemmas = " ".join([token.lemma_ for token in docs])
    return lemmas


def stem(doc: str, language="english") -> str:
    """
    Stemms a sequence of words.
    Args:
        doc (str): A sequence to be stemmed.
        language: The language of a doc under stemming.
    Returns:
        str: Stemmed word sequence.
    """

    stemmer = nltk.stem.SnowballStemmer(language)
    tokens = word_tokenize(doc)
    stemms = [stemmer.stem(token) for token in tokens]
    stemms = " ".join(stemms)
    return stemms
