import spacy


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
