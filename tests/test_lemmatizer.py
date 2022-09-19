import spacy
from conspiracies.supernodes import lemmatize


def test_lemmatization():
    nlp = spacy.load("en_core_web_lg")
    test_phrases = "he goes to school, beautiful weather, democratic society, democrats"
    lemmas = lemmatize(test_phrases, nlp=nlp)

    assert isinstance(lemmas, str)
    assert (
        lemmas == "he go to school , beautiful weather , democratic society , democrats"
    )
