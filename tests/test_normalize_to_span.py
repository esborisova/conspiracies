import spacy
from spacy.tokens import Span
from relationextraction import SpacyRelationExtractor
from collections import Counter
from heads_extract_component import HeadwordsExtraction


def test_normalize_to_span():

    nlp = spacy.load("en_core_web_lg")
    nlp.add_pipe("heads_extraction")

    doc = nlp("Mette Frederiksen is the Danish politician.")

    normalized_token = doc[0]._.normalize_to_span

    assert isinstance(normalized_token, Span)
    assert normalized_token.text == "Mette"
