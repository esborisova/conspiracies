import spacy
from spacy.tokens import Span
from relationextraction import SpacyRelationExtractor
from collections import Counter
from heads_extract_component import HeadwordsExtraction


def test_normalize_to_span():

    nlp = spacy.load("en_core_web_lg")
    nlp.add_pipe("heads_extraction")

    doc = nlp("Mette Frederiksen is the Danish politician.")

    normalized_token = doc[0]._.to_span  # Token

    assert isinstance(normalized_token, Span)
    assert normalized_token.text == "Mette"

    normalized_token = doc[0:2]._.to_span  # Span

    assert isinstance(normalized_token, Span)
    assert normalized_token.text == "Mette Frederiksen"

    normalized_token = doc[:]._.to_span  # Doc

    assert isinstance(normalized_token, Span)
    assert normalized_token.text == "Mette Frederiksen is the Danish politician."


def test_normalize_to_entity():

    nlp = spacy.load("en_core_web_lg")
    nlp.add_pipe("heads_extraction", config={"normalize_to_entity": True})

    doc = nlp("Mette Frederiksen is the Danish politician.")

    normalized_token = doc[0]._.to_span

    assert isinstance(normalized_token, Span)
    assert normalized_token.text == "Mette Frederiksen"


def test_normalize_to_span():

    nlp = spacy.load("en_core_web_lg")
    nlp.add_pipe("heads_extraction", config={"normalize_to_noun_chunk": True, force=True})

    doc = nlp("Mette Frederiksen is the Danish politician.")

    noun_chunk = doc[1]._.to_span

    assert isinstance(noun_chunk, Span)
    assert noun_chunk.text == "Mette Frederiksen"
