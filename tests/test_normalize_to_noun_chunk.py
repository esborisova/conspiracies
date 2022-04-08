import spacy
from spacy.tokens import Span
from relationextraction import SpacyRelationExtractor
from collections import Counter
from heads_extract_component import HeadwordsExtraction


def test_normalize_to_span():

    nlp = spacy.load("en_core_web_lg")
    nlp.add_pipe("heads_extraction", config={"normalize_to_noun_chunk": True})

    doc = nlp("Mette Frederiksen is the Danish politician.")

    noun_chunk = doc[1]._.normalize_to_span

    assert isinstance(noun_chunk, Span)
    assert noun_chunk.text == "Mette Frederiksen"
