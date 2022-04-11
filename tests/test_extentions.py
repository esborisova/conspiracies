import spacy
from spacy.tokens import Doc, Span, Token
from relationextraction import SpacyRelationExtractor
from collections import Counter
from heads_extract_component import HeadwordsExtraction


def test_extentions():

    nlp = spacy.load("en_core_web_lg")
    doc = nlp("Mette Frederiksen is the Danish Politician")
    nlp.add_pipe("heads_extraction")

    assert isinstance(doc[:]._.most_common_ancestor, Span)  # Doc
    assert isinstance(doc[0:2]._.most_common_ancestor, Span)  # Span
