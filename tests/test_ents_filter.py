import spacy
from spacy.tokens import Span
from relationextraction import SpacyRelationExtractor
from heads_extract_component import contains_ents
from collections import Counter


def test_ents_filter():

    nlp = spacy.load("en_core_web_lg")
    nlp.add_pipe("heads_extraction")

    doc = nlp("Mette Frederiksen is the Danish politician.")
    span = doc[:]
    span1 = doc[1:2]
    span2 = doc[2:4]

    assert contains_ents(span) == True
    assert contains_ents(span1) == True
    assert contains_ents(span2) == False
