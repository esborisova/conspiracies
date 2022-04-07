import spacy
from spacy.tokens import Span
from relationextraction import SpacyRelationExtractor
from heads_extract_component import HeadwordsExtraction
from collections import Counter


def test_ents_filter():

    nlp = spacy.load("en_core_web_lg")
    nlp.add_pipe("heads_extraction")

    doc = nlp("Mette Frederiksen is the Danish politician.")
    span = doc[:]
    span1 = doc[1:2]
    span2 = doc[2:4]

    assert span._.contains_ents == True
    assert span1._.contains_ents == True
    assert span2._.contains_ents == False