import spacy
from conspiracies.component_HeadWordExtraction import contains_ents


def test_ents_filter():

    nlp = spacy.load("en_core_web_lg")
    nlp.add_pipe("heads_extraction")

    doc = nlp("Mette Frederiksen is the Danish politician.")
    assert contains_ents(doc[:]) is True
    assert contains_ents(doc[1:2]) is True
    assert contains_ents(doc[2:4]) is False
