import spacy
from conspiracies.HeadWordExtractionComponent import get_entity_label


def test_ents_label_exctractor():

    nlp = spacy.load("en_core_web_lg")

    docs = nlp("Mette Frederiksen is the Danish politician.")

    assert get_entity_label(docs[0:2]) == "PERSON"
    assert get_entity_label(docs[4:5]) == "NORP"

    assert isinstance(get_entity_label(docs[0:2]), str)
    assert isinstance(get_entity_label(docs[4:5]), str)
