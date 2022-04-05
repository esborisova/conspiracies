from extract_heads import (
    most_common_ancestor,
    normalize_token_to_span,
    contains_ents,
    extract_entities,
)
import spacy
from spacy.tokens import Span
from relationextraction import SpacyRelationExtractor
from collections import Counter


def test_normalize_token():

    nlp = spacy.load("en_core_web_lg")
    doc = nlp("The quick brown fox named Charles, jumps over the lazy dog.")

    normalized_token = normalize_token_to_span(doc[3])

    assert isinstance(normalized_token, Span)


def test_headword():

    nlp = spacy.load("en_core_web_lg")
    test_sents = ["Mette Frederiksen is the Danish politician."]

    config = {"confidence_threshold": 2.7, "model_args": {"batch_size": 10}}
    nlp.add_pipe("relation_extractor", config=config)

    pipe = nlp.pipe(test_sents)

    if not Span.has_extension("most_common_ancestor"):
        Span.set_extension("most_common_ancestor", getter=most_common_ancestor)

    for d in pipe:
        for span in d._.relation_head:
            headword = span._.most_common_ancestor

            assert isinstance(headword, Span)
            assert headword.text == "Frederiksen"


def test_ents_check():

    nlp = spacy.load("en_core_web_lg")
    test_sents = ["Mette Frederiksen is the Danish politician."]

    config = {"confidence_threshold": 2.7, "model_args": {"batch_size": 10}}
    nlp.add_pipe("relation_extractor", config=config)

    pipe = nlp.pipe(test_sents)

    if not Span.has_extension("most_common_ancestor"):
        Span.set_extension("most_common_ancestor", getter=most_common_ancestor)

    heads = []

    for d in pipe:
        for span in d._.relation_head:
            heads.append(span._.most_common_ancestor)

    for span in heads:
        assert contains_ents(span) == True


def test_ents_extractor():

    nlp = spacy.load("en_core_web_lg")
    test_sents = ["Mette Frederiksen is the Danish politician."]

    config = {"confidence_threshold": 2.7, "model_args": {"batch_size": 10}}
    nlp.add_pipe("relation_extractor", config=config)

    pipe = nlp.pipe(test_sents)

    if not Span.has_extension("extract_entities"):
        Span.set_extension("extract_entities", getter=extract_entities)

    for d in pipe:
        for span in d._.relation_head:
            entity = span._.extract_entities

            assert isinstance(entity, Span)
            assert entity.text == "Mette Frederiksen"
