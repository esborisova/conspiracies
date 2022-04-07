import spacy
from spacy.tokens import Span
from relationextraction import SpacyRelationExtractor
from collections import Counter
from heads_extract_component import HeadwordsExtraction


def test_normalize_token():

    nlp = spacy.load("en_core_web_lg")
    nlp.add_pipe("heads_extraction")

    doc = nlp("The quick brown fox named Charles, jumps over the lazy dog.")
    # doc1 = nlp("Mette Frederiksen is the Danish politician.")

    normalized_token = doc[3]._.normalize_token_to_span

    assert isinstance(normalized_token, Span)
    assert normalized_token.text == "fox"

    # normalized_token = normalize_token_to_span(doc1[1], normalize_to_entity = True)

    # assert isinstance(normalized_token, Span)
    # assert normalized_token.text == 'Mette Frederiksen'

    # normalized_token = normalize_token_to_span(doc1[1], normalize_to_noun_chunk = True)

    # assert isinstance(normalized_token, Span)
    # assert normalized_token.text == 'Mette Frederiksen'


def test_headword():

    nlp = spacy.load("en_core_web_lg")

    test_sents = ["Mette Frederiksen is the Danish politician."]

    config = {"confidence_threshold": 2.7, "model_args": {"batch_size": 10}}
    nlp.add_pipe("relation_extractor", config=config)
    nlp.add_pipe("heads_extraction")

    pipe = nlp.pipe(test_sents)

    for d in pipe:
        for span in d._.relation_head:
            headword = span._.most_common_ancestor

            assert isinstance(headword, Span)
            assert headword.text == "Frederiksen"