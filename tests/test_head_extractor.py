import spacy
from spacy.tokens import Doc, Span, Token
from relationextraction import SpacyRelationExtractor
from collections import Counter
from heads_extract_component import HeadwordsExtraction


def test_head_extractor():

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
