from .utils import nlp_en  # noqa

from spacy.tokens import Span

import conspiracies  # noqa
from relationextraction import SpacyRelationExtractor  # noqa


def test_head_extractor(nlp_en):

    test_sents = ["Mette Frederiksen is the Danish politician."]
    config = {"confidence_threshold": 2.7, "model_args": {"batch_size": 10}}

    nlp_en.add_pipe("relation_extractor", config=config)
    nlp_en.add_pipe("heads_extraction")

    pipe = nlp_en.pipe(test_sents)

    for d in pipe:
        for span in d._.relation_head:
            headword = span._.most_common_ancestor

            assert isinstance(headword, Span)
            assert headword.text == "Frederiksen"
