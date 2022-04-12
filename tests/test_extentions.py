import spacy
from spacy.tokens import Span, Doc

import conspiracies  # noqa


def test_extentions():
    nlp = spacy.blank("en")
    nlp.add_pipe("heads_extraction")

    doc = Doc(nlp.vocab, words="Mette Frederiksen is the Danish Politician".split())
    doc.set_ents([Span(doc, 0, 2, "PERSON")])

    assert isinstance(doc._.most_common_ancestor, Span)  # Doc
    assert isinstance(doc[0:2]._.most_common_ancestor, Span)  # Span
