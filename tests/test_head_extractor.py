from .utils import nlp_en  # noqa

from spacy.tokens import Span

import conspiracies  # noqa


def test_most_common_ancestor(nlp_en):
    nlp_en.add_pipe("heads_extraction")
    doc = nlp_en("Mette Frederiksen is the Danish politician.")

    assert doc[0:1]._.most_common_ancestor.text == "Mette"  # Single token Span
    assert doc[0:2]._.most_common_ancestor.text == "Frederiksen"  # Span
    assert doc._.most_common_ancestor.text == "is"  # Doc
