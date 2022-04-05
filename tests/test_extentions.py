from extract_heads import (
    most_common_ancestor,
    set_extensions,
)
import spacy
from spacy.tokens import Doc, Span, Token
from relationextraction import SpacyRelationExtractor
from collections import Counter


def test_extentions():

    set_extensions(
        extention_name="most_common_ancestor",
        extention=most_common_ancestor,
        levels=[Doc, Span, Token],
    )

    assert Doc.has_extension("most_common_ancestor")
    assert Span.has_extension("most_common_ancestor")
    assert Token.has_extension("most_common_ancestor")