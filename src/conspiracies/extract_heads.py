"""
Functions for headwords extractions from a span
"""

import spacy
from spacy.tokens import Span, Token, Doc
from collections import Counter
from warnings import warn
from typing import Callable


def normalize_token_to_span(token: Token, normalize_to_entity: bool = False) -> Span:
    """
    Normalize token to a span.

    Args:
        token(Token): The token to normalize.

        normalize_to_entity(bool, optional): If a token is an entity returns a token normilized to an entity.

    Returns:
        Span: The normalized token.
    """

    if normalize_to_entity and token.ent_type:
        for ent in token.doc.ents:
            if token in ent:
                return ent
    else:
        doc = token.doc
        return doc[token.i : token.i + 1]


def most_common_ancestor(span: Span, raise_error: bool = False) -> Span:
    """
    Find the most common ancestor in a span.

    Args:
        span(Span): The span to find the most common ancestor of.

        raise_error(bool): Raises warning message if no ancestor is found within a span.

    Returns:
        Span: The most common ancestor of the span.
    """
    ancestors_in_span = Counter(
        [ancestor for token in span for ancestor in token.ancestors if ancestor in span]
    )
    most_common_ancestor = ancestors_in_span.most_common()[0][0]

    normalized_token = normalize_token_to_span(most_common_ancestor)

    if len(normalized_token) != 1:
        error_message = f"None of the tokens in the span ({span}) contains an ancestor within this span."

        if raise_error:
            warn(error_message)

    return normalized_token


def set_extensions(extention_name: str, extention: Callable, levels: list):
    """
    Set a custom attribute on a Doc, Span or Token level which becomes available via Level._.

    Args:
        extention_name(str): The name of an attribute to be added.

        extention(Callable): An attribute (function) to be added.

        level(list): A list with levels to which an attribute should be added.
    """
    for level in levels:
        if not level.has_extension(extention_name):
            if level == Token or level == Span:
                level.set_extension(extention_name, getter=extention)
            else:
                level.set_extension(
                    extention_name, getter=lambda doc: extention(doc[:])
                )