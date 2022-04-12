"""Headwords extraction as a spaCy component. """
import spacy
from spacy.tokens import Doc, Span, Token
from typing import Union
from spacy.language import Language
from collections import Counter
from warnings import warn


@Language.factory(
    "heads_extraction",
    default_config={
        "raise_error": False,
        "normalize_to_entity": False,
        "normalize_to_noun_chunk": False,
    },
)
def create_headwords_component(
    nlp: Language,
    name: str,
    raise_error: bool,
    normalize_to_entity: bool,
    normalize_to_noun_chunk: bool,
):
    """
    Allows HeadwordsExtraction to be added to a spaCy pipe using nlp.add_pipe("heads_extraction").
    """
    return HeadwordsExtraction(
        nlp,
        name=name,
        raise_error=raise_error,
        normalize_to_entity=normalize_to_entity,
        normalize_to_noun_chunk=normalize_to_noun_chunk,
    )


class HeadwordsExtraction:
    def __init__(
        self,
        nlp: Language,
        name: str,
        raise_error: bool,
        normalize_to_entity: bool,
        normalize_to_noun_chunk: bool,
        force: bool = True
    ):
        """Initialise components"""

        self.raise_error = raise_error
        self.normalize_to_entity = normalize_to_entity
        self.normalize_to_noun_chunk = normalize_to_noun_chunk

        if not Token.has_extension("to_span") or force:
            Token.set_extension("to_span", getter=self.to_span, force=force)
        if not Span.has_extension("to_span") or force:
            Span.set_extension("to_span", getter=lambda span: span, force=force)
        if not Doc.has_extension("to_span") or force:
            Doc.set_extension("to_span", getter=lambda doc: doc[:], force=force)

        if not Doc.has_extension("most_common_ancestor") or force:
            Doc.set_extension(
                "most_common_ancestor",
                getter=lambda doc: self.most_common_ancestor(doc[:], force=force),
            )
        if not Span.has_extension("most_common_ancestor") or force:
            Span.set_extension("most_common_ancestor", getter=self.most_common_ancestor, force=force)

    def __call__(self, doc: Doc):
        """Run the pipeline component"""
        return doc

    def to_entity(self, token: Token) -> Span:
        """
        Normalize token to an entity.

        Args:

            token(Token): The token to normalize.

        Returns:
            Span: The entity.
        """

        for ent in token.doc.ents:
            if token in ent:
                return ent

    def to_noun_chunk(self, token: Token) -> Span:
        """
        Normalize token to a noun chunk.

        Args:

            token(Token): The token to normalize.

        Returns:
            Span: The noun chunk.
        """

        doc = token.doc
        for noun_chunk in doc.noun_chunks:
            if token in noun_chunk:
                return noun_chunk

    def to_span(
        self,
        token: Token,
        normalize_to_entity: bool = False,
        normalize_to_noun_chunk: bool = False,
    ) -> Span:
        """
        Normalize token to a span.

        Args:

            token(Token): The token to normalize.

            normalize_to_entity(bool, optional): If a token is an entity returns a token normilized to an entity.

            normalize_to_noun_chunk(bool, optional): If True, returns a base noun phrase which a token is part of.

        Returns:
            Span: The normalized token.
        """

        if self.normalize_to_entity:
            ent = self.to_entity(token)
            return ent

        if self.normalize_to_noun_chunk:
            noun_chunk = self.to_noun_chunk(token)
            return noun_chunk

        else:
            doc = token.doc
            return doc[token.i : token.i + 1]

    def most_common_ancestor(
        self, span: Union[Doc, Span], raise_error: bool = False
    ) -> Span:
        """
        Find the most common ancestor in a span.

        Args:
           span(Span): The span to find the most common ancestor of.

           raise_error(bool): Raises warning message if no ancestor is found within a span.

        Returns:
            Span: The most common ancestor of the span.
        """
        ancestors_in_span = Counter(
            [
                ancestor
                for token in span
                for ancestor in token.ancestors
                if ancestor in span
            ]
        )
        most_common_ancestor = ancestors_in_span.most_common()[0][0]

        normalized_token = self.to_span(most_common_ancestor)

        if len(normalized_token) != 1:
            error_message = f"None of the tokens in the span ({span}) contains an ancestor within this span."

            if raise_error:
                warn(error_message)

        return normalized_token


def contains_ents(span: Union[Doc, Span]) -> bool:
    """
    Check if a span contains entities.

    Args:
        span (Span): The span to find entites in.

    Returns:
        bool: Returns True if a token has an entity label.
    """

    for token in span:
        if token.ent_type:
            return True
    return False
