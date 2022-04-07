"""Headwords extraction as a spaCy component. """
import spacy
from spacy.tokens import Doc, Span, Token
from spacy.language import Language
from collections import Counter
from warnings import warn


@Language.factory("heads_extraction")
def create_headwords_component(nlp: Language, name: str):
    """
    Allows HeadwordsExtraction to be added to a spaCy pipe using nlp.add_pipe("heads_extraction").
    """
    return HeadwordsExtraction(nlp)


class HeadwordsExtraction:
    def __init__(self, nlp: Language):
        """Initialise components"""

        extensions = [
            "normalize_token_to_span",
            "most_common_ancestor",
            "contains_ents",
        ]

        functions = [
            self.normalize_token_to_span,
            self.most_common_ancestor,
            self.contains_ents,
        ]

        for extention, function in zip(extensions, functions):
            if extention == "normalize_token_to_span":
                if not Token.has_extension(extention):
                    Token.set_extension(extention, getter=function)
            else:
                if not Doc.has_extension(extention):
                    Doc.set_extension(extention, getter=lambda doc: function(doc[:]))
                if not Span.has_extension(extention):
                    Span.set_extension(extention, getter=function)

    def __call__(self, doc: Doc):
        """Run the pipeline component"""
        return doc

    def normalize_token_to_span(
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

        if normalize_to_entity and token.ent_type:
            for ent in token.doc.ents:
                if token in ent:
                    return ent

        if normalize_to_noun_chunk:
            doc = token.doc
            for noun_chunk in doc.noun_chunks:
                if token in noun_chunk:
                    return noun_chunk
        else:
            doc = token.doc
            return doc[token.i : token.i + 1]

    def most_common_ancestor(self, span: Span, raise_error: bool = False) -> Span:
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

        normalized_token = self.normalize_token_to_span(most_common_ancestor)

        if len(normalized_token) != 1:
            error_message = f"None of the tokens in the span ({span}) contains an ancestor within this span."

            if raise_error:
                warn(error_message)

        return normalized_token

    def contains_ents(self, span: Span) -> bool:
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