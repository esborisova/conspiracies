import spacy
from spacy.tokens import Span, Token
from collections import Counter


def normalize_token_to_span(token: Token) -> Span:
    """
    Normalize token to a span. If the token is an entity return the entity.

    Args:
        token(Token): The token to normalize.

    Returns:
        Span: The normalized token.
    """
    
    doc = token.doc
    
    return doc[token.i:token.i + 1]


def most_common_ancestor(span: Span) -> Span:
    """
    Find the most common ancestor in a span.

    Args:
        span(Span): The span to find the most common ancestor of.

    Returns:
        Span: The most common ancestor of the span.
    """
    ancestors_in_span = Counter([ancestor for token in span for ancestor
                                 in token.ancestors if ancestor in span])
    most_common_ancestor = ancestors_in_span.most_common()[0][0]

    normalized_token = normalize_token_to_span(most_common_ancestor)
 
    #Check that a span contains a single token 
    if len(normalized_token) !=1:
      raise ValueError 
    else:
      return normalized_token 



def extract_entities(span: Span) -> Span:
    """
    Find an entity in a span.
    
    Args:
        span (Span): The span to find an entity in.
    
    Returns:
        Span: An entity extracted from the span.
    """

    if span.ents:
      return span.ents[0]


def contains_ents(span: Span) -> bool:
  """
    Check if a token is an entity.
    
    Args:
        span (Span): The span with a token.
    
    Returns:
        bool: If a token is an entity returns True.
    """
    
  if span[0].ent_type_:
    return True
  else:
    return False 
