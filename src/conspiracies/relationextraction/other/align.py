"""Directly copied from the spacy-transformers library https://raw.githubusercontent.com/explosion/spacy-transformers/master/spacy_transformers/align.py
Pasted here to avoid clashing torch versions
"""

import numpy
from typing import cast, Dict, List, Tuple, Callable, Set, Optional
from spacy_alignments.tokenizations import get_alignments
from spacy.tokens import Span, Token
from thinc.types import Ragged, Floats2d


def get_token_positions(spans: List[Span]) -> Dict[Token, int]:
    token_positions: Dict[Token, int] = {}
    for span in spans:
        for token in span.doc:
            if token not in token_positions:
                token_positions[token] = len(token_positions)
    return token_positions


def get_alignment_via_offset_mapping(spans: List[Span], token_data) -> Ragged:
    # This function uses the offset mapping provided by Huggingface. I'm not
    # sure whether there's a bug here but I'm getting weird errors.
    # Tokens can occur more than once, and we need the alignment of each token
    # to its place in the concatenated wordpieces array.
    token_positions = get_token_positions(spans)
    alignment: List[Set[int]] = [set() for _ in range(len(token_positions))]
    wp_start = 0
    for i, span in enumerate(spans):
        for j, token in enumerate(span):
            position = token_positions[token]
            for char_idx in range(token.idx, token.idx + len(token)):
                wp_j = token_data.char_to_token(i, char_idx)
                if wp_j is not None:
                    alignment[position].add(wp_start + wp_j)
        wp_start += len(token_data.input_ids[i])
    lengths: List[int] = []
    flat: List[int] = []
    for a in alignment:
        lengths.append(len(a))
        flat.extend(sorted(a))
    align = Ragged(numpy.array(flat, dtype="i"), numpy.array(lengths, dtype="i"))
    return align


def get_alignment(
    spans: List[Span],
    wordpieces: List[List[str]],
    special_tokens: Optional[List[str]] = None,
) -> Ragged:
    """Compute a ragged alignment array that records, for each unique token in
    `spans`, the corresponding indices in the flattened `wordpieces` array.
    For instance, imagine you have two overlapping spans:

        [[I, like, walking], [walking, outdoors]]

    And their wordpieces are:

        [[I, like, walk, ing], [walk, ing, out, doors]]

    We want to align "walking" against [walk, ing, walk, ing], which have
    indices [2, 3, 4, 5] once the nested wordpieces list is flattened.

    The nested alignment list would be:

    [[0], [1], [2, 3, 4, 5], [6, 7]]
      I   like    walking    outdoors

    Which gets flattened into the ragged array:

    [0, 1, 2, 3, 4, 5, 6, 7]
    [1, 1, 4, 2]

    The ragged format allows the aligned data to be computed via:

    tokens = Ragged(wp_tensor[align.data], align.lengths)

    This produces a ragged format, indicating which tokens need to be collapsed
    to make the aligned array. The reduction is deferred for a later step, so
    the user can configure it. The indexing is especially efficient in trivial
    cases like this where the indexing array is completely continuous.
    """
    if len(spans) != len(wordpieces):
        raise ValueError("Cannot align batches of different sizes.")
    if special_tokens is None:
        special_tokens = []
    # Tokens can occur more than once, and we need the alignment of each token
    # to its place in the concatenated wordpieces array.
    token_positions = get_token_positions(spans)
    alignment: List[Set[int]] = [set() for _ in range(len(token_positions))]
    wp_start = 0
    for i, (span, wp_toks) in enumerate(zip(spans, wordpieces)):
        sp_toks = [token.text for token in span]
        wp_toks_filtered = wp_toks
        # In the case that the special tokens do not appear in the text, filter
        # them out for alignment purposes so that special tokens like "<s>" are
        # not aligned to the character "s" in the text. (If the special tokens
        # appear in the text, it's not possible to distinguish them from the
        # added special tokens, so they may be aligned incorrectly.)
        if not any([special in span.text for special in special_tokens]):
            wp_toks_filtered = [
                tok if tok not in special_tokens else "" for tok in wp_toks
            ]
        span2wp, wp2span = get_alignments(sp_toks, wp_toks_filtered)
        for token, wp_js in zip(span, span2wp):
            position = token_positions[token]
            alignment[position].update(wp_start + j for j in wp_js)
        wp_start += len(wp_toks)
    lengths: List[int] = []
    flat: List[int] = []
    for a in alignment:
        lengths.append(len(a))
        flat.extend(sorted(a))
    align = Ragged(numpy.array(flat, dtype="i"), numpy.array(lengths, dtype="i"))
    return align
