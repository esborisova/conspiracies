"""
Pipeline for headwords/entities extractions and frequency count
"""

from relationextraction import SpacyRelationExtractor
import spacy
from spacy.tokens import Span
from collections import Counter
from extract_heads import most_common_ancestor, set_extensions


nlp = spacy.load("en_core_web_lg")

test_sents = ["Mette Frederiksen is the Danish politician."]

config = {"confidence_threshold": 2.7, "model_args": {"batch_size": 10}}
nlp.add_pipe("relation_extractor", config=config)

pipe = nlp.pipe(test_sents)

set_extensions(
    extention_name="most_common_ancestor", extention=most_common_ancestor, levels=[Span]
)

heads_spans = []

for d in pipe:
    for span in d._.relation_head:
        heads_spans.append(span._.most_common_ancestor)
    for span in d._.relation_tail:
        heads_spans.append(span._.most_common_ancestor)

# Filter out headwords without an entity type
# filtered_heads = list(filter(contains_ents, heads_spans))