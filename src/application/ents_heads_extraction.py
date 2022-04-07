"""
Pipeline for headwords/entities extractions and frequency count
"""

from relationextraction import SpacyRelationExtractor
from heads_extract_component import HeadwordsExtraction
import spacy
from spacy.tokens import Span
from collections import Counter


nlp = spacy.load("en_core_web_lg")

test_sents = ["Mette Frederiksen is the Danish politician."]

config = {"confidence_threshold": 2.7, "model_args": {"batch_size": 10}}
nlp.add_pipe("relation_extractor", config=config)
nlp.add_pipe("heads_extraction")

pipe = nlp.pipe(test_sents)

heads_spans = []

for d in pipe:
    for span in d._.relation_head:
        heads_spans.append(span._.most_common_ancestor)
    for span in d._.relation_tail:
        heads_spans.append(span._.most_common_ancestor)

# Filter out headwords without an entity type
# filtered_heads = list(filter(contains_ents, heads_spans))