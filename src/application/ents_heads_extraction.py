from relationextraction import SpacyRelationExtractor
import spacy
from spacy.tokens import Span
from collections import Counter
from extract_heads import most_common_ancestor, extract_entities, contains_ents

nlp = spacy.load("en_core_web_lg")

test_sents = ["Mette Frederiksen is the Danish politician."]

config = {"confidence_threshold": 2.7, "model_args": {"batch_size": 10}}
nlp.add_pipe("relation_extractor", config=config)

pipe = nlp.pipe(test_sents)

if not Span.has_extension("most_common_ancestor"):
    Span.set_extension("most_common_ancestor", getter=most_common_ancestor)


if not Span.has_extension("extract_entities"):
    Span.set_extension("extract_entities", getter=extract_entities)


heads_spans = []
ents_spans = []

for d in pipe:
    for span in d._.relation_head:
        heads_spans.append(span._.most_common_ancestor)
        ents_spans.append(span._.extract_entities)
    for span in d._.relation_tail:
        heads_spans.append(span._.most_common_ancestor)
        ents_spans.append(span._.extract_entities)

# Filter out headwords without an entity type
filtered_heads = list(filter(contains_ents, heads_spans))

print(filtered_heads)
print(ents_spans)
