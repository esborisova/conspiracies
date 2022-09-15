"""
Pipeline for headwords/entities extractions and frequency count
"""
import spacy

import pandas as pd
from relationextraction import SpacyRelationExtractor  # noqa
from conspiracies.HeadWordExtractionComponent import contains_ents, get_entity_label


def main():

    nlp = spacy.load("en_core_web_lg")

    test_sents = ["Mette Frederiksen is the Danish politician."]

    config = {"confidence_threshold": 2.7, "model_args": {"batch_size": 10}}
    nlp.add_pipe("relation_extractor", config=config)
    nlp.add_pipe("heads_extraction")

    docs = nlp.pipe(test_sents)

    heads_spans = []
    ents_spans = []

    for d in docs:
        for span in d._.relation_head:
            heads_spans.append(span._.most_common_ancestor)
            if span.ents:
                ents_spans.append(list(span.ents))
        for span in d._.relation_tail:
            heads_spans.append(span._.most_common_ancestor)
            if span.ents:
                ents_spans.append(list(span.ents))

    # Filter out headwords without an entity type
    filtered_heads = list(filter(contains_ents, heads_spans))

    # Flatten the list with ents
    flat_ents_list = [ent for sublist in ents_spans for ent in sublist]

    # Generate a frequency-ranked list of extracted heads/entities
    ents_heads = [filtered_heads, flat_ents_list]
    df = pd.DataFrame(columns=["head/entity", "entity_label"])

    for item in ents_heads:

        data = pd.DataFrame()

        span_to_text = [span.text for span in item]
        ents_tags = [get_entity_label(span) for span in item]
        data["head/entity"] = span_to_text
        data["entity_label"] = ents_tags
        data = (
            data.groupby(["head/entity", "entity_label"])
            .size()
            .reset_index(name="frequency")
        )
        df = df.append(data, ignore_index=True)

    df = df.groupby(["head/entity", "entity_label"])["frequency"].sum().reset_index()
    df = df.sort_values(by=["frequency"], ascending=False, ignore_index=True)
    df.to_csv("ranked_heads_ents_list.csv", index=False)


if __name__ == "__main__":
    main()
