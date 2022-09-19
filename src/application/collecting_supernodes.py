import spacy
import pandas as pd
from conspiracies.supernodes import lemmatize

nlp = spacy.load("en_core_web_lg")

df = pd.read_csv("ranked_heads_ents_list.csv")

heads_and_entities = sorted(df.values.tolist(), key=lambda x: x[2])

supernodes = []
max_seeds = 4
set_of_seeds = set()

while len(heads_and_entities) > 0:

    if not set_of_seeds:
        seed = heads_and_entities.pop()[0]
        set_of_seeds.add(seed)

    phrases_with_seed = [
        (phrase, freq)
        for phrase, tag, freq in heads_and_entities
        if any(
            lemmatize(seed, nlp=nlp) in lemmatize(phrase, nlp=nlp)
            for seed in set_of_seeds
        )
    ]

    if phrases_with_seed:
        most_freq_substr = max(phrases_with_seed, key=lambda x: x[1])[0]
    else:
        supernodes.append(set_of_seeds)
        set_of_seeds = set()
        continue

    set_of_seeds.add(most_freq_substr)
    heads_and_entities = [
        entity for entity in heads_and_entities if entity[0] != most_freq_substr
    ]

    if len(set_of_seeds) == max_seeds:

        supernodes.append(set_of_seeds)
        set_of_seeds = set()
        continue
