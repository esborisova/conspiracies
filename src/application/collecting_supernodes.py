import nltk
from nltk.tokenize import word_tokenize
import pandas as pd
from operator import itemgetter
from supernodes import preprocess_seeds, find_most_freq_substring

stemmer = nltk.stem.SnowballStemmer("english")

df = pd.read_csv("Bridgegate.csv")

entities = df["rank"].tolist()
tokens = [word_tokenize(entity) for entity in entities]
stems = [
    list(map(lambda token: stemmer.stem(token), list_of_tokens))
    for list_of_tokens in tokens
]
joined_stems = [" ".join(stem) for stem in stems]
df["stems"] = joined_stems

current_list = df.values.tolist()
sorted_current_list = sorted(current_list, key=itemgetter(2), reverse=True)

supernodes = []
max_seeds = 4

while len(sorted_current_list) > 0:

    seeds = []

    seeds.append(sorted_current_list[0][0])
    sorted_current_list.pop(0)

    seeds_stems = preprocess_seeds(seeds, stemmer)
    substrings = [
        entity
        for seed in seeds_stems
        for entity in sorted_current_list
        if seed in entity[3]
    ]

    if len(substrings) != 0:

        most_freq_substring = find_most_freq_substring(substrings)

        for item in reversed(sorted_current_list):
            if item == most_freq_substring:
                seeds.append(most_freq_substring[0])
                sorted_current_list.pop(sorted_current_list.index(most_freq_substring))

                # if len(seeds) == max_seeds:
                #    supernodes.append(seeds)
                #    return to line 23 (this supernode is complere. Start collecting seeds for a new supernode)
                # else:
                #    jump to line 31 (continue collecting substrings)

            else:
                supernodes.append(seeds)
                # start from line 23 again (new supernode)

    else:
        supernodes.append(seeds)
        # start from line 23 again (new supernode)
