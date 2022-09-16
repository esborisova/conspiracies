from operator import itemgetter
from typing import List, Tuple

from nltk.tokenize import word_tokenize


def preprocess_seeds(seeds_list: List[str], stemmer) -> List[str]:

    seeds_tokens = [word_tokenize(seed) for seed in seeds_list]
    seeds_stems = [
        list(map(stemmer.stem, list_of_tokens)) for list_of_tokens in seeds_tokens
    ]
    joined_seeds_stems = [" ".join(seed) for seed in seeds_stems]

    return joined_seeds_stems


def find_most_freq_substring(
    list_of_substrings: List[Tuple[str, str, int]]
) -> Tuple[str, str, int]:

    sorted_list = sorted(list_of_substrings, key=itemgetter(2), reverse=True)

    return sorted_list[0]
