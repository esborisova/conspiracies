import nltk
import pandas as pd
from nltk.tokenize import word_tokenize

stemmer = nltk.stem.SnowballStemmer("english")

df = pd.read_csv("Bridgegate.csv")

entities = df["rank"].tolist()
tokens = [word_tokenize(entity) for entity in entities]
stems = [list(map(stemmer.stem, list_of_tokens)) for list_of_tokens in tokens]
joined_stems = [" ".join(stem) for stem in stems]

head_and_entities = sorted(df.values.tolist(), key=lambda x: x[2])

# Step-0: The current entity/concept list is set equal to the original list.
# The maximum number of seed nodes in asupernode isset to k.
current_list = head_and_entities  # TODO: remove if irrelevant or add copy
phrases = current_list  # TODO replace with actual phrases
supernodes = []
max_seeds = 4

S_seed_nodes = set()
while len(current_list) > 0:

    # Step-1: If the current list is empty, then Quit (supernode construction
    # iscomplete). Otherwise, select the highest ranked entity/concept in the current
    # list (in the first iteration, the entire original list isthe current list). Let
    # this entity be E_1. Add E_1 to the list of seed nodes for the new supernode, S.
    # Remove E_1 from the current list. Set the seed-node list size, |S|=1.
    if not S_seed_nodes:
        e_1 = current_list.pop()[0]
        S_seed_nodes.add(e_1)

    # Step-2: Find all phrases/arguments where any of the seed nodes in the
    # set S (i.e. the set representing the supernode under construction) appears as a
    # sub-string, and let this be called P.
    phrases_w_seed = [
        (phrase, freq)
        for phrase, tag, freq in phrases
        if any(seed in phrase for seed in S_seed_nodes)
    ]

    # Step-3: Compute the most frequent entity/concept in the original list (other than
    # the seed nodes already extracted) in P. Let this be E.
    if phrases_w_seed:
        E = max(phrases_w_seed, key=lambda x: x[1])[0]
    else:
        # Step-4: If E has been processed before (i.e., it is no longer in the current
        # list), then jump to Step-6.
        # Step-6: The current list of seed nodes, S, is the new supernode. Return to
        # Step-1 to start creating a new supernode.
        supernodes.append(S_seed_nodes)
        S_seed_nodes = set()
        continue

    # Step-5: If E is in the current list, then add it to the list of seed nodes, S.
    # Remove it from the current list of entities/concepts. Increase the size count,
    # |S| =|S|+ 1. If |S| = k (where k is the maximum size of the supernode seed list
    # S), then go to Step-6.
    S_seed_nodes.add(E)
    current_list = [entity for entity in current_list if entity[0] != E]

    if len(S_seed_nodes) == max_seeds:
        # Step-6: The current list of seed nodes, S, is the new supernode. Return to
        # Step-1 to start creating a new supernode.
        supernodes.append(S_seed_nodes)
        S_seed_nodes = set()
        continue

    # Otherwise jump to Step-2
