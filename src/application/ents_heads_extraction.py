from relationextraction import SpacyRelationExtractor
import spacy
from collections import Counter
from extract_heads import get_headword, get_entities, filter_ne_type, create_tuples

nlp = spacy.load("da_core_news_lg")

test_sents = [
    "Pernille Blume vinder delt EM-sølv i Ungarn.",
    "Pernille Blume blev nummer to ved EM på langbane i disciplinen 50 meter fri.",
    "Hurtigst var til gengæld hollænderen Ranomi Kromowidjojo, der sikrede sig guldet i tiden 23,97 sekunder.",
    "Og at formen er til en EM-sølvmedalje tegner godt, siger Pernille Blume med tanke på, at hun få uger siden var smittet med corona.",
    "Ved EM tirsdag blev det ikke til medalje for den danske medley for mixede hold i 4 x 200 meter fri.",
    "Politiet skal etterforske Siv Jensen etter mulig smittevernsbrudd.",
    "En av Belgiens mest framträdande virusexperter har flyttats med sin familj till skyddat boende efter hot från en beväpnad högerextremist.",
]

config = {"confidence_threshold": 2.7, "model_args": {"batch_size": 10}}
nlp.add_pipe("relation_extractor", config=config)

pipe = nlp.pipe(test_sents)

# Collect all args1 and args2
args = []

for d in pipe:
    args.append(d._.relation_head)
    args.append(d._.relation_tail)

# Extract all headwords and their entity lables from args
heads = get_headword(noun_phrases=args, pos_to_keep=["PROPN", "NOUN", "PRON"])

# Count heads frequency
heads_freq = Counter(heads)

# Extract all entities and their labels from args
entities = get_entities(args)

# Count entities frequency
entities_freq = Counter(entities)

# Sum frequencies across heads and entities
merged_freq = {
    k: heads_freq.get(k, 0) + entities_freq.get(k, 0)
    for k in set(heads_freq) | set(entities_freq)
}

# Filter out entity types
filtered_freq = filter_ne_type(
    ents_heads=merged_freq, ents_to_keep=["LOC", "MISC", "ORG", "PER"]
)

# Rank entities/heads by frequency
ranked_freq = {
    key: value
    for key, value in sorted(
        filtered_freq.items(), key=lambda item: item[1], reverse=True
    )
}

# Create a list of tuples with ents/heads, type tag, freq
ents_tuples = create_tuples(ranked_freq)
