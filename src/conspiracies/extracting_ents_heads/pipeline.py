from relationextraction import SpacyRelationExtractor
import spacy
from collections import Counter
import pandas as pd
from functions import*

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

args = []

#Collect all args1 and args2 
for d in pipe:

  args.append(d._.relation_head)
  args.append(d._.relation_tail)

#Extract all headwords and their entaty lables from args 
heads = get_headword(args)

#Count heads frequency
heads_freq = Counter(heads)

#Extract all entaties and their labels from args
entities = get_entities(args)

#Count entaties frequency
entities_freq = Counter(entities)

#Sum frequencies across heads and entaties
merged_freq = {k: heads_freq.get(k, 0) + entities_freq.get(k, 0) for k in set(heads_freq) | set(entities_freq)}

#Filter out enataty types 
filtered_freq = filter_ne_type(merged_freq)

#Rank entaties/heads by frequency 
ranked_freq = {key: value for key, value in sorted(filtered_freq.items(), key=lambda item: item[1],reverse=True)}

#Create a list of tuples with ents/heads, type tag, freq 
ents_tuples = make_tuples(ranked_freq)
