from relationextraction import SpacyRelationExtractor
import spacy
from collections import Counter
import pandas as pd
from functions import*
import pytest


def test_heads_ents_extractor():
    nlp = spacy.load("da_core_news_lg")
    
    test_sents = [
        "Hurtigst var til gengæld hollænderen Ranomi Kromowidjojo, der sikrede sig guldet i tiden 23,97 sekunder.",
    ]
    
    
    config = {"confidence_threshold": 2.7, "model_args": {"batch_size": 10}}
    nlp.add_pipe("relation_extractor", config=config)
    
    pipe = nlp.pipe(test_sents)
    
    args = []
    
    for d in pipe:
        args.append(d._.relation_head)
        args.append(d._.relation_tail)
    
    assert get_headword(args) == ['sekunder%%', 'sekunder%%']

    assert get_entities(args) == ['23,97%%MISC'] 



def test_ne_filter():

    ent_dict = {'sekunder%%': 2, '23,97%%MISC': 1}

    assert filter_ne_type(ent_dict) == {'23,97%%MISC': 1}



def test_tuples_creator():

    ent_dict = {'23,97%%MISC': 1}

    assert make_tuples(ent_dict) == [('23,97', 'MISC', 1)]








   

                
    
 





 
     

