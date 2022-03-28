import spacy
from collections import Counter


def get_headword(noun_phrases: list) -> list:
  """Extracts headwords with their entity labels from noun phrases
   Args:
       noun_phrases (list): A list of lists with strings (noun phrases) 
   Returns:
       headwords (list): A list of strings containing headwords and their entity types
  """ 

  headwords = []

  for phrase in noun_phrases:
    for span in phrase:
      for word in span:
        if word.head.pos_ == "PROPN" or word.head.pos_ == "NOUN" or word.head.pos_ == "PRON":
          headwords.append(f"{word.head}%%{word.head.ent_type_}") 
  return headwords
          


def get_entities(noun_phrases: list) -> list:
  """Extracts entities with their named entity labels from noun phrases
   Args:
       noun_phrases (list): A list of lists with strings (noun phrases) 
   Returns:
       entities (list): A list of strings containing entities and their types
  """ 

  entities = []

  for phrase in noun_phrases:
    for span in phrase:
      if span.ents:
        entities.append(f"{span.ents[0]}%%{span.ents[0][0].ent_type_}")
  return entities



def filter_ne_type(ents_heads: dict) -> dict:
  """Narrows down entities/headwords to the predifiend list of named entity types
   Args:
       ents_heads (dict): A dictionary with entities/headwords as keys and their frequencies as values 
   Returns:
       new_dict (list): A dictionary containing only entities/headwords (and their freq) belonging to the defined group of entity types 
  """ 

  ent_types = ['LOC', 'MISC', 'ORG', 'PER']

  new_dict = {}  
  
  for tag in ent_types:
    for key, value in ents_heads.items():
      if tag in key:
        new_dict[key] = value
  return new_dict



def make_tuples(ranked_ents_heads: dict) -> list:
  """Creates a list of tuples with: Entity/headword, its named entity type, its frequency
   Args:
       ranked_ents_heads (dict): A dictionary with entities/headwords as keys and their frequencies as values 
   Returns:
       list_of_tuples (list): A list of tuples containing entities/headwords, their entity type label and frequency 
  """ 
  
  list_of_tuples = []
  
  for key, value in ranked_ents_heads.items():
    splitted_key = key.split("%%")
    splitted_key.append(value)
    list_of_tuples.append(tuple(splitted_key))
   
  return list_of_tuples
