import spacy
from collections import Counter
from typing import List


def get_headword(noun_phrases: List[list],
                 pos_to_keep: List[str]) -> List[str]:
  """Extracts headwords with their entity labels from noun phrases
   Args:
       noun_phrases (List[list]): A list of lists with strings (noun phrases) 
       pos_to_keep (List[str]): A list with part of speech tags 

   Returns:
       List[str]: A list of strings containing headwords and their entity types
  """ 

  headwords = []

  for phrase in noun_phrases:
    for span in phrase:
      for word in span:
        if word.head.pos_ in pos_to_keep:
          headwords.append(f"{word.head}%%{word.head.ent_type_}") 
  return headwords
          


def get_entities(noun_phrases: List[str]) -> List[str]:
  """Extracts entities with their named entity labels from noun phrases
   Args:
       noun_phrases (List[str]): A list of lists with strings (noun phrases) 
    
   Returns:
       List[str]: A list of strings containing entities and their types
  """ 

  entities = []

  for phrase in noun_phrases:
    for span in phrase:
      if span.ents:
        entities.append(f"{span.ents[0]}%%{span.ents[0][0].ent_type_}")
  return entities



def filter_ne_type(ents_heads: dict,
                   ents_to_keep: List[str]) -> dict:
  """Narrows down entities/headwords to the predifiend list of named entity types
   Args:
       ents_heads (dict): A dictionary with entities/headwords as keys and their frequencies as values 
       ents_to_keep (List[str]): A list with named entity lables

   Returns:
       dict: A dictionary containing only entities/headwords (and their freq) belonging to the defined group of entity types 
  """ 

  new_dict = {}  
  
  for tag in ents_to_keep:
    for key, value in ents_heads.items():
      if tag in key:
        new_dict[key] = value
  return new_dict



def create_tuples(ents_heads: dict) -> List[tuple]:
  """Creates a list of tuples with: Entity/headword, its named entity type, its frequency
   Args:
       ents_heads (dict): A dictionary with entities/headwords as keys and their frequencies as values 
       
   Returns:
       List[tuple]: A list of tuples containing entities/headwords, their entity type label and frequency 
  """ 
  
  list_of_tuples = []
  
  for key, value in ents_heads.items():
    splitted_key = key.split("%%")
    splitted_key.append(value)
    list_of_tuples.append(tuple(splitted_key))
   
  return list_of_tuples
