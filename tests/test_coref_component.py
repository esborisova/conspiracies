import spacy
from conspiracies.coref import CoreferenceComponent


def test_coref_component():
    nlp = spacy.load("da_core_news_sm")
    nlp.add_pipe("allennlp_coref")

    text = (
        "Aftalepartierne bag Rammeaftalen om plan for genåbning af Danmark blev i"
        + " foråret 2021 enige om at nedsætte en ekspertgruppe, der fik til opgave at "
        + "komme med input til den langsigtede strategi for håndtering af "
        + "coronaepidemien i Danmark. Ekspertgruppen er nu klar med sin rapport."
    )

    print(text)
    print("\n")
    doc = nlp(text)
    doc._.coref_chains
    for sent in doc.sents:
        print(sent)
        print(sent._.coref_chains)
