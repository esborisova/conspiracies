from spacy.tokens import Span

from .utils import nlp_da

from conspiracies.coref import CoreferenceComponent  # noqa


def test_coref_component(nlp_da):
    # test adding pipe works:
    nlp_da.add_pipe("allennlp_coref")

    text = (
        "Aftalepartierne bag Rammeaftalen om plan for genåbning af Danmark blev i"
        + " foråret 2021 enige om at nedsætte en ekspertgruppe, der fik til opgave at "
        + "komme med input til den langsigtede strategi for håndtering af "
        + "coronaepidemien i Danmark. Ekspertgruppen er nu klar med sin rapport."
    )

    doc = nlp(text)
    # test attributes is set a
    assert isinstance(doc._.coref_chains, list)
    for sent in doc.sents:
        assert isinstance(sent._.coref_chains, list)
        assert isinstance(sent._.coref_chains[0], Span)
