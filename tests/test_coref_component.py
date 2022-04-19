from spacy.tokens import Span

from .utils import nlp_da  # noqa

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

    resolve_coref_text = (
        "Aftalepartierne bag Rammeaftalen om plan for genåbning af Danmark blev i"
        + " foråret 2021 enige om at nedsætte en ekspertgruppe, en ekspertgruppe fik til opgave at "
        + "komme med input til den langsigtede strategi for håndtering af "
        + "coronaepidemien i Danmark. en ekspertgruppe er nu klar med en ekspertgruppe rapport."
    )

    resolve_coref_spans = [
        "Aftalepartierne bag Rammeaftalen om plan for genåbning af Danmark",
        "blev i foråret 2021 enige om at nedsætte en ekspertgruppe, en ekspertgruppe fik til opgave at komme med input til den langsigtede strategi for håndtering af coronaepidemien i Danmark. ",
        "en ekspertgruppe er nu klar med en ekspertgruppe rapport.",
    ]

    doc = nlp_da(text)
    # test attributes is set a
    assert isinstance(doc._.coref_clusters, list)
    assert doc._.resolve_coref == resolve_coref_text
    for i, sent in enumerate(doc.sents):
        assert isinstance(sent._.coref_clusters, list)
        if sent._.coref_clusters != []:
            assert sent._.resolve_coref == resolve_coref_spans[i]
            assert isinstance(sent._.coref_clusters[0], tuple)
            assert isinstance(sent._.coref_clusters[0][0], int)
            assert isinstance(sent._.coref_clusters[0][1], Span)
