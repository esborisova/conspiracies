from spacy.tokens import Span

from .utils import nlp_da_w_coref, nlp_da  # noqa

from conspiracies.coref import CoreferenceComponent  # noqa


def test_coref_clusters(nlp_da_w_coref):

    text = (
        "Aftalepartierne bag Rammeaftalen om plan for genåbning af Danmark blev i"
        + " foråret 2021 enige om at nedsætte en ekspertgruppe, der fik til opgave at "
        + "komme med input til den langsigtede strategi for håndtering af "
        + "coronaepidemien i Danmark. Ekspertgruppen er nu klar med sin rapport."
    )

    doc = nlp_da_w_coref(text)
    # test attributes is set as intended
    assert isinstance(doc._.coref_clusters, list)
    for sent in doc.sents:
        assert isinstance(sent._.coref_clusters, list)
        assert isinstance(sent._.coref_clusters[0][1], Span)


def test_resolve_coref(nlp_da_w_coref):
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

    doc = nlp_da_w_coref(resolve_coref_text)
    # test for doc
    assert doc._.resolve_coref == resolve_coref_text

    # test for spans
    for i, sent in enumerate(doc.sents):
        if sent._.coref_clusters != []:
            assert sent._.resolve_coref == resolve_coref_spans[i]
