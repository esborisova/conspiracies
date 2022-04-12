import pytest
import spacy


@pytest.fixture()
def nlp_en():
    return spacy.load("en_core_web_sm")


@pytest.fixture()
def nlp_da():
    return spacy.load("da_core_news_sm")
