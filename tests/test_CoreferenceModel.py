import spacy
from conspiracies.coref.CoreferenceModel import CoreferenceModel


def test_CoreferenceModel():
    model = CoreferenceModel()
    nlp = spacy.load("da_core_news_sm")
    text = [
        "Hej Kenneth, har du en fed teksts vi kan skrive om dig?",
        "Ja, det kan du tro min fine ven.",
    ]
    docs = nlp.pipe(text)

    out = model.predict_batch_docs(docs)
