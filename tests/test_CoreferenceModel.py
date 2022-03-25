import spacy
from spacy.tokens import Doc

from conspiracies.coref.CoreferenceModel import CoreferenceModel


def test_CoreferenceModel():
    model = CoreferenceModel()  # check that the model loads as intended

    
    nlp = spacy.load("da_core_news_sm")
    text = [
        "Hej Kenneth, har du en fed teksts vi kan skrive om dig?",
        "Ja, det kan du tro min fine ven.",
    ]
    docs = nlp.pipe(text)

    outputs = model.predict_batch_docs(docs)

    for output in outputs:
        assert isinstance(output, dict)

