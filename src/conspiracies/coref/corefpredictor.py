from typing import Iterable, List
from allennlp.predictors.predictor import Predictor
from allennlp.data import DatasetReader, Instance
from spacy.tokens import Doc
from allennlp.models import Model
from allennlp_models.coref.dataset_readers.conll import ConllCorefReader
from allennlp.models.archival import load_archive
from allennlp.common.util import import_module_and_submodules, prepare_environment


@Predictor.register("coreference_resolution_v1")
class CoreferenceModel(Predictor):
    def __init__(self, model: Model, dataset_reader: DatasetReader) -> None:
        super().__init__(model, dataset_reader)

    def predict_batch_docs(self, docs: List[Doc]) -> List[Instance]:
        """Convert a list of docs to Instance and predict the batch

        Args:
            docs (List[Doc]): _description_

        Returns:
            _type_: _description_
        """
        instances = [self._doc_to_instance(doc) for doc in docs]
        return self.predict_batch_instance(instances)

    def _doc_to_instance(self, doc: Doc) -> Instance:
        sentences = [[token.text for token in sentence] for sentence in doc.sents]
        instance = self._dataset_reader.text_to_instance(sentences)
        return instance


def load_custom_predictor(
    model_path="/home/lasse/conspiracies/conlldata_danish_twitter_lr1_lr1/ser_folder/",
):
    archive = load_archive(model_path)
    config = archive.config
    prepare_environment(config)
    my_model = archive.model
    dataset_reader = archive.validation_dataset_reader
    predictor = CoreferenceModel(model=my_model, dataset_reader=dataset_reader)
    return predictor


if __name__ == "__main__":
    import spacy

    model = load_custom_predictor()

    nlp = spacy.load("da_core_news_sm")

    text = [
        "Hej Kenneth, har du en fed teksts vi kan skrive om dig?",
        "Ja, det kan du tro min fine ven.",
    ]
    docs = nlp.pipe(text)

    out = model.predict_batch_docs(docs)
