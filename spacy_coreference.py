from typing import Dict, Iterable, Iterator, List

import spacy
from allennlp.models.archival import load_archive
from allennlp_models.coref.predictors.coref import CorefPredictor
from spacy import Vocab
from spacy.language import Language
from spacy.pipeline import TrainablePipe
from spacy.tokens import Doc, Span
from spacy.util import minibatch

from src.coref import CoreferenceModel


# Give a path to the model directory
def load_custom_predictor(
    model_path="./conlldata_danish_twitter_lr1_lr1/ser_folder/", cuda_device: int = -1
) -> CoreferenceModel:
    """Load coference model

    Args:
        model_path (str, optional): _description_. Defaults to "./conlldata_danish_twitter_lr1_lr1/ser_folder/".
        device (int, optional): Cuda device. If >= 0 will use the corresponding GPU, below 0 is CPU. Defaults to -1.

    Returns:
        CorefenceModel:
    """

    archive = load_archive(model_path, cuda_device=cuda_device)
    my_model = archive.model
    dataset_reader = archive.validation_dataset_reader
    predictor = CoreferenceModel(model=my_model, dataset_reader=dataset_reader)
    return predictor


# Edited from the spaCy website (https://spacy.io/usage/processing-pipelines)
@Language.factory(
    "Allennlp_coref",
    default_config={
        "model_path": "/home/lasse/conspiracies/conlldata_danish_twitter_lr1_lr1/ser_folder/",
        "cuda_device": -1,
    },
)
def create_coref_component(nlp: Language, name: str, model_path: str, cuda_device: int):
    return CoreferenceExtension(
        nlp.vocab, name=name, model_path=model_path, cuda_device=cuda_device
    )


class CoreferenceExtension(TrainablePipe):
    def __init__(self, vocab: Vocab, name: str, model_path: str, cuda_device: int):

        self.name = name
        self.vocab = vocab
        self.model = load_custom_predictor(
            model_path=model_path, cuda_device=cuda_device
        )

        self.predictor = load_custom_predictor()
        # Register custom extension on the Doc and Span
        if not Doc.has_extension("coref_chains"):
            Doc.set_extension("coref_chains", default=[])
        if not Span.has_extension("coref_chains"):
            Span.set_extension("coref_chains", default=[])
        if not Doc.has_extension("coref_clusters"):
            Doc.set_extension("coref_clusters", default=[])

    def set_annotations(self, docs: Iterable[Doc], model_output) -> None:
        """Set the coref attributes on Doc and Token level
        Args:
            docs (Iterable[Doc]): The documents to modify.
            model_output: (Dict): A batch of outputs from KnowledgeTriplets.extract_relations().
        """
        for doc, prediction in zip(docs, model_output):
            clusters = prediction["clusters"]
            doc._.coref_clusters.append(clusters)

            coref_chains = [
                (
                    clusters.index(d),
                    [doc[cluster_ids[0] : cluster_ids[1] + 1] for cluster_ids in d],
                )
                for d in clusters
            ]
            doc._.coref_chains.append(coref_chains)
            for span in doc.sents:
                for cluster, corefs in coref_chains:
                    for coref in corefs:
                        if span == coref.sent:
                            span._.coref_chains.append(coref)

    def __call__(self, doc: Doc) -> Doc:
        """Apply the pipe to one document. The document is modified in place,
        and returned. This usually happens under the hood when the nlp object
        is called on a text and all components are applied to the Doc.
        docs (Doc): The Doc to process.
        RETURNS (Doc): The processed Doc.
        DOCS: https://spacy.io/api/transformer#call
        """
        outputs = self.predict([doc])
        self.set_annotations([doc], outputs)
        # Add the extracted coreference clusters to the Doc and Span

        ## get the text

        # clusters = self.predictor.predict(doc.text)['clusters']
        # coref_chains = [[doc[cluster_ids[0]:cluster_ids[1]+1] for cluster_ids in d] for d in clusters]
        # doc._.clusters = clusters
        # doc._.coref_chains.append(coref_chains)
        # for span in doc.sents:
        #     for corefs in coref_chains:
        #         for coref in corefs:
        #             if span == coref.sent:
        #                 span._.coref_chains.append(coref)
        return doc

    def pipe(self, stream: Iterable[Doc], *, batch_size: int = 128) -> Iterator[Doc]:
        """Apply the pipe to a stream of documents. This usually happens under
        the hood when the nlp object is called on a text and all components are
        applied to the Doc. Batch size is controlled by `batch_size` when
        instatiating the nlp.pipe object.
        stream (Iterable[Doc]): A stream of documents.
        batch_size (int): The number of documents to buffer.
        YIELDS (Doc): Processed documents in order.
        DOCS: https://spacy.io/api/transformer#pipe
        """
        for outer_batch in minibatch(stream, batch_size):
            outer_batch = list(outer_batch)
            self.set_annotations(outer_batch, self.predict(outer_batch))

            yield from outer_batch

    def predict(self, docs: Iterable[Doc]) -> Dict:
        """Apply the pipeline's model to a batch of docs, without modifying them.
        Returns the extracted features as the FullTransformerBatch dataclass.
        docs (Iterable[Doc]): The documents to predict.
        RETURNS (Dict): The extracted features.
        DOCS: https://spacy.io/api/transformer#predict
        """

        return self.model.predict_batch_docs(docs)


if __name__ == "__main__":
    # Add the component to the pipeline and configure it
    nlp = spacy.load("da_core_news_trf")
    nlp.add_pipe("Allennlp_coref")

    text = "Aftalepartierne bag Rammeaftalen om plan for genåbning af Danmark blev i foråret 2021 enige om at nedsætte en ekspertgruppe, der fik til opgave at komme med input til den langsigtede strategi for håndtering af coronaepidemien i Danmark. Ekspertgruppen er nu klar med sin rapport."
    # text = "Lasse er en sød fyr. Han bor i Aarhus"

    print(text)
    print("\n")
    doc = nlp(text)
    doc._.coref_chains
    for sent in doc.sents:
        print(sent)
        print(sent._.coref_chains)
