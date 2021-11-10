from typing import Dict, Optional, List, Any

import torch
import torch.nn.functional as F
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import FeedForward, TextFieldEmbedder, Seq2SeqEncoder
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn import util
from allennlp.training.metrics import CategoricalAccuracy, F1Measure
from overrides import overrides
from allennlp.training.metrics import FBetaMeasure
from allennlp.training.metrics.metric import Metric

from scibert.models.text_classifier import TextClassifier
from scibert.resources.metric import FBetaMeasureMultiLabel


@Model.register("bert_multilabel_text_classifier")
class BertMultilabelTextClassifier(TextClassifier):
    """
    Implements a basic text classifier:
    1) Embed tokens using `text_field_embedder`
    2) Get the CLS token
    3) Final feedforward layer

    Optimized with CrossEntropyLoss.  Evaluated with CategoricalAccuracy & F1.
    """
    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 verbose_metrics: bool = False,
                 dropout: float = 0.2,
                 threshold: float = 0.5,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None,
                 ) -> None:
        super(TextClassifier, self).__init__(vocab, regularizer)

        self.text_field_embedder = text_field_embedder
        self.dropout = torch.nn.Dropout(dropout)
        self.num_classes = self.vocab.get_vocab_size("labels")
        self.ths = torch.LongTensor([threshold for _ in range(self.num_classes)])

        self.classifier_feedforward = torch.nn.Linear(self.text_field_embedder.get_output_dim(), self.num_classes)

        self.label_accuracy = CategoricalAccuracy()
        self.label_f1_metrics = {}

        # TODO threshold
        self._micro_f1 = FBetaMeasureMultiLabel(average="micro", threshold=threshold)
        self._macro_f1 = FBetaMeasureMultiLabel(average="macro", threshold=threshold)

        self.verbose_metrics = verbose_metrics

        for i in range(self.num_classes):
            self.label_f1_metrics[vocab.get_token_from_index(index=i, namespace="labels")] = F1Measure(positive_label=i)

        self.loss = torch.nn.BCEWithLogitsLoss()

        initializer(self)

    @overrides
    def forward(self,
                text: Dict[str, torch.LongTensor],
                labels: torch.IntTensor = None,
                metadata:  List[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        text : Dict[str, torch.LongTensor]
            From a ``TextField``
        label : torch.IntTensor, optional (default = None)
            From a ``LabelField``
        metadata : ``List[Dict[str, Any]]``, optional, (default = None)
            Metadata containing the original tokenization of the premise and
            hypothesis with 'premise_tokens' and 'hypothesis_tokens' keys respectively.
        Returns
        -------
        An output dictionary consisting of:
        label_logits : torch.FloatTensor
            A tensor of shape ``(batch_size, num_labels)`` representing unnormalised log probabilities of the label.
        label_probs : torch.FloatTensor
            A tensor of shape ``(batch_size, num_labels)`` representing probabilities of the label.
        loss : torch.FloatTensor, optional
            A scalar loss to be optimised.
        """

        # print("SHAPES")
        # print("text len =", len(text))
        # print("labels shape =", labels.shape)
        # print("SHAPES END")

        embedded_text = self.text_field_embedder(text)
        pooled = self.dropout(embedded_text[:, 0, :])

        # compute logits
        logits = self.classifier_feedforward(pooled)

        # print("logits shape =", logits.shape)

        class_probs = torch.sigmoid(logits)

        # print("logits =", logits)
        # print("class_probs =", class_probs)


        # print("class_probs shape =", class_probs.shape)
        #
        #
        #
        # print(class_probs[0])
        # class_probs = torch.Tensor(class_probs).T
        # print(class_probs[0])
        # print(class_probs)
        #
        # print("class_probs[0] shape =", class_probs[0].shape)

        output_dict = {"logits": logits, "probs": class_probs}

        if labels is not None:
            labels = labels.type_as(logits)

            losses = self.loss(logits, labels.float().view(-1, self.num_classes))
            #losses = self.loss(logits, labels)

            print("logits.shape =", logits.shape)
            print("labels.shape =", labels.shape)

            # losses = [self.loss(logits, label) for logit, label in zip(logits, labels)]
            output_dict["loss"] = losses

            # compute F1 per label
            # for i in range(self.num_classes):
            #     metric = self.label_f1_metrics[self.vocab.get_token_from_index(index=i, namespace="labels")]
            #     metric(class_probs, labels)
                # map(lambda x, y: metric(x, y), zip(class_probs, labels))
            # self.label_accuracy(logits, labels)
            cloned_logits, cloned_labels = logits.clone(), labels.clone()
            self._micro_f1(cloned_logits, cloned_labels)
            self._macro_f1(cloned_logits, cloned_labels)
        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        micro = self._micro_f1.get_metric(reset)
        macro = self._macro_f1.get_metric(reset)

        metrics = {
            "micro_precision": micro["precision"],
            "micro_recall": micro["recall"],
            "micro_fscore": micro["fscore"],
            "macro_precision": macro["precision"],
            "macro_recall": macro["recall"],
            "macro_fscore": macro["fscore"],
        }
        return metrics
