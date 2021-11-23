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
#from allennlp.training.metrics import FBetaMeasure
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

        #self.label_accuracy = CategoricalAccuracy()
        self.label_f1_metrics = {}
        for i in range(self.num_classes):
            self.label_f1_metrics[vocab.get_token_from_index(index=i, namespace="labels")] = F1Measure(positive_label=i)

        # TODO threshold
        #self._micro_f1 = FBetaMeasureMultiLabel(average="micro", threshold=threshold)
        #self.f1_measure = FBetaMeasureMultiLabel(average="macro", threshold=threshold)

        self.verbose_metrics = verbose_metrics

        # for i in range(self.num_classes):
        #     self.label_f1_metrics[vocab.get_token_from_index(index=i, namespace="labels")] = F1Measure(positive_label=i)

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

        # TODO point 0    
        # print("TEXT = ", text)

        embedded_text = self.text_field_embedder(text)
        
        # TODO point 1
        # print("EMBEDDED TEXT = ", embedded_text[:, 0, :])

        pooled = self.dropout(embedded_text[:, 0, :])

        # print("POOLED= ", pooled)

        # compute logits
        logits = self.classifier_feedforward(pooled)

        # print("logits shape =", logits.shape)

        class_probs = torch.sigmoid(logits)

        # print("logits =", logits)
        
        print(f"Params count = {sum(p.numel() for p in self.parameters() if p.requires_grad)}")

        # print("Params grad:", next(self.parameters()).grad.mean())
        # for p in self.parameters():
        #     if p.requires_grad:
        #         print(p.name)
        # print(f"Params names = {sum(p.numel() for p in self.parameters() if p.requires_grad)}")


        # TODO point 2
        print("class_probs =", class_probs)
        print("class_probs shape =", class_probs.shape)
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
            # TODO point 3
            print("LABELS=", labels)

            labels = labels.type_as(logits)
        
            print("LOGITS=", logits)


            # print("NEW_LABELS=", labels.float().view(-1, self.num_classes))


            print("Params grad:", next(self.parameters()).retain_grad())
            # losses = self.loss(logits, labels.float().view(-1, self.num_classes))
            losses = self.loss(logits, labels)
            print("Params grad:", next(self.parameters()).retain_grad())
            # print("Params grad mean:", torch.mean(next(self.parameters()).grad))
        
            # losses = self.loss(logits, labels)

            # print("logits.shape =", logits.shape)
            # print("labels.shape =", labels.shape)

            # TODO point 4
            print(f"losses = {losses}")

            # losses = [self.loss(logits, label) for logit, label in zip(logits, labels)]
            output_dict["loss"] = losses

            # compute F1 per label
            # for i in range(self.num_classes):
            #     metric = self.label_f1_metrics[self.vocab.get_token_from_index(index=i, namespace="labels")]
            #     metric(class_probs, labels)
            #     map(lambda x, y: metric(x, y), zip(class_probs, labels))
            # self.label_accuracy(logits, labels)
            cloned_logits, cloned_labels = logits.clone(), labels.clone()
            # self._micro_f1(cloned_logits, cloned_labels)
            # self._macro_f1(cloned_logits, cloned_labels)
            # self.f1_measure(cloned_logits, cloned_labels)

            # print("LABELS=", labels)

            for i in range(self.num_classes):
                metric = self.label_f1_metrics[self.vocab.get_token_from_index(index=i, namespace="labels")]
                # metric(class_probs[:, i], labels[:, i])
                # print(f"compute metric for {i} label")
        # prediciton mode
        else:
            # print("class_probs shape=", class_probs.shape)
            # print("class_probs =", class_probs) 
            # print("ths =", self.ths[0]) #  why 0 ?
            # preds = torch.where(class_probs > 0.5, True, False)
            pred_ids = (class_probs > 0.5).nonzero(as_tuple=False)
            # print("preds shape=", preds.shape)
            # print("preds =", preds)
            
            if pred_ids.nelement() != 0:
                labels = [[self.vocab.get_token_from_index(x, namespace="labels")
                            for x in labels_ids] for labels_ids in pred_ids]
            else:
                labels = [[]]
            
            output_dict['prediction'] = labels
 

        return output_dict

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
#        micro = self._micro_f1.get_metric(reset)
#        macro = self._macro_f1.get_metric(reset)
       # precision, recall, f1 = self.f1_measure.get_metric(reset)

        #metrics = {
#            "micro_precision": micro["precision"],
#            "micro_recall": micro["recall"],
#            "micro_fscore": micro["fscore"],
#            "macro_precision": macro["precision"],
#            "macro_recall": macro["recall"],
#            "macro_fscore": macro["fscore"],
#            "precision": precision,
#            "recal": recall,
#            "f1": f1,
#        }
#        return metrics
        metric_dict = {}

        sum_f1 = 0.0
        for name, metric in self.label_f1_metrics.items():
            metric_val = metric.get_metric(reset)
            metric_dict[name + '_P'] = metric_val[0]
            metric_dict[name + '_R'] = metric_val[1]
            metric_dict[name + '_F1'] = metric_val[2]
            if name != 'none':  # do not consider `none` label in averaging F1
                sum_f1 += metric_val[2]

        return metric_dict

