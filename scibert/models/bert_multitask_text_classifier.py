from typing import Dict, Optional, List, Any

import numpy as np
import torch
import torch.nn.functional as F
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import FeedForward, TextFieldEmbedder, Seq2SeqEncoder
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn import util
from allennlp.training.metrics import CategoricalAccuracy, F1Measure
from overrides import overrides
from scibert.models.text_classifier import TextClassifier


@Model.register("bert_multitask_text_classifier")
class BertMultitaskTextClassifier(TextClassifier):
    """
    Implements a basic text classifier:
    1) Embed tokens using `text_field_embedder`
    2) Get the CLS token
    3) Final feedforward layer

    Optimized with CrossEntropyLoss.  Evaluated with CategoricalAccuracy & F1.
    """
    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 classifier_feedforward: FeedForward,
                 classifier_feedforward_2: FeedForward,
                 classifier_feedforward_3: FeedForward,
                 verbose_metrics: bool = False,
                 weighted_loss: bool = False,
                 dropout: float = 0.2,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None,
                 ) -> None:
        super(TextClassifier, self).__init__(vocab, regularizer)

        self.text_field_embedder = text_field_embedder
        self.dropout = torch.nn.Dropout(dropout)
        self.num_classes = self.vocab.get_vocab_size("labels")
        self.num_classes_sections = self.vocab.get_vocab_size("section_labels")
        self.num_classes_cite_worthiness = self.vocab.get_vocab_size("cite_worthiness_labels")

        #self.classifier_feedforward = torch.nn.Linear(self.text_field_embedder.get_output_dim(), self.num_classes)
        #self.classifier_feedforward_2 = torch.nn.Linear(self.text_field_embedder.get_output_dim(),
        #                                                self.num_classes_sections)
        #self.classifier_feedforward_3 = torch.nn.Linear(self.text_field_embedder.get_output_dim(),
        #                                                self.num_classes_cite_worthiness)

        #self.classifier_feedforward = FeedForward(input_dim=self.text_field_embedder.get_output_dim(), num_layers=2, hidden_dims=[20, self.num_classes], 
        #                                          activations=[torch.nn.ReLU(), torch.nn.LazyLinear(self.num_classes, device=torch.device('cuda:0'))])
        #self.classifier_feedforward_2 = FeedForward(input_dim=self.text_field_embedder.get_output_dim(), num_layers=2, 
        #                                            hidden_dims=[20, self.num_classes_sections], 
        #                                            activations=[torch.nn.ReLU(), torch.nn.LazyLinear(self.num_classes, device=torch.device('cuda:0'))])
        #self.classifier_feedforward_3 = FeedForward(input_dim=self.text_field_embedder.get_output_dim(), num_layers=2, hidden_dims=[20, self.num_classes_cite_worthiness], 
        #                                            activations=[torch.nn.ReLU(), torch.nn.LazyLinear(self.num_classes, device=torch.device('cuda:0'))])
        
        self.classifier_feedforward = classifier_feedforward
        self.classifier_feedforward_2 = classifier_feedforward_2
        self.classifier_feedforward_3 = classifier_feedforward_3
        


        self.report_auxiliary_metrics = True
    
        self.label_accuracy = CategoricalAccuracy()
        self.label_f1_metrics = {}
        self.label_f1_metrics_sections = {}
        self.label_f1_metrics_cite_worthiness = {}

        self.verbose_metrics = verbose_metrics

        for i in range(self.num_classes):
            self.label_f1_metrics[vocab.get_token_from_index(index=i, namespace="labels")] = F1Measure(positive_label=i)
        for i in range(self.num_classes_sections):
            self.label_f1_metrics_sections[vocab.get_token_from_index(index=i, namespace="section_labels")] =\
                F1Measure(positive_label=i)
        for i in range(self.num_classes_cite_worthiness):
            self.label_f1_metrics_cite_worthiness[vocab.get_token_from_index(index=i, namespace="cite_worthiness_labels")] =\
                F1Measure(positive_label=i)

        self.loss = torch.nn.CrossEntropyLoss()
        self.weighted_loss = weighted_loss

        if self.weighted_loss:
            weights = [0.32447342, 0.88873626, 0.92165242, 3.67613636, 4.49305556, 4.6884058]
            class_weights = torch.FloatTensor(weights)  # .cuda()
            self.loss_main_task = torch.nn.CrossEntropyLoss(weight=class_weights)

        initializer(self)

    @overrides
    def forward(self,
                text: Dict[str, torch.LongTensor],
                label: torch.IntTensor = None,
                citing_paper_id: Optional[str] = None,
                cited_paper_id: Optional[str] = None,
                section_label: Optional[torch.Tensor] = None,
                is_citation: Optional[torch.Tensor] = None,
                metadata: Optional[str] = None) -> Dict[str, torch.Tensor]:
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
        embedded_text = self.text_field_embedder(text)
        pooled = self.dropout(embedded_text[:, 0, :])
        # TODO: add pooling flag      
        # pooled = self.dropout(torch.mean(embedded_text, 1))

        if label is not None:
            logits = self.classifier_feedforward(pooled)
            class_probs = F.softmax(logits, dim=1)
            
            predictions = class_probs.cpu().data.numpy()
            argmax_indices = np.argmax(predictions, axis=-1)
            labels = [self.vocab.get_token_from_index(x, namespace="labels")
                 for x in argmax_indices]
            output_dict = {"probabilities": class_probs}
            output_dict['prediction'] = labels
            #citation_text = []
            #print(text.keys())
            #for batch_text in text['tokens']:
            #    citation_text.append([self.vocab.get_token_from_index(token_id.item()) for token_id in batch_text])
            #output_dict['citation_text'] = citation_text


            output_dict['logits'] = logits
            
            # loss = self.loss(logits, label)
            loss = self.loss_main_task(logits, label)
            output_dict["loss"] = loss

            # compute F1 per label
            for i in range(self.num_classes):
                metric = self.label_f1_metrics[self.vocab.get_token_from_index(index=i, namespace="labels")]
                metric(class_probs, label)

            self.label_accuracy(logits, label)
            output_dict['labels'] = label

        if section_label is not None:  # this is the first scaffold task
            logits = self.classifier_feedforward_2(pooled)
            class_probs = F.softmax(logits, dim=1)

            output_dict = {"logits": logits}

            loss = self.loss(logits, section_label)
            output_dict["loss"] = loss

            for i in range(self.num_classes_sections):
                metric = self.label_f1_metrics_sections[self.vocab.get_token_from_index(index=i, namespace="section_labels")]
                metric(logits, section_label)

        if is_citation is not None:  # second scaffold task
            logits = self.classifier_feedforward_3(pooled)
            class_probs = F.softmax(logits, dim=1)

            output_dict = {"logits": logits}

            loss = self.loss(logits, is_citation)
            output_dict["loss"] = loss

            for i in range(self.num_classes_cite_worthiness):
                metric = self.label_f1_metrics_cite_worthiness[
                    self.vocab.get_token_from_index(index=i, namespace="cite_worthiness_labels")]
                metric(logits, is_citation)

        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metric_dict = {}

        sum_f1 = 0.0
        for name, metric in self.label_f1_metrics.items():
            metric_val = metric.get_metric(reset)
            metric_dict[name + '_P'] = metric_val[0]
            metric_dict[name + '_R'] = metric_val[1]
            metric_dict[name + '_F1'] = metric_val[2]
            if name != 'none':  # do not consider `none` label in averaging F1
                sum_f1 += metric_val[2]

        names = list(self.label_f1_metrics.keys())
        total_len = len(names) if 'none' not in names else len(names) - 1
        average_f1 = sum_f1 / total_len
        # metric_dict['combined_metric'] = (accuracy + average_f1) / 2
        metric_dict['average_F1'] = average_f1

        if self.report_auxiliary_metrics:
            sum_f1 = 0.0
            for name, metric in self.label_f1_metrics_sections.items():
                metric_val = metric.get_metric(reset)
                metric_dict['aux-sec--' + name + '_P'] = metric_val[0]
                metric_dict['aux-sec--' + name + '_R'] = metric_val[1]
                metric_dict['aux-sec--' + name + '_F1'] = metric_val[2]
                if name != 'none':  # do not consider `none` label in averaging F1
                    sum_f1 += metric_val[2]
            names = list(self.label_f1_metrics_sections.keys())
            total_len = len(names) if 'none' not in names else len(names) - 1
            average_f1 = sum_f1 / total_len
            # metric_dict['combined_metric'] = (accuracy + average_f1) / 2
            metric_dict['aux-sec--' + 'average_F1'] = average_f1

            sum_f1 = 0.0
            for name, metric in self.label_f1_metrics_cite_worthiness.items():
                metric_val = metric.get_metric(reset)
                metric_dict['aux-worth--' + name + '_P'] = metric_val[0]
                metric_dict['aux-worth--' + name + '_R'] = metric_val[1]
                metric_dict['aux-worth--' + name + '_F1'] = metric_val[2]
                if name != 'none':  # do not consider `none` label in averaging F1
                    sum_f1 += metric_val[2]
            names = list(self.label_f1_metrics_cite_worthiness.keys())
            total_len = len(names) if 'none' not in names else len(names) - 1
            average_f1 = sum_f1 / total_len
            # metric_dict['combined_metric'] = (accuracy + average_f1) / 2
            metric_dict['aux-worth--' + 'average_F1'] = average_f1

        return metric_dict

    @classmethod
    def from_params(cls, vocab: Vocabulary, params: Params) -> 'ScaffoldBilstmAttentionClassifier':

        embedder_params = params.pop("text_field_embedder")
        text_field_embedder = TextFieldEmbedder.from_params(embedder_params, vocab=vocab)

        classifier_feedforward = FeedForward.from_params(params.pop("classifier_feedforward"))
        classifier_feedforward_2 = FeedForward.from_params(params.pop("classifier_feedforward_2"))
        classifier_feedforward_3 = FeedForward.from_params(params.pop("classifier_feedforward_3"))

        weighted_loss = params.pop_bool("weighted_loss", False)
        verbose_metrics = params.pop_bool("verbose_metrics", False)

        initializer = InitializerApplicator.from_params(params.pop('initializer', []))
        regularizer = RegularizerApplicator.from_params(params.pop('regularizer', []))

        return cls(vocab=vocab,
                   text_field_embedder=text_field_embedder,
                   citation_text_encoder=citation_text_encoder
                   classifier_feedforward=classifier_feedforward,
                   classifier_feedforward_2=classifier_feedforward_2,
                   classifier_feedforward_3=classifier_feedforward_3,
                   verbose_metrics=verbose_metrics,
                   weighted_loss=weighted_loss,
                   initializer=initializer,
                   regularizer=regularizer)
