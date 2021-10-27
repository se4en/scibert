import json
import operator
from typing import Tuple

from overrides import overrides

from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.service.predictors.predictor import Predictor

from scibert.data import read_jurgens_jsonline
from scibert.helper import JsonFloatEncoder
from scibert.constants import NONE_LABEL_NAME


@Predictor.register('predictor_aclarc')
class CitationIntentPredictorACL(Predictor):
    """"Predictor wrapper for the CitationIntentClassifier"""
    
    @overrides
    def predict_json(self, inputs: JsonDict) -> JsonDict:
        return_dict = {}
        citation = read_jurgens_jsonline(inputs)
        if len(citation.text) == 0:
            print('empty context, skipping')
            return {}
        print("INTENT=", citation.intent)
        instance = self._dataset_reader.text_to_instance(
            text=citation.text,
            label=citation.intent,
            metadata=citation.citing_paper_id,
            #cited_paper_id=citation.cited_paper_id,
            #citation_excerpt_index=citation.citation_excerpt_index,
            #sents_before=citation.sents_before,
            #sents_after=citation.sents_after,
            #cleaned_cite_text=citation.cleaned_cite_text
        )
        print("INSTANCE=", instance)
        outputs = self._model.forward_on_instance(instance)

        return_dict['citation_id'] = citation.citation_id
        #return_dict['citingPaperId'] = outputs['citing_paper_id']
        #return_dict['citedPaperId'] = outputs['cited_paper_id']
        return_dict['probabilities'] = outputs['probabilities']
        return_dict['prediction'] = outputs['prediction']
        return_dict['text'] = citation.text
        #return_dict['attn_output'] = outputs['attn_output']
        return return_dict

    @overrides
    def dump_line(self, outputs: JsonDict) -> str:  # pylint: disable=no-self-use
        """
        If you don't want your outputs in JSON-lines format
        you can override this function to output them differently.
        """
        keys = ['citation_id', 'prediction', 'probabilities', 'text']
        # keys = ['citation_id', 'prediction', 'probabilities', 'citation_text', 'attn_output']
        for k in outputs.copy():
            if k not in keys:
                outputs.pop(k)
        return json.dumps(outputs, cls=JsonFloatEncoder) + "\n"
