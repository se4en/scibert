""" A predictor that works with the flat format of s2 data
i.e., one excerpt per dictionary instead of a nested dict"""
import json
import operator
from typing import Tuple

from overrides import overrides

from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.service.predictors.predictor import Predictor

from scibert.data import Citation, read_s2_excerpt
from scibert.helper import JsonFloatEncoder
from scibert.constants import NONE_LABEL_NAME
from scibert.constants import S2_CATEGORIES_MULTICLASS
from scibert.helper import is_sentence


@Predictor.register('predictor_scicite')
class PredictorSciCite(Predictor):
    """"Predictor wrapper for the CitationIntentClassifier"""
    
    @overrides
    def predict_json(self, inputs: JsonDict) -> JsonDict:
        return_dict = {}

        citation = read_s2_excerpt(inputs)
        # skip if the excerpt is not a valid sentence
        if len(citation.text) < 5 or not is_sentence(citation.text):
            return_dict['citingPaperId'] = inputs['citingPaperId']
            return_dict['citedPaperId'] = inputs['citedPaperId']
            return_dict['prediction'] = ''
        else:
            instance = self._dataset_reader.text_to_instance(
                text=citation.text,
                label=citation.intent,
                #citing_paper_id=citation.citing_paper_id,
                #cited_paper_id=citation.cited_paper_id,
                #citation_excerpt_index=citation.citation_excerpt_index
            )
            outputs = self._model.forward_on_instance(instance)

            #return_dict['citingPaperId'] = outputs.get('citing_paper_id')
            # return_dict['citedPaperId'] = outputs.get('cited_paper_id')
            return_dict['citation_id'] = citation.citation_id
            return_dict['probabilities'] = outputs.get('probabilities')
            return_dict['prediction'] = outputs['prediction']
            return_dict['original_label'] = citation.intent
            return_dict['text'] = citation.text
            #return_dict['attention_dist'] = outputs.get('attn_dist')
        return return_dict

    @overrides
    def dump_line(self, outputs: JsonDict) -> str:  # pylint: disable=no-self-use
        """
        If you don't want your outputs in JSON-lines format
        you can override this function to output them differently.
        """
        keys = ['citation_id', 'probabilities', 'prediction', 'original_label', 'text']
        #keys = ['citedPaperId', 'citingPaperId', 'excerptCitationIntents', 'prediction', 'probabilities',
        #        'citation_text']
        for k in outputs.copy():
            if k not in keys:
                outputs.pop(k)
        return json.dumps(outputs, cls=JsonFloatEncoder) + "\n"
