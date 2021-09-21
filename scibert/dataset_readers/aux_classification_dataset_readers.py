""" Data readers for scaffold tasks in multitask training """


from typing import Dict, List, Any
import logging

import jsonlines
from overrides import overrides
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import LabelField, TextField, MultiLabelField, ListField, ArrayField, MetadataField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Tokenizer, WordTokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("aclarc_section_title_data_reader")
class AclSectionTitleDatasetReader(DatasetReader):
    def __init__(self,
                 lazy: bool = False,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 ) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

    @overrides
    def _read(self, file_path):
        with jsonlines.open(file_path) as f_in:
            for json_object in f_in:
                yield self.text_to_instance(
                    text=json_object.get('text'),
                    section_name=json_object.get('section_name'),
                    citing_paper_id=json_object.get('citing_paper_id'),
                    cited_paper_id=json_object.get('cited_paper_id')
                )

    @overrides
    def text_to_instance(self,
                         text: str,
                         section_name: str = None,
                         citing_paper_id: str = None,
                         cited_paper_id: str = None) -> Instance:  # type: ignore
        text_tokens = self._tokenizer.tokenize(text)
        fields = {
            'text': TextField(text_tokens, self._token_indexers),
        }

        if section_name is not None:
            fields['section_label'] = LabelField(section_name, label_namespace="section_labels")
        fields['citing_paper_id'] = MetadataField(citing_paper_id)
        fields['cited_paper_id'] = MetadataField(cited_paper_id)
        return Instance(fields)


@DatasetReader.register("aclarc_cite_worthiness_data_reader")
class AclCiteWorthinessDatasetReader(DatasetReader):
    def __init__(self,
                 lazy: bool = False,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 ) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

    @overrides
    def _read(self, file_path):
        with jsonlines.open(file_path) as f_in:
            for json_object in f_in:
                yield self.text_to_instance(
                    text=json_object.get('text'),
                    is_citation=json_object.get('is_citation'),
                    citing_paper_id=json_object.get('citing_paper_id'),
                    cited_paper_id=json_object.get('cited_paper_id')
                )

    @overrides
    def text_to_instance(self,
                         text: str,
                         is_citation: bool = None,
                         citing_paper_id: str = None,
                         cited_paper_id: str = None) -> Instance:  # type: ignore
        text_tokens = self._tokenizer.tokenize(text)
        fields = {
            'text': TextField(text_tokens, self._token_indexers),
        }

        if is_citation is not None:
            fields['is_citation'] = LabelField(str(is_citation), label_namespace="cite_worthiness_labels")
        fields['citing_paper_id'] = MetadataField(citing_paper_id)
        fields['cited_paper_id'] = MetadataField(cited_paper_id)
        return Instance(fields)
