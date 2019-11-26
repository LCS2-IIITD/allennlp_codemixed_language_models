"""
Implements the DatasetReader for GirNet.
This is modified version of:
https://github.com/allenai/allennlp/blob/2850579831f392467276f1ab6d5cda3fdb45c3ba/allennlp/data/dataset_readers/simple_language_modeling.py
"""
import json
import logging
import math
from typing import Dict, Iterable, Union, Optional, List

from allennlp.common.checks import ConfigurationError
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.token_indexers.token_indexer import TokenIndexer
# from allennlp.data.tokenizers import SpacyTokenizer
from allennlp.data.tokenizers import Token
from allennlp.data.tokenizers.tokenizer import Tokenizer
from overrides import overrides

logger = logging.getLogger(__name__)


@DatasetReader.register("girnet_lm_reader")
class GirNetLMDatasetReader(DatasetReader):
    """

    Reads sentences, one per line, for language modeling. This does not handle arbitrarily formatted
    text with sentences spanning multiple lines.

    Parameters
    ----------
    tokenizer : ``Tokenizer``, optional
        Tokenizer to use to split the input sentences into words or other kinds of tokens. Defaults
        to ``SpacyTokenizer()``.
    token_indexers : ``Dict[str, TokenIndexer]``, optional
        Indexers used to define input token representations. Defaults to
        ``{"tokens": SingleIdTokenIndexer()}``.
    max_sequence_length: ``int``, optional
        If specified, sentences with more than this number of tokens will be dropped.
    start_tokens : ``List[str]``, optional (default=``None``)
        These are prepended to the tokens provided to the ``TextField``.
    end_tokens : ``List[str]``, optional (default=``None``)
        These are appended to the tokens provided to the ``TextField``.
    """

    def __init__(
            self,
            tokenizer: Tokenizer = None,
            token_indexers: Dict[str, TokenIndexer] = None,
            max_sequence_length: int = None,
            start_tokens: List[str] = None,
            end_tokens: List[str] = None,
    ) -> None:
        super().__init__(True)
        self._tokenizer = tokenizer or SpacyTokenizer()
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        if max_sequence_length is not None:
            self._max_sequence_length: Union[float, Optional[int]] = max_sequence_length
        else:
            self._max_sequence_length = math.inf

        self._start_tokens = [Token(st) for st in (start_tokens or [])]
        self._end_tokens = [Token(et) for et in (end_tokens or [])]

        logger.info("Creating GirNetLMDatasetReader")
        logger.info("max_sequence_length=%s", max_sequence_length)

    def text_to_instance(self) -> Instance:  # type: ignore
        raise RuntimeError("text_to_instance doesn't make sense here")

    @overrides
    def _read(self, file_path: str) -> Iterable[Instance]:
        """
        :param file_path: here is the file path is a json, with three keys, lang1, lang1, cm
        :return:
        """
        logger.info("JSON for file_path %s", file_path)
        try:
            file_paths = json.loads(file_path)
        except json.JSONDecodeError:
            raise ConfigurationError(
                "the file_path for the InterleavingDatasetReader "
                "needs to be a JSON-serialized dictionary {reader_name -> file_path}"
            )

        # read all the files at once

        logger.info("Loading for file_path %s", file_paths['lang1'])
        with open(file_paths['lang1']) as f:
            self.lang1_sent = f.readlines()
            no_lang1 = len(self.lang1_sent)

        logger.info("Loading for file_path %s", file_paths['lang2'])
        with open(file_paths['lang2']) as f:
            self.lang2_sent = f.readlines()
            no_lang2 = len(self.lang2_sent)

        logger.info("Loading for file_path %s", file_paths['cm'])
        with open(file_paths['cm']) as f:
            self.cm_sent = f.readlines()
            no_cm = len(self.cm_sent)

        # we iterate with file having the max numbers of sentences
        max_num = max(len(self.lang1_sent), len(self.lang2_sent), len(self.cm_sent))
        instances = 0
        dropped_instances = 0

        while instances < max_num:

            fields: Dict[str, TextField] = {
                "lang1": self.tokenise_sent(self.lang1_sent[instances % no_lang1]),
                "lang2": self.tokenise_sent(self.lang2_sent[instances % no_lang2]),
                "cm": self.tokenise_sent(self.cm_sent[instances % no_cm])
            }
            instance = Instance(fields)
            instances = instances + 1
            if (instance.fields["lang1"].sequence_length() <= self._max_sequence_length) and (
                    instance.fields["lang2"].sequence_length() <= self._max_sequence_length) and (
                    instance.fields["cm"].sequence_length() <= self._max_sequence_length):
                yield instance
            else:
                dropped_instances += 1

        logger.info(f"Tried these many instances {instances}")
        if not dropped_instances:
            logger.info(f"No instances dropped from {file_path}.")
        else:
            logger.warning(f"Dropped {dropped_instances} instances from {file_path}.")

    def tokenise_sent(
            self,  # type: ignore
            sentence: str,
    ) -> TextField:
        tokenized = self._tokenizer.tokenize(sentence)
        tokenized_with_ends = []
        tokenized_with_ends.extend(self._start_tokens)
        tokenized_with_ends.extend(tokenized)
        tokenized_with_ends.extend(self._end_tokens)
        return TextField(tokenized_with_ends, self._token_indexers)
