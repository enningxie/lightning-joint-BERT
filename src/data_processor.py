# -*- coding: utf-8 -*-
# Created by xieenning at 2020/10/28
from transformers.data.processors import DataProcessor
import json
import dataclasses
from dataclasses import dataclass
from typing import Optional, Union, List
import os
import numpy as np
from torchnlp.encoders import LabelEncoder
from torchnlp.utils import collate_tensors
from collections import defaultdict
from src.utils import is_whitespace
from transformers import PreTrainedTokenizer, BertTokenizer
from transformers.utils import logging

logger = logging.get_logger(__name__)


@dataclass
class InputExample:
    guid: str
    text_query: str
    text_context: Optional[str] = None
    label_slot_raw: Optional[List[str]] = None
    label_intent_raw: Optional[List[str]] = None

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(dataclasses.asdict(self), indent=2) + "\n"


@dataclass(frozen=True)
class InputFeatures:
    input_ids: List[int]
    attention_mask: Optional[List[int]] = None
    token_type_ids: Optional[List[int]] = None
    slot_mask: Optional[List[int]] = None
    label_slot: Optional[List[int]] = None
    label_intent: Optional[List[int]] = None

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(dataclasses.asdict(self)) + "\n"


def convert_examples_to_features(
        examples: List[InputExample],
        tokenizer: PreTrainedTokenizer,
        slot_label_encoder: LabelEncoder,
        intent_label_encoder: LabelEncoder,
        max_length: Optional[int] = None
):
    if max_length is None:
        max_length = tokenizer.max_len

    # slot / intent label id generate.
    truncation_size = max_length // 2
    features = []
    for example in examples:
        tmp_label_slot_origin = slot_label_encoder.batch_encode(example.label_slot_raw).numpy().tolist()
        if example.label_intent_raw:
            tmp_label_intent_origin = intent_label_encoder.batch_encode(example.label_intent_raw).numpy().tolist()
        else:
            tmp_label_intent_origin = []
        # 手动对query/context进行截断，截断长度分别为`max_length // 2`
        tmp_text_query = example.text_query[:truncation_size]  # query向后截断
        tmp_label_slot_origin = tmp_label_slot_origin[:truncation_size]
        tmp_text_context = example.text_context[-truncation_size:]  # context向前截断
        # TODO: 可选query/context顺序交换
        tokenizer_outputs = tokenizer.encode_plus(list(tmp_text_query), list(tmp_text_context), padding='max_length',
                                                  max_length=max_length)
        # 处理slot_mask
        tmp_slot_mask = np.asarray([0] * max_length)
        tmp_slot_mask[1:len(tmp_text_query) + 1] = 1
        # tmp_slot_mask[len(tmp_text_context)+2: len(tmp_text_context)+len(tmp_text_query)+2] = 1
        tmp_slot_mask = list(tmp_slot_mask)

        # 处理tmp_label_slot
        tmp_label_slot = [0] * max_length
        tmp_label_slot[1:len(tmp_text_query) + 1] = tmp_label_slot_origin

        # 处理tmp_label_intent
        tmp_label_intent = np.asarray([0] * intent_label_encoder.vocab_size)
        tmp_label_intent[tmp_label_intent_origin] = 1
        tmp_label_intent = list(tmp_label_intent)

        feature = InputFeatures(**tokenizer_outputs, slot_mask=tmp_slot_mask, label_slot=tmp_label_slot,
                                label_intent=tmp_label_intent)
        features.append(feature)

    for i, example in enumerate(examples[:5]):
        logger.info("*** Example ***")
        logger.info("guid: %s" % example.guid)
        logger.info("features: %s" % features[i])
    return features


class JointBERTDataProcessor(DataProcessor):
    """Processor for the Joint BERT data set."""

    def __init__(self):

        self.slot_label_encoder = None
        self.intent_label_encoder = None

        self.intent_counter = None

    @staticmethod
    def _read_from_json(file_path):
        raw_data = json.load(open(file_path))
        return raw_data

    def get_labels(self):
        """Gets the list of labels for this data set."""
        if self.intent_label_encoder is None:
            raise ValueError("Please run `get_train_examples` function first.")
        return self.slot_label_encoder.vocab, self.intent_label_encoder.vocab

    def get_train_examples(self, data_dir):
        """Gets a collection of :class:`InputExample` for the train set."""
        return self._create_examples(self._read_from_json(os.path.join(data_dir, "train_data.json")), "train")

    def get_dev_examples(self, data_dir):
        """Gets a collection of :class:`InputExample` for the dev set."""
        return self._create_examples(self._read_from_json(os.path.join(data_dir, "val_data.json")), "dev")

    def get_test_examples(self, data_dir):
        """Gets a collection of :class:`InputExample` for the test set."""
        return self._create_examples(self._read_from_json(os.path.join(data_dir, "test_data.json")), "test")

    def _create_examples(self, samples, set_type):
        """Creates examples for the training, dev and test sets."""
        examples = []
        if set_type == 'train':
            tmp_intent_counter = defaultdict(int)
            slot_label_set = set()
            intent_label_set = set()
        else:
            tmp_intent_counter = None
            slot_label_set = None
            intent_label_set = None
        for i, sample in enumerate(samples):
            guid = f"{set_type}-{i}"
            assert len(sample[0]) == len(sample[1]), f"Error slot label length occurred at {i} in {set_type} dataset."
            tmp_query_char_list = []
            label_slot_raw = []
            for tmp_char, tmp_slot_label in zip(sample[0], sample[1]):
                # 去除空格
                if is_whitespace(tmp_char):
                    continue
                tmp_query_char_list.append(tmp_char)
                label_slot_raw.append(tmp_slot_label)
            text_query = ''.join(tmp_query_char_list).lower()  # 转小写
            text_context = sample[4][0].lower()  # 转小写，现只支持单条context
            label_intent_raw = sample[2]
            if set_type == 'train':
                slot_label_set.update(label_slot_raw)
                intent_label_set.update(label_intent_raw)
                for tmp_item in label_intent_raw:
                    tmp_intent_counter[tmp_item] += 1
            examples.append(InputExample(guid=guid, text_query=text_query, text_context=text_context,
                                         label_slot_raw=label_slot_raw,
                                         label_intent_raw=label_intent_raw))
        if set_type == 'train':
            self.intent_counter = tmp_intent_counter
            self.slot_label_encoder = LabelEncoder(list(slot_label_set), reserved_labels=[], unknown_index=None)
            self.intent_label_encoder = LabelEncoder(list(intent_label_set), reserved_labels=[], unknown_index=None)
        return examples


if __name__ == '__main__':
    tmp_processor = JointBERTDataProcessor()
    tmp_train_data_path = '/Data/enningxie/Codes/Intent-Detection-models/data/training_data_0916'
    tmp_model_name_or_path = "/Data/public/pretrained_models/pytorch/chinese-bert-wwm-ext"
    tmp_tokenizer = BertTokenizer.from_pretrained(tmp_model_name_or_path)
    tmp_train_examples = tmp_processor.get_train_examples(tmp_train_data_path)
    tmp_slot_label, tmp_intent_label = tmp_processor.get_labels()

    tmp_features = convert_examples_to_features(
        tmp_train_examples,
        tmp_tokenizer,
        slot_label_encoder=tmp_processor.slot_label_encoder,
        intent_label_encoder=tmp_processor.intent_label_encoder,
        max_length=128
    )
    print('Break point.')
