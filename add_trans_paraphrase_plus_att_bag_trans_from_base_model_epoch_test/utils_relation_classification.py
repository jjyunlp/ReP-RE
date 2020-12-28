# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" BERT classification fine-tuning: utilities to work with GLUE tasks """

from __future__ import absolute_import, division, print_function

import csv
import logging
import os
import sys
from io import open
import json
import numpy as np


from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score
from collections import Counter

logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_a_google=None, text_a_baidu=None, text_a_xiaoniu=None, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_a_google = text_a_google
        self.text_a_baidu = text_a_baidu
        self.text_a_xiaoniu = text_a_xiaoniu
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data.
    e1_index and e2_index is the location of last word in e1 and e2 in textb
    """

    def __init__(self, 
                bag_input_ids, 
                bag_input_mask, 
                bag_segment_ids,                 
                bag_entity_position,
                label_id):
        self.bag_input_ids = bag_input_ids
        self.bag_input_mask = bag_input_mask
        self.bag_segment_ids = bag_segment_ids        
        self.bag_entity_position = bag_entity_position
        self.label_id = label_id    # bag share a label


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir, train_file ):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()
    
    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()
        
    @classmethod
    def _read_json(cls, input_file, quotechar=None):
        """Reads a json file.
        {
            "id": "1", # id是文件全局唯一量，可防止数据重复，字符串格式
            "head": "all basotho convention", # 头实体，小写字符串
            "tail": "tom thabane", # 尾实体，小写字符串
            # 根据tokens拼接的句子，全部小写，并以空格表示词次间隔
            "sentence": "tom thabane resigned in october last year to form the all ... ",
            "relation": "org:founded_by", # 实体间关系
            "translation": { # 翻译部分已经过tokenize和小写化操作
                "xiaoniu": "tom thabane resigned last october and formed all basotho ... .",
                "baidu": "tom thabane resigned in october last year and established ...",
                "google": "tom thabane resigned in october last year , forming all ...."
            },
            # 原始数据信息
            "original_info": {...}
        }
        """

        with open(input_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            logger.info(f"Load file from {input_file}, total {len(data)} instances.")            
            return data


class FeatureOutputProcessor(DataProcessor):
    """Processor for the any data to output features for each token."""

    def get_train_examples(self, data_dir):
        """See base class."""        
        return self._create_examples_from_json(
            self._read_json(os.path.join(data_dir, "semeval_train.json")), "test")
    
    def get_dev_examples(self, data_dir):
        """See base class."""        
        return self._create_examples_from_json(
            self._read_json(os.path.join(data_dir, "semeval_dev.json")), "test")

    def get_test_examples(self, data_dir):
        """See base class."""        
        return self._create_examples_from_json(
            self._read_json(os.path.join(data_dir, "semeval_test.json")), "test")            

    @staticmethod
    def _create_examples_from_json(json_data, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for inst in json_data:
            id = inst["id"]
            head = inst["head"]
            tail = inst["tail"]
            relation = inst["relation"]
            original_sen = inst["original_info"]["sentence"]
            xiaoniu_sen = inst["direct_translation"]["xiaoniu"]
            baidu_sen = inst["direct_translation"]["baidu"]
            google_sen = inst["direct_translation"]["google"]
            guid = "%s-%s" % (set_type, id)

            # need space
            original_sen = original_sen.replace('<e1>', ' [E1] ')
            original_sen = original_sen.replace('</e1>', ' [/E1] ')
            original_sen = original_sen.replace('<e2>', ' [E2] ')
            original_sen = original_sen.replace('</e2>', ' [/E2] ')            

            text_b = " [E] ".join([head, tail])
            label = relation
            text_b = None
            label = None

            examples.append(
                InputExample(guid=guid, 
                text_a=original_sen, 
                text_a_google=google_sen,
                text_a_baidu=baidu_sen,
                text_a_xiaoniu=xiaoniu_sen,
                text_b=text_b, label=label))
        return examples


class SemEvalProcessor(DataProcessor):
    """Processor for the SemEval-2010 data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info(f"LOOKING AT {data_dir}")
        return self._create_examples_from_json(
            self._read_json(os.path.join(data_dir, "semeval_train_tag_wrap_full.json")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""        
        return self._create_examples_from_json(
            self._read_json(os.path.join(data_dir, "semeval_dev_tag_wrap.json")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""        
        return self._create_examples_from_json(
            self._read_json(os.path.join(data_dir, "semeval_test_tag_wrap.json")), "test")

    def get_labels(self, data_dir):
        """See base class."""
        label_file = os.path.join(data_dir, "relation_dict.json")
        label_dict = json.load(open(label_file))
        id_2_label = {}                
        for k, v in label_dict.items():
            id_2_label[v] = k
        label_list = [] # index-label
        for i in range(len(label_dict)):
            label_list.append(id_2_label[i])
        return label_list, id_2_label

    @staticmethod
    def _create_examples_from_json(json_data, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for inst in json_data:
            id = inst["id"]
            head = inst["head"]
            tail = inst["tail"]
            relation = inst["relation"]
            original_sen = inst["tag_wrapped"]['human']
            xiaoniu_sen = inst["tag_wrapped"]["xiaoniu"]
            baidu_sen = inst["tag_wrapped"]["baidu"]
            google_sen = inst["tag_wrapped"]["google"]
            guid = "%s-%s" % (set_type, id)

            text_b = " [E] ".join([head, tail])
            label = relation
            text_b = None            

            examples.append(
                InputExample(guid=guid, 
                            text_a=original_sen, 
                            text_a_google=google_sen,
                            text_a_baidu=baidu_sen,
                            text_a_xiaoniu=xiaoniu_sen,
                            text_b=text_b, 
                            label=label))
        return examples


class SemEvalMergeDataProcessor(DataProcessor):
    """Processor for the SemEval-2010 data set.
    Merge all translated data
    """

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info(f"LOOKING AT {data_dir}")
        return self._create_examples_from_json(
            self._read_json(os.path.join(data_dir, "semeval_train_tag_wrap.json")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""        
        return self._create_examples_from_json(
            self._read_json(os.path.join(data_dir, "semeval_dev_tag_wrap.json")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""        
        return self._create_examples_from_json(
            self._read_json(os.path.join(data_dir, "semeval_test_tag_wrap.json")), "test")

    def get_labels(self, data_dir):
        """See base class."""
        label_file = os.path.join(data_dir, "relation_dict.json")
        label_dict = json.load(open(label_file))
        id_2_label = {}                
        for k, v in label_dict.items():
            id_2_label[v] = k
        label_list = [] # index-label
        for i in range(len(label_dict)):
            label_list.append(id_2_label[i])
        return label_list, id_2_label

    @staticmethod
    def _create_examples_from_json(json_data, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for inst in json_data:
            id = inst["id"]
            head = inst["head"]
            tail = inst["tail"]
            relation = inst["relation"]
            original_sen = inst["tag_wrapped"]['human']
            xiaoniu_sen = inst["tag_wrapped"]["xiaoniu"]
            baidu_sen = inst["tag_wrapped"]["baidu"]
            google_sen = inst["tag_wrapped"]["google"]
            guid = "%s-%s" % (set_type, id)

            text_b = " [E] ".join([head, tail])
            label = relation
            text_b = None            

            examples.append(InputExample(guid=guid, text_a=original_sen, label=label))
            if set_type == "train":
                examples.append(InputExample(guid=guid + "_g", text_a=google_sen, label=label))
                examples.append(InputExample(guid=guid + "_b", text_a=baidu_sen, label=label))
                examples.append(InputExample(guid=guid + "_x", text_a=xiaoniu_sen, label=label))        
        return examples


class TacredProcessor(DataProcessor):
    """Processor for the TACRED data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info(f"LOOKING AT {data_dir}")
        return self._create_examples_from_json(
            self._read_json(os.path.join(data_dir, "tacred_train_tag_wrap_full.json")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""        
        return self._create_examples_from_json(
            self._read_json(os.path.join(data_dir, "tacred_dev_tag_wrap.json")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""        
        return self._create_examples_from_json(
            self._read_json(os.path.join(data_dir, "tacred_test_tag_wrap.json")), "test")

    def get_labels(self, data_dir):
        """See base class."""
        label_file = os.path.join(data_dir, "relation_dict.json")
        label_dict = json.load(open(label_file))
        id_2_label = {}                
        for k, v in label_dict.items():
            id_2_label[v] = k
        label_list = [] # index-label
        for i in range(len(label_dict)):
            label_list.append(id_2_label[i])
        return label_list, id_2_label

    @staticmethod
    def _create_examples_from_json(json_data, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for inst in json_data:
            id = inst["id"]
            head = inst["head"]
            tail = inst["tail"]
            relation = inst["relation"]
            original_sen = inst["tag_wrapped"]['human']
            pad_sen = "[PAD]"
            if relation == "no_relation":
                xiaoniu_sen = pad_sen
                baidu_sen = pad_sen
                google_sen = pad_sen
            else:
                xiaoniu_sen = inst["tag_wrapped"]["xiaoniu"]
                baidu_sen = inst["tag_wrapped"]["baidu"]
                google_sen = inst["tag_wrapped"]["google"]
            guid = "%s-%s" % (set_type, id)
            
            label = relation
            text_b = None            

            examples.append(
                InputExample(guid=guid, 
                            text_a=original_sen, 
                            text_a_google=google_sen,
                            text_a_baidu=baidu_sen,
                            text_a_xiaoniu=xiaoniu_sen,                            
                            label=label))
        return examples


def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer, output_mode,
                                 cls_token_at_end=False,
                                 cls_token='[CLS]',
                                 cls_token_segment_id=1,
                                 sep_token='[SEP]',
                                 sep_token_extra=False,
                                 pad_on_left=False,
                                 pad_token=0,
                                 pad_token_segment_id=0,
                                 sequence_a_segment_id=0, 
                                 sequence_b_segment_id=1,
                                 mask_padding_with_zero=True):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """

    # label_map = {label : i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        bag_text_a = [example.text_a, example.text_a_google, example.text_a_baidu, example.text_a_xiaoniu]        
        bag_tokens = []
        bag_input_ids = []
        bag_input_mask = []
        bag_segment_ids = []
        bag_entity_position = []    # [head_start, head_end, tail_start, tail_end]                

        label_id = label_list.index(example.label)

        for text_a in bag_text_a: # human-label sen, google sen, baidu sen, xiaoniu sen
            head_start = 0
            head_end = 0
            tail_start = 0
            tail_end = 0
            # tokens_a = tokenizer.tokenize(text_a)
            # for merge data approach
            if text_a is None:
                continue      
            tokens_a = text_a.strip().split()   # Already tokenized in pre-processed                                                                                                                              

            tokens_b = None
            if example.text_b:
                tokens_b = tokenizer.tokenize(example.text_b)
                # Modifies `tokens_a` and `tokens_b` in place so that the total
                # length is less than the specified length.
                # Account for [CLS], [SEP], [SEP] with "- 3". " -4" for RoBERTa.
                special_tokens_count = 4 if sep_token_extra else 3
                _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - special_tokens_count)
            else:
                # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
                special_tokens_count = 3 if sep_token_extra else 2
                if len(tokens_a) > max_seq_length - special_tokens_count:
                    tokens_a = tokens_a[:(max_seq_length - special_tokens_count)]

            tokens = tokens_a + [sep_token]
            if sep_token_extra:
                # roberta uses an extra separator b/w pairs of sentences
                tokens += [sep_token]
            segment_ids = [sequence_a_segment_id] * len(tokens)

            if tokens_b:
                tokens += tokens_b + [sep_token]
                segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)

            if cls_token_at_end:
                tokens = tokens + [cls_token]
                segment_ids = segment_ids + [cls_token_segment_id]
            else:
                tokens = [cls_token] + tokens
                segment_ids = [cls_token_segment_id] + segment_ids
            
            # Get entity tag's index
            # only a little, hundreds
            #if '[E1]' not in tokens or '[/E1]' not in tokens or '[E2]' not in tokens or '[/E2]' not in tokens:
            #    print(tokens)
            
            if '[E1]' in tokens:
                head_start = tokens.index('[E1]')
            if '[/E1]' in tokens:
                head_end = tokens.index('[/E1]')
            if '[E2]' in tokens:
                tail_start = tokens.index('[E2]')
            if '[/E2]' in tokens:
                tail_end = tokens.index('[/E2]')

            input_ids = tokenizer.convert_tokens_to_ids(tokens)            

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding_length = max_seq_length - len(input_ids)
            if pad_on_left:
                input_ids = ([pad_token] * padding_length) + input_ids
                input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
                segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
            else:
                input_ids = input_ids + ([pad_token] * padding_length)
                input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
                segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length
            
            bag_tokens.append(tokens)
            bag_input_ids.append(input_ids)
            bag_input_mask.append(input_mask)
            bag_segment_ids.append(segment_ids)
            bag_entity_position.append([head_start, tail_start])
            
        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))            
            #logger.info(source)
            for i in range(len(bag_tokens)):
                logger.info("tokens: %s" % " ".join([str(x) for x in bag_tokens[i]]))
                logger.info("input_ids: %s" % " ".join([str(x) for x in bag_input_ids[i]]))
                logger.info("input_mask: %s" % " ".join([str(x) for x in bag_input_mask[i]]))
                logger.info("segment_ids: %s" % " ".join([str(x) for x in bag_segment_ids[i]]))
                logger.info("entity_position: %s" % " ".join([str(x) for x in bag_entity_position[i]]))
                logger.info("-------------------------------------------------------")
            logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
                InputFeatures(bag_input_ids=bag_input_ids,
                              bag_input_mask=bag_input_mask,
                              bag_segment_ids=bag_segment_ids,
                              bag_entity_position=bag_entity_position,
                              label_id=label_id
                              ))
    return features

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

def macro_f1_for_semeval(preds, labels):
    """
    evaluate metric for SemEval-2010
    """
    pred_total = [0] * 19
    pred_correct = [0] * 19
    label_total = [0] * 19
    assert len(preds) == len(labels)
    for i, label in enumerate(labels):
        pred = preds[i]
        label_total[label] += 1
        if pred == label:
            pred_correct[label] += 1
        pred_total[pred] += 1
    # Ensure Other:0, 1 and 2 are same label with reverse direction
    p_list = []
    r_list = []
    macro_f1 = []
    for index in range(1, len(label_total), 2):
        # two directions
        if pred_total[index] + pred_total[index + 1] != 0:
            p_list.append((pred_correct[index] + pred_correct[index + 1])
                          / (pred_total[index] + pred_total[index + 1]))
        else:
            p_list.append(0.0)
        if label_total[index] + label_total[index + 1] != 0:
            r_list.append((pred_correct[index] + pred_correct[index + 1])
                          / (label_total[index] + label_total[index + 1]))
        else:
            r_list.append(0.0)

    for i in range(len(p_list)):
        if p_list[i] + r_list[i] == 0:
            macro_f1.append(0.0)
        else:
            macro_f1.append(2 * p_list[i] * r_list[i] / (p_list[i] + r_list[i]))
    p_list = np.asarray(p_list)
    r_list = np.asarray(r_list)
    macro_f1 = np.asarray(macro_f1)
    return {
        "acc": p_list.mean(),
        "recall": r_list.mean(),
        "macro_f1": macro_f1.mean()
    }

def score_tacred(key, prediction, verbose=False):
    """
    evaluate metric for TACRED
    """
    # NO_RELATION = "no_relation"
    NO_RELATION = 0
    correct_by_relation = Counter()
    guessed_by_relation = Counter()
    gold_by_relation = Counter()

    # Loop over the data to compute a score
    for row in range(len(key)):
        gold = key[row]
        guess = prediction[row]

        if gold == NO_RELATION and guess == NO_RELATION:
            pass
        elif gold == NO_RELATION and guess != NO_RELATION:
            guessed_by_relation[guess] += 1
        elif gold != NO_RELATION and guess == NO_RELATION:
            gold_by_relation[gold] += 1
        elif gold != NO_RELATION and guess != NO_RELATION:
            guessed_by_relation[guess] += 1
            gold_by_relation[gold] += 1
            if gold == guess:
                correct_by_relation[guess] += 1

    # Print verbose information
    if verbose:
        print("Per-relation statistics:")
        relations = gold_by_relation.keys()
        longest_relation = 0
        for relation in sorted(relations):
            longest_relation = max(len(relation), longest_relation)
        for relation in sorted(relations):
            # (compute the score)
            correct = correct_by_relation[relation]
            guessed = guessed_by_relation[relation]
            gold = gold_by_relation[relation]
            prec = 1.0
            if guessed > 0:
                prec = float(correct) / float(guessed)
            recall = 0.0
            if gold > 0:
                recall = float(correct) / float(gold)
            f1 = 0.0
            if prec + recall > 0:
                f1 = 2.0 * prec * recall / (prec + recall)
            # (print the score)
            sys.stdout.write(("{:<" + str(longest_relation) + "}").format(relation))
            sys.stdout.write("  P: ")
            if prec < 0.1: sys.stdout.write(' ')
            if prec < 1.0: sys.stdout.write(' ')
            sys.stdout.write("{:.2%}".format(prec))
            sys.stdout.write("  R: ")
            if recall < 0.1: sys.stdout.write(' ')
            if recall < 1.0: sys.stdout.write(' ')
            sys.stdout.write("{:.2%}".format(recall))
            sys.stdout.write("  F1: ")
            if f1 < 0.1: sys.stdout.write(' ')
            if f1 < 1.0: sys.stdout.write(' ')
            sys.stdout.write("{:.2%}".format(f1))
            sys.stdout.write("  #: %d" % gold)
            sys.stdout.write("\n")
        print("")

    # Print the aggregate score
    if verbose:
        print("Final Score:")
    prec_micro = 1.0
    if sum(guessed_by_relation.values()) > 0:
        prec_micro = float(sum(correct_by_relation.values())) / float(sum(guessed_by_relation.values()))
    recall_micro = 0.0
    if sum(gold_by_relation.values()) > 0:
        recall_micro = float(sum(correct_by_relation.values())) / float(sum(gold_by_relation.values()))
    f1_micro = 0.0
    if prec_micro + recall_micro > 0.0:
        f1_micro = 2.0 * prec_micro * recall_micro / (prec_micro + recall_micro)
    print("Precision (micro): {:.3%}".format(prec_micro))
    print("   Recall (micro): {:.3%}".format(recall_micro))
    print("       F1 (micro): {:.3%}".format(f1_micro))
    return {
        "prec_micro": prec_micro,
        "recall_micro": recall_micro,
        "f1_micro": f1_micro
    }

def compute_metrics(task_name, preds, labels):
    assert len(preds) == len(labels)
    if task_name == "semeval":
        return {"semeval": macro_f1_for_semeval(preds, labels)}
    elif task_name == "tacred":
        return {"tacred": score_tacred(labels, preds)}
    else:
        raise KeyError(task_name)

processors = {
    "semeval": SemEvalProcessor,
    "semeval_merge_data": SemEvalMergeDataProcessor,
    "tacred": TacredProcessor,
    "bert-feature": FeatureOutputProcessor,
}

output_modes = {
    "semeval": "classification",
    "tacred": "classification",
}

GLUE_TASKS_NUM_LABELS = {
    "semeval": 19,
    "tacred": 53,
}
