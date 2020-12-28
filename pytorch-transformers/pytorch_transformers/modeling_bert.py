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
"""PyTorch BERT model. """

from __future__ import absolute_import, division, print_function, unicode_literals

import json
import logging
import math
import os
import sys
from io import open

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
import torch.nn.functional as F

from .modeling_utils import PreTrainedModel, prune_linear_layer
from .configuration_bert import BertConfig
from .file_utils import add_start_docstrings

logger = logging.getLogger(__name__)

BERT_PRETRAINED_MODEL_ARCHIVE_MAP = {
    'bert-base-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-pytorch_model.bin",
    'bert-large-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-pytorch_model.bin",
    'bert-base-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-pytorch_model.bin",
    'bert-large-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-pytorch_model.bin",
    'bert-base-multilingual-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-uncased-pytorch_model.bin",
    'bert-base-multilingual-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-cased-pytorch_model.bin",
    'bert-base-chinese': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese-pytorch_model.bin",
    'bert-base-german-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-german-cased-pytorch_model.bin",
    'bert-large-uncased-whole-word-masking': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-whole-word-masking-pytorch_model.bin",
    'bert-large-cased-whole-word-masking': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-whole-word-masking-pytorch_model.bin",
    'bert-large-uncased-whole-word-masking-finetuned-squad': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-whole-word-masking-finetuned-squad-pytorch_model.bin",
    'bert-large-cased-whole-word-masking-finetuned-squad': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-whole-word-masking-finetuned-squad-pytorch_model.bin",
    'bert-base-cased-finetuned-mrpc': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-finetuned-mrpc-pytorch_model.bin",
}

def load_tf_weights_in_bert(model, config, tf_checkpoint_path):
    """ Load tf checkpoints in a pytorch model.
    """
    try:
        import re
        import numpy as np
        import tensorflow as tf
    except ImportError:
        logger.error("Loading a TensorFlow model in PyTorch, requires TensorFlow to be installed. Please see "
            "https://www.tensorflow.org/install/ for installation instructions.")
        raise
    tf_path = os.path.abspath(tf_checkpoint_path)
    logger.info("Converting TensorFlow checkpoint from {}".format(tf_path))
    # Load weights from TF model
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []
    for name, shape in init_vars:
        logger.info("Loading TF weight {} with shape {}".format(name, shape))
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        arrays.append(array)

    for name, array in zip(names, arrays):
        name = name.split('/')
        # adam_v and adam_m are variables used in AdamWeightDecayOptimizer to calculated m and v
        # which are not required for using pretrained model
        if any(n in ["adam_v", "adam_m", "global_step"] for n in name):
            logger.info("Skipping {}".format("/".join(name)))
            continue
        pointer = model
        for m_name in name:
            if re.fullmatch(r'[A-Za-z]+_\d+', m_name):
                l = re.split(r'_(\d+)', m_name)
            else:
                l = [m_name]
            if l[0] == 'kernel' or l[0] == 'gamma':
                pointer = getattr(pointer, 'weight')
            elif l[0] == 'output_bias' or l[0] == 'beta':
                pointer = getattr(pointer, 'bias')
            elif l[0] == 'output_weights':
                pointer = getattr(pointer, 'weight')
            elif l[0] == 'squad':
                pointer = getattr(pointer, 'classifier')
            else:
                try:
                    pointer = getattr(pointer, l[0])
                except AttributeError:
                    logger.info("Skipping {}".format("/".join(name)))
                    continue
            if len(l) >= 2:
                num = int(l[1])
                pointer = pointer[num]
        if m_name[-11:] == '_embeddings':
            pointer = getattr(pointer, 'weight')
        elif m_name == 'kernel':
            array = np.transpose(array)
        try:
            assert pointer.shape == array.shape
        except AssertionError as e:
            e.args += (pointer.shape, array.shape)
            raise
        logger.info("Initialize PyTorch weight {}".format(name))
        pointer.data = torch.from_numpy(array)
    return model


def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish}


try:
    from apex.normalization.fused_layer_norm import FusedLayerNorm as BertLayerNorm
except (ImportError, AttributeError) as e:
    logger.info("Better speed can be achieved with apex installed from https://www.github.com/nvidia/apex .")
    BertLayerNorm = torch.nn.LayerNorm

class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """
    def __init__(self, config):
        super(BertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None, position_ids=None):
        seq_length = input_ids.size(1)
               
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        
        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.output_attentions = config.output_attentions

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask, head_mask=None):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask    # attention_mask like: [1,1,1,...0,0,0,0]

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if self.output_attentions else (context_layer,)
        return outputs


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):
    def __init__(self, config):
        super(BertAttention, self).__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        mask = torch.ones(self.self.num_attention_heads, self.self.attention_head_size)
        heads = set(heads) - self.pruned_heads  # Convert to set and emove already pruned heads
        for head in heads:
            # Compute how many pruned heads are before the head and move the index accordingly
            head = head - sum(1 if h < head else 0 for h in self.pruned_heads)
            mask[head] = 0
        mask = mask.view(-1).contiguous().eq(1)
        index = torch.arange(len(mask))[mask].long()

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(self, input_tensor, attention_mask, head_mask=None):
        self_outputs = self.self(input_tensor, attention_mask, head_mask)
        attention_output = self.output(self_outputs[0], input_tensor)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str) or (sys.version_info[0] == 2 and isinstance(config.hidden_act, unicode)):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):
    def __init__(self, config):
        super(BertLayer, self).__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask, head_mask=None):
        attention_outputs = self.attention(hidden_states, attention_mask, head_mask)
        attention_output = attention_outputs[0]
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + attention_outputs[1:]  # add attentions if we output them
        return outputs


class BertEncoder(nn.Module):
    def __init__(self, config):
        super(BertEncoder, self).__init__()
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask, head_mask=None):
        all_hidden_states = ()
        all_attentions = ()
        for i, layer_module in enumerate(self.layer):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(hidden_states, attention_mask, head_mask[i])
            hidden_states = layer_outputs[0]

            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # Add last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs  # last-layer hidden state, (all hidden states), (all attentions)


class BertPooler(nn.Module):
    def __init__(self, config):
        super(BertPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states, entity_position, sep_ids, sen_lens=None):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.

        # output all [E1] or [E2] for this batch
        pooled_output = None
        if entity_position is not None:            
            head_indices = entity_position.select(1, 0)     # select(dim, index)
            tail_indices = entity_position.select(1, 1)
            head_token_tensor = torch.gather(hidden_states, 1, head_indices.view(-1, 1).unsqueeze(2).repeat(1, 1, hidden_states.size()[2])).squeeze()
            tail_token_tensor = torch.gather(hidden_states, 1, tail_indices.view(-1, 1).unsqueeze(2).repeat(1, 1, hidden_states.size()[2])).squeeze()                        
            head_pooled_output = self.dense(head_token_tensor)
            tail_pooled_output = self.dense(tail_token_tensor)
            # if last batch has only 1 instance, 
            pooled_output = torch.cat([head_pooled_output, tail_pooled_output], -1)            
        elif sep_ids is not None:
            # Output first [SEP]
            sep_token_tensor = torch.gather(hidden_states, 1, sep_ids.view(-1, 1).unsqueeze(2).repeat(1, 1, hidden_states.size()[2])).squeeze()            
            sep_pooled_output = self.dense(sep_token_tensor) 
            first_token_tensor = hidden_states[:, 0]
            pooled_output = torch.cat([first_token_tensor, sep_pooled_output], -1)        
        elif sen_lens is not None:
            
            # out all words' last layer hidden states and average pooling
            batch_sen_ave_hidden_states = []    #
            for i in range(hidden_states.size()[0]):    # batch size                                
                sen_hidden_states = torch.mean(hidden_states[i, 1: sen_lens[i]+1, : ], dim=0) # skip cls
                batch_sen_ave_hidden_states.append(sen_hidden_states)
            batch_sen_ave_hidden_states = torch.stack(batch_sen_ave_hidden_states)
            #print(batch_sen_ave_hidden_states)
            #print(batch_sen_ave_hidden_states.size())
            pooled_output = batch_sen_ave_hidden_states
            #exit()
            # Or
            """
            for i in range(hidden_states.size()[0]):    # batch size     
                if i == 0:
                    a = torch.mean(hidden_states[i, 1: sen_len + 1 ]) # skip cls
                else:
                    a = torch.cat([a, torch.mean(hidden_states[i, 1: sen_len + 1 ])], 1)
                for sen_len in sen_lens[i]:                    
                    sen_hidden_states = torch.mean(hidden_states[i, 1: sen_len + 1 ]) # skip cls
                    batch_sen_ave_hidden_states.append(sen_hidden_states)
            """
            

        if pooled_output is not None:
            pooled_output = self.activation(pooled_output)                
            if len(pooled_output.size()) == 1:            
                pooled_output = pooled_output.unsqueeze(0)        

        # [CLS]
        first_token_tensor = hidden_states[:, 0]
        cls_pooled_output = self.dense(first_token_tensor)
        cls_pooled_output = self.activation(cls_pooled_output)
        if len(cls_pooled_output.size()) == 1:            
            cls_pooled_output = cls_pooled_output.unsqueeze(0)       
        # all valid words' average pooling

        return (cls_pooled_output, pooled_output)


class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super(BertPredictionHeadTransform, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str) or (sys.version_info[0] == 2 and isinstance(config.hidden_act, unicode)):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertLMPredictionHead(nn.Module):
    def __init__(self, config):
        super(BertLMPredictionHead, self).__init__()
        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(config.hidden_size,
                                 config.vocab_size,
                                 bias=False)

        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states


class BertOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super(BertOnlyMLMHead, self).__init__()
        self.predictions = BertLMPredictionHead(config)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class BertOnlyNSPHead(nn.Module):
    def __init__(self, config):
        super(BertOnlyNSPHead, self).__init__()
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, pooled_output):
        seq_relationship_score = self.seq_relationship(pooled_output)
        return seq_relationship_score


class BertPreTrainingHeads(nn.Module):
    def __init__(self, config):
        super(BertPreTrainingHeads, self).__init__()
        self.predictions = BertLMPredictionHead(config)
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, sequence_output, pooled_output):
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score


class BertPreTrainedModel(PreTrainedModel):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """
    config_class = BertConfig
    pretrained_model_archive_map = BERT_PRETRAINED_MODEL_ARCHIVE_MAP
    load_tf_weights = load_tf_weights_in_bert
    base_model_prefix = "bert"

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


BERT_START_DOCSTRING = r"""    The BERT model was proposed in
    `BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding`_
    by Jacob Devlin, Ming-Wei Chang, Kenton Lee and Kristina Toutanova. It's a bidirectional transformer
    pre-trained using a combination of masked language modeling objective and next sentence prediction
    on a large corpus comprising the Toronto Book Corpus and Wikipedia.

    This model is a PyTorch `torch.nn.Module`_ sub-class. Use it as a regular PyTorch Module and
    refer to the PyTorch documentation for all matter related to general usage and behavior.

    .. _`BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding`:
        https://arxiv.org/abs/1810.04805

    .. _`torch.nn.Module`:
        https://pytorch.org/docs/stable/nn.html#module

    Parameters:
        config (:class:`~pytorch_transformers.BertConfig`): Model configuration class with all the parameters of the model. 
            Initializing with a config file does not load the weights associated with the model, only the configuration.
            Check out the :meth:`~pytorch_transformers.PreTrainedModel.from_pretrained` method to load the model weights.
"""

BERT_INPUTS_DOCSTRING = r"""
    Inputs:
        **input_ids**: ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Indices of input sequence tokens in the vocabulary.
            To match pre-training, BERT input sequence should be formatted with [CLS] and [SEP] tokens as follows:

            (a) For sequence pairs:

                ``tokens:         [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]``
                
                ``token_type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1``

            (b) For single sequences:

                ``tokens:         [CLS] the dog is hairy . [SEP]``
                
                ``token_type_ids:   0   0   0   0  0     0   0``

            Bert is a model with absolute position embeddings so it's usually advised to pad the inputs on
            the right rather than the left.

            Indices can be obtained using :class:`pytorch_transformers.BertTokenizer`.
            See :func:`pytorch_transformers.PreTrainedTokenizer.encode` and
            :func:`pytorch_transformers.PreTrainedTokenizer.convert_tokens_to_ids` for details.
        **attention_mask**: (`optional`) ``torch.FloatTensor`` of shape ``(batch_size, sequence_length)``:
            Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.
        **token_type_ids**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Segment token indices to indicate first and second portions of the inputs.
            Indices are selected in ``[0, 1]``: ``0`` corresponds to a `sentence A` token, ``1``
            corresponds to a `sentence B` token
            (see `BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding`_ for more details).
        **position_ids**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Indices of positions of each input sequence tokens in the position embeddings.
            Selected in the range ``[0, config.max_position_embeddings - 1]``.
        **head_mask**: (`optional`) ``torch.FloatTensor`` of shape ``(num_heads,)`` or ``(num_layers, num_heads)``:
            Mask to nullify selected heads of the self-attention modules.
            Mask values selected in ``[0, 1]``:
            ``1`` indicates the head is **not masked**, ``0`` indicates the head is **masked**.
"""

@add_start_docstrings("The bare Bert Model transformer outputting raw hidden-states without any specific head on top.",
                      BERT_START_DOCSTRING, BERT_INPUTS_DOCSTRING)
class BertModel(BertPreTrainedModel):
    r"""
    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **last_hidden_state**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, hidden_size)``
            Sequence of hidden-states at the output of the last layer of the model.
        **pooler_output**: ``torch.FloatTensor`` of shape ``(batch_size, hidden_size)``
            Last layer hidden-state of the first token of the sequence (classification token)
            further processed by a Linear layer and a Tanh activation function. The Linear
            layer weights are trained from the next sentence prediction (classification)
            objective during Bert pretraining. This output is usually *not* a good summary
            of the semantic content of the input, you're often better with averaging or pooling
            the sequence of hidden-states for the whole input sequence.
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained('bert-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids)
        last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple

    """
    def __init__(self, config):
        super(BertModel, self).__init__(config)

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)

        self.init_weights()

    def _resize_token_embeddings(self, new_num_tokens):
        old_embeddings = self.embeddings.word_embeddings
        new_embeddings = self._get_resized_embeddings(old_embeddings, new_num_tokens)
        self.embeddings.word_embeddings = new_embeddings
        return self.embeddings.word_embeddings

    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
            See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, 
                head_mask=None, entity_position=None, sep_ids=None, sen_lens=None):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # We can specify head_mask for each layer
            head_mask = head_mask.to(dtype=next(self.parameters()).dtype) # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.config.num_hidden_layers
        
        embedding_output = self.embeddings(input_ids, position_ids=position_ids, token_type_ids=token_type_ids)
        encoder_outputs = self.encoder(embedding_output,
                                       extended_attention_mask,
                                       head_mask=head_mask)
        # 0 for last layer hidden states
        sequence_output = encoder_outputs[0]
        # entity_position is not None will return head_pooled_output cat tail_pooled_output        
        # always output first token([CLS])
        cls_pooled_output, pooled_output = self.pooler(sequence_output, entity_position, sep_ids, sen_lens)

        # outputs = (sequence_output, pooled_output,) + encoder_outputs[1:]  # add hidden_states and attentions if they are here
        outputs = (cls_pooled_output, pooled_output,) + encoder_outputs[1:]  # add hidden_states and attentions if they are here
              
        return outputs  # sequence_output, pooled_output, (hidden_states), (attentions)


@add_start_docstrings("""Bert Model with two heads on top as done during the pre-training:
    a `masked language modeling` head and a `next sentence prediction (classification)` head. """,
    BERT_START_DOCSTRING, BERT_INPUTS_DOCSTRING)
class BertForPreTraining(BertPreTrainedModel):
    r"""
        **masked_lm_labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Labels for computing the masked language modeling loss.
            Indices should be in ``[-1, 0, ..., config.vocab_size]`` (see ``input_ids`` docstring)
            Tokens with indices set to ``-1`` are ignored (masked), the loss is only computed for the tokens with labels
            in ``[0, ..., config.vocab_size]``
        **next_sentence_label**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for computing the next sequence prediction (classification) loss. Input should be a sequence pair (see ``input_ids`` docstring)
            Indices should be in ``[0, 1]``.
            ``0`` indicates sequence B is a continuation of sequence A,
            ``1`` indicates sequence B is a random sequence.

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when both ``masked_lm_labels`` and ``next_sentence_label`` are provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Total loss as the sum of the masked language modeling loss and the next sequence prediction (classification) loss.
        **prediction_scores**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, config.vocab_size)``
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        **seq_relationship_scores**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, 2)``
            Prediction scores of the next sequence prediction (classification) head (scores of True/False continuation before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForPreTraining.from_pretrained('bert-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids)
        prediction_scores, seq_relationship_scores = outputs[:2]

    """
    def __init__(self, config):
        super(BertForPreTraining, self).__init__(config)

        self.bert = BertModel(config)
        self.cls = BertPreTrainingHeads(config)

        self.init_weights()
        self.tie_weights()

    def tie_weights(self):
        """ Make sure we are sharing the input and output embeddings.
            Export to TorchScript can't handle parameter sharing so we are cloning them instead.
        """
        self._tie_or_clone_weights(self.cls.predictions.decoder,
                                   self.bert.embeddings.word_embeddings)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                masked_lm_labels=None, next_sentence_label=None):

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids, 
                            head_mask=head_mask)

        sequence_output, pooled_output = outputs[:2]
        prediction_scores, seq_relationship_score = self.cls(sequence_output, pooled_output)

        outputs = (prediction_scores, seq_relationship_score,) + outputs[2:]  # add hidden states and attention if they are here

        if masked_lm_labels is not None and next_sentence_label is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
            next_sentence_loss = loss_fct(seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))
            total_loss = masked_lm_loss + next_sentence_loss
            outputs = (total_loss,) + outputs

        return outputs  # (loss), prediction_scores, seq_relationship_score, (hidden_states), (attentions)


@add_start_docstrings("""Bert Model with a `language modeling` head on top. """,
    BERT_START_DOCSTRING, BERT_INPUTS_DOCSTRING)
class BertForMaskedLM(BertPreTrainedModel):
    r"""
        **masked_lm_labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Labels for computing the masked language modeling loss.
            Indices should be in ``[-1, 0, ..., config.vocab_size]`` (see ``input_ids`` docstring)
            Tokens with indices set to ``-1`` are ignored (masked), the loss is only computed for the tokens with labels
            in ``[0, ..., config.vocab_size]``

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``masked_lm_labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Masked language modeling loss.
        **prediction_scores**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, config.vocab_size)``
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForMaskedLM.from_pretrained('bert-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, masked_lm_labels=input_ids)
        loss, prediction_scores = outputs[:2]

    """
    def __init__(self, config):
        super(BertForMaskedLM, self).__init__(config)

        self.bert = BertModel(config)
        self.cls = BertOnlyMLMHead(config)

        self.init_weights()
        self.tie_weights()

    def tie_weights(self):
        """ Make sure we are sharing the input and output embeddings.
            Export to TorchScript can't handle parameter sharing so we are cloning them instead.
        """
        self._tie_or_clone_weights(self.cls.predictions.decoder,
                                   self.bert.embeddings.word_embeddings)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                masked_lm_labels=None):

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids, 
                            head_mask=head_mask)

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)

        outputs = (prediction_scores,) + outputs[2:]  # Add hidden states and attention if they are here
        if masked_lm_labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
            outputs = (masked_lm_loss,) + outputs

        return outputs  # (masked_lm_loss), prediction_scores, (hidden_states), (attentions)


@add_start_docstrings("""Bert Model with a `next sentence prediction (classification)` head on top. """,
    BERT_START_DOCSTRING, BERT_INPUTS_DOCSTRING)
class BertForNextSentencePrediction(BertPreTrainedModel):
    r"""
        **next_sentence_label**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for computing the next sequence prediction (classification) loss. Input should be a sequence pair (see ``input_ids`` docstring)
            Indices should be in ``[0, 1]``.
            ``0`` indicates sequence B is a continuation of sequence A,
            ``1`` indicates sequence B is a random sequence.

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``next_sentence_label`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Next sequence prediction (classification) loss.
        **seq_relationship_scores**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, 2)``
            Prediction scores of the next sequence prediction (classification) head (scores of True/False continuation before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForNextSentencePrediction.from_pretrained('bert-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids)
        seq_relationship_scores = outputs[0]

    """
    def __init__(self, config):
        super(BertForNextSentencePrediction, self).__init__(config)

        self.bert = BertModel(config)
        self.cls = BertOnlyNSPHead(config)

        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                next_sentence_label=None):

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids, 
                            head_mask=head_mask)

        #pooled_output = outputs[1]
        pooled_output = outputs[0]

        seq_relationship_score = self.cls(pooled_output)

        outputs = (seq_relationship_score,) + outputs[2:]  # add hidden states and attention if they are here
        if next_sentence_label is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            next_sentence_loss = loss_fct(seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))
            outputs = (next_sentence_loss,) + outputs

        return outputs  # (next_sentence_loss), seq_relationship_score, (hidden_states), (attentions)


@add_start_docstrings("""Bert Model transformer with a sequence classification/regression head on top (a linear layer on top of
    the pooled output) e.g. for GLUE tasks. """,
    BERT_START_DOCSTRING, BERT_INPUTS_DOCSTRING)
class BertForSequenceClassification(BertPreTrainedModel):
    r"""
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for computing the sequence classification/regression loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.
            If ``config.num_labels == 1`` a regression loss is computed (Mean-Square loss),
            If ``config.num_labels > 1`` a classification loss is computed (Cross-Entropy).

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Classification (or regression if config.num_labels==1) loss.
        **logits**: ``torch.FloatTensor`` of shape ``(batch_size, config.num_labels)``
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, logits = outputs[:2]

    """
    def __init__(self, config):
        super(BertForSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)

        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, labels=None):

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids, 
                            head_mask=head_mask)

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)



@add_start_docstrings("""Pretrain for paraphrase with similarity. """,
    BERT_START_DOCSTRING, BERT_INPUTS_DOCSTRING)
class BertForPretrainParaphraseWithSimilarity(BertPreTrainedModel):
    r"""
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for computing the sequence classification/regression loss.

    """
    def __init__(self, config):
        super(BertForPretrainParaphraseWithSimilarity, self).__init__(config)
        # self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)
        self.cos = torch.nn.CosineSimilarity()
        #self.margin = 0.4
        #self.margin = torch.tensor([0.4], requires_grad=False)

        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, margin=None):

        s1_outputs, _ = self.bert(input_ids.select(1, 0),
                            attention_mask=attention_mask.select(1, 0),
                            token_type_ids=token_type_ids.select(1, 0),                            
                            )
        s1_pool_outputs = s1_outputs[1]
        s1_pool_outputs = self.dropout(s1_pool_outputs)
        
        s2_outputs, _ = self.bert(input_ids.select(1, 1),
                            attention_mask=attention_mask.select(1, 1),
                            token_type_ids=token_type_ids.select(1, 1),                            
                            )          
        s2_pool_outputs = s2_outputs[1]
        s2_pool_outputs = self.dropout(s2_pool_outputs)
        # [batch_size, hidden_state_size]
        #print(s1_pool_outputs.size())
        
        # find a negative sample t in this batch for each s1
        t_dict = {}  # which paraphrase pair
        tt_dict = {}     # first or second, <s, s'>
        max_cos_value_dict = {}
        for i in range(s1_pool_outputs.size()[0]):
            max_cos_value = -1.0
            for j in range(s1_pool_outputs.size()[0]):
                if i == j:
                    continue                
                # cos default dim=1, but here are [768] , [768] -> [1,768] [1,768]
                cos_value = self.cos(s1_pool_outputs[i].unsqueeze(0), s1_pool_outputs[j].unsqueeze(0))
                if float(cos_value) > max_cos_value:
                    max_cos_value = float(cos_value)
                    max_cos_value_dict[i] = cos_value
                    t_dict[i] = j
                    tt_dict[i] = 0
                cos_value = self.cos(s1_pool_outputs[i].unsqueeze(0), s2_pool_outputs[j].unsqueeze(0))
                if float(cos_value) > max_cos_value:
                    max_cos_value = float(cos_value)
                    max_cos_value_dict[i] = cos_value
                    t_dict[i] = j
                    tt_dict[i] = 1
        
        # sum negatives' cos value
        negative_batch_loss = max_cos_value_dict[0]
        for i in range(1, len(max_cos_value_dict)):        
            negative_batch_loss += max_cos_value_dict[i]                        
        negative_batch_loss = negative_batch_loss/len(max_cos_value_dict)
        # compute loss
        # loss(s1, s2) = max(0, margin - cos(h(s1), h(s2)) + cos(h(s1), h(t)))
        
        #loss = self.margin - torch.sum(self.cos(s1_pool_outputs, s2_pool_outputs)) + negative_batch_loss
        a = torch.log(1 + 2 * (torch.exp(2.5 - self.cos(s1_pool_outputs, s2_pool_outputs).mean())))
        b = torch.log(1 + 2 * (torch.exp(0.5 + negative_batch_loss)))
        #print(self.cos(s1_pool_outputs, s2_pool_outputs).mean(), negative_batch_loss)
        #print(a, b)        
        
        loss = a + b
        #print(loss)
        
        return loss  # (loss), logits, (hidden_states), (attentions)



@add_start_docstrings("""Pretrain for paraphrase with similarity. """,
    BERT_START_DOCSTRING, BERT_INPUTS_DOCSTRING)
class BertForPretrainParaphraseIndependentlyBySimilarity(BertPreTrainedModel):
    r"""
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for computing the sequence classification/regression loss.

    """
    def __init__(self, config):
        super(BertForPretrainParaphraseIndependentlyBySimilarity, self).__init__(config)

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)        
        self.cos = torch.nn.CosineSimilarity()
        #self.margin = 0.4
        #self.margin = torch.tensor([0.4], requires_grad=False)

        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, margin=None, labels=None):

        cls_s1_outputs, _ = self.bert(input_ids.select(1, 0),
                            attention_mask=attention_mask.select(1, 0),
                            token_type_ids=token_type_ids.select(1, 0),                            
                            )
        
        s1_pool_outputs = self.dropout(cls_s1_outputs)
        
        cls_s2_outputs, _ = self.bert(input_ids.select(1, 1),
                            attention_mask=attention_mask.select(1, 1),
                            token_type_ids=token_type_ids.select(1, 1),                            
                            )          
        
        s2_pool_outputs = self.dropout(cls_s2_outputs)
        # [batch_size, hidden_state_size]
        #print(s1_pool_outputs.size())
        #exit()
        similarity = self.cos(s1_pool_outputs, s2_pool_outputs)
        
        similarity = similarity * labels    # label=-1 for positive, label=1 for negative        
        
        loss = torch.mean(similarity)                        
        
        return loss  # (loss), logits, (hidden_states), (attentions)


@add_start_docstrings("""Pretrain for paraphrase with a two-gram classification task. """,
    BERT_START_DOCSTRING, BERT_INPUTS_DOCSTRING)
class BertForPretrainParaphraseWithClassificationTask(BertPreTrainedModel):
    r"""
        [CLS] sen1 [SEP] sen2 [SEP]
        label = ['0, '1']
    """
    def __init__(self, config):
        super(BertForPretrainParaphraseWithClassificationTask, self).__init__(config)
        #self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.paraphrase_classifier = nn.Linear(config.hidden_size, self.config.num_labels)        

        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):

        pooled_outputs, _ = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,               
                            )        
        pooled_outputs = self.dropout(pooled_outputs)                                        
        logits = self.paraphrase_classifier(pooled_outputs)

        outputs = (logits,)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            #loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            loss = loss_fct(logits.view(-1, 2), labels.view(-1))

            outputs = (loss,) + outputs

        return outputs  # (loss, logits)           


@add_start_docstrings("""Pretrain for paraphrase with a two-gram classification task. """,
    BERT_START_DOCSTRING, BERT_INPUTS_DOCSTRING)
class BertForPretrainParaphraseOutClsSepByClassification(BertPreTrainedModel):
    r"""
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
    """
    def __init__(self, config):
        super(BertForPretrainParaphraseOutClsSepByClassification, self).__init__(config)
        #self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.paraphrase_classifier = nn.Linear(config.hidden_size * 2, self.config.num_labels)        

        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, sep_ids=None, labels=None):
        # output [CLS] and first [SEP]
        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,               
                            sep_ids=sep_ids,
                            )
        pooled_outputs = outputs[1]
        
        pooled_outputs = self.dropout(pooled_outputs)           
        logits = self.paraphrase_classifier(pooled_outputs)

        outputs = (logits,)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            #loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            loss = loss_fct(logits.view(-1, 2), labels.view(-1))

            outputs = (loss,) + outputs

        return outputs  # (loss, logits)
           


@add_start_docstrings("""Pretrain for paraphrase with a two-gram classification task. """,
    BERT_START_DOCSTRING, BERT_INPUTS_DOCSTRING)
class BertForPretrainParaphraseIndependentlyWithClassificationTask(BertPreTrainedModel):
    r"""
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
    """
    def __init__(self, config):
        super(BertForPretrainParaphraseIndependentlyWithClassificationTask, self).__init__(config)
        #self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # [CLS1, CLS2, CLS1-CLS2, CLS1*CLS2]
        self.paraphrase_classifier = nn.Linear(config.hidden_size*4, self.config.num_labels)        
        # save the pretrained nsp classifier in this model
        self.cls = BertOnlyNSPHead(config)
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None, sen_len=None):

        a_pooled_outputs, _ = self.bert(input_ids.select(1, 0),
                            attention_mask=attention_mask.select(1, 0),
                            token_type_ids=token_type_ids.select(1, 0),
                            )        
        #a_pooled_outputs = self.dropout(a_pooled_outputs)                                        

        b_pooled_outputs, _ = self.bert(input_ids.select(1, 1),
                            attention_mask=attention_mask.select(1, 1),
                            token_type_ids=token_type_ids.select(1, 1),               
                            )        
        #b_pooled_outputs = self.dropout(b_pooled_outputs)                                        

        # fake use, for saving this parameters into new model
        self.cls(a_pooled_outputs)
        # pooled_outputs = torch.cat([a_pooled_outputs, b_pooled_outputs], -1)
        minus_pooled_outputs = a_pooled_outputs - b_pooled_outputs
        product_pooled_outputs = a_pooled_outputs * b_pooled_outputs    # element-wise product
        pooled_outputs = torch.cat([a_pooled_outputs, b_pooled_outputs, minus_pooled_outputs, product_pooled_outputs], -1)
        
        pooled_outputs = self.dropout(pooled_outputs)                                        
        logits = self.paraphrase_classifier(pooled_outputs)

        outputs = (logits,)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            #loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            loss = loss_fct(logits.view(-1, 2), labels.view(-1))

            outputs = (loss,) + outputs

        return outputs  # (loss, logits)



@add_start_docstrings("""Pretrain for paraphrase with a two-gram classification task. """,
    BERT_START_DOCSTRING, BERT_INPUTS_DOCSTRING)
class BertForPretrainParaphraseIndependentlyOutE1E2WithClassificationTask(BertPreTrainedModel):
    r"""
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
    """
    def __init__(self, config):
        super(BertForPretrainParaphraseIndependentlyOutE1E2WithClassificationTask, self).__init__(config)
        #self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # [h1, h2, h1-h2, h1 * h2], h1=[E1][E2]
        self.paraphrase_classifier = nn.Linear(config.hidden_size*8, self.config.num_labels)                
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None, sen_len=None, entity_position=None):

        a_pooled_outputs, a_e1e2_pooled_outputs = self.bert(input_ids.select(1, 0),
                            attention_mask=attention_mask.select(1, 0),
                            token_type_ids=token_type_ids.select(1, 0),               
                            entity_position=entity_position.select(1, 0),
                            )        
        #a_pooled_outputs = self.dropout(a_pooled_outputs)                                        

        b_pooled_outputs, b_e1e2_pooled_outputs = self.bert(input_ids.select(1, 1),
                            attention_mask=attention_mask.select(1, 1),
                            token_type_ids=token_type_ids.select(1, 1),
                            entity_position=entity_position.select(1, 1)
                            )        
        #b_pooled_outputs = self.dropout(b_pooled_outputs)                                        
        
        # pooled_outputs = torch.cat([a_pooled_outputs, b_pooled_outputs], -1)
        minus_pooled_outputs = a_e1e2_pooled_outputs - b_e1e2_pooled_outputs
        product_pooled_outputs = a_e1e2_pooled_outputs * b_e1e2_pooled_outputs    # element-wise product
        pooled_outputs = torch.cat([a_e1e2_pooled_outputs, b_e1e2_pooled_outputs, minus_pooled_outputs, product_pooled_outputs], -1)
        
        pooled_outputs = self.dropout(pooled_outputs)                                        
        logits = self.paraphrase_classifier(pooled_outputs)

        outputs = (logits,)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            #loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            loss = loss_fct(logits.view(-1, 2), labels.view(-1))

            outputs = (loss,) + outputs

        return outputs  # (loss, logits)




@add_start_docstrings("""Pretrain for paraphrase with a two-gram classification task. """,
    BERT_START_DOCSTRING, BERT_INPUTS_DOCSTRING)
class BertForPretrainParaphraseIndependentlyWithCosTask(BertPreTrainedModel):
    r"""
        cos(cls1, cls2)
    """
    def __init__(self, config):
        super(BertForPretrainParaphraseIndependentlyWithClassificationTask, self).__init__(config)
        #self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # [CLS1, CLS2, CLS1-CLS2, CLS1*CLS2]
        #self.paraphrase_classifier = nn.Linear(config.hidden_size*4, self.config.num_labels)        
        # save the pretrained nsp classifier in this model        
        self.cos = torch.nn.CosineSimilarity()
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None, sen_len=None):

        a_pooled_outputs, _ = self.bert(input_ids.select(1, 0),
                            attention_mask=attention_mask.select(1, 0),
                            token_type_ids=token_type_ids.select(1, 0),               
                            )        
        a_pooled_outputs = self.dropout(a_pooled_outputs)

        b_pooled_outputs, _ = self.bert(input_ids.select(1, 1),
                            attention_mask=attention_mask.select(1, 1),
                            token_type_ids=token_type_ids.select(1, 1),               
                            )                
        b_pooled_outputs = self.dropout(b_pooled_outputs)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            #loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            loss = loss_fct(logits.view(-1, 2), labels.view(-1))

            outputs = (loss,) + outputs

        return outputs  # (loss, logits)

        


@add_start_docstrings("""Pretrain for paraphrase with a two-gram classification task. """,
    BERT_START_DOCSTRING, BERT_INPUTS_DOCSTRING)
class BertForPretrainParaphraseSingleSenOutWordAvePoolingWithClassificationTask(BertPreTrainedModel):
    r"""
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:

    """
    def __init__(self, config):
        super(BertForPretrainParaphraseSingleSenOutWordAvePoolingWithClassificationTask, self).__init__(config)
        #self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # [h1, h2, h1-h2, h1*h2]
        # h is the average pooling of all words
        self.paraphrase_classifier = nn.Linear(config.hidden_size*4, self.config.num_labels)        
        # save the pretrained nsp classifier in this model
        #self.cls = BertOnlyNSPHead(config)
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None, sen_lens=None):
        """
        sel_len: [batch_size], each sen's length in this batch
        """

        a_cls_pooled_outputs, a_pooled_outputs = self.bert(input_ids.select(1, 0),
                            attention_mask=attention_mask.select(1, 0),
                            token_type_ids=token_type_ids.select(1, 0),
                            sen_lens=sen_lens.select(1, 0)
                            )
        #a_pooled_outputs = self.dropout(a_pooled_outputs)                                        

        b_cls_pooled_outputs, b_pooled_outputs = self.bert(input_ids.select(1, 1),
                            attention_mask=attention_mask.select(1, 1),
                            token_type_ids=token_type_ids.select(1, 1),               
                            sen_lens=sen_lens.select(1, 0)
                            )        
        #b_pooled_outputs = self.dropout(b_pooled_outputs)                                        

        # fake use, for saving this parameters into new model
        #self.cls(a_pooled_outputs)
        # pooled_outputs = torch.cat([a_pooled_outputs, b_pooled_outputs], -1)
        minus_pooled_outputs = a_pooled_outputs - b_pooled_outputs
        product_pooled_outputs = a_pooled_outputs * b_pooled_outputs    # element-wise product
        pooled_outputs = torch.cat([a_pooled_outputs, b_pooled_outputs, minus_pooled_outputs, product_pooled_outputs], -1)
        
        pooled_outputs = self.dropout(pooled_outputs)                                        
        logits = self.paraphrase_classifier(pooled_outputs)

        outputs = (logits,)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            #loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            loss = loss_fct(logits.view(-1, 2), labels.view(-1))

            outputs = (loss,) + outputs

        return outputs  # (loss, logits)



@add_start_docstrings("""Pretrain for paraphrase with a two-gram classification task. """,
    BERT_START_DOCSTRING, BERT_INPUTS_DOCSTRING)
class BertForPretrainParaphraseByMultiTask(BertPreTrainedModel):
    r"""
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
    """
    def __init__(self, config):
        super(BertForPretrainParaphraseByMultiTask, self).__init__(config)
        #self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # classify [CLS] sen1 [SEP] sen2 [SEP]
        assert self.config.num_labels == 2
        self.paraphrase_classifier = nn.Linear(config.hidden_size, self.config.num_labels)        
        # for calculate the similarity between sen1's [CLS] and sen2's [CLS]
        self.cos = torch.nn.CosineSimilarity()
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        # Task2: similarity of two sentences
        sen1_outputs = self.bert(input_ids.select(1, 1),
                            attention_mask=attention_mask.select(1, 1),
                            token_type_ids=token_type_ids.select(1, 1),               
                            )
        sen1_pooled_outputs = sen1_outputs[1]
        sen1_pooled_outputs = self.dropout(sen1_pooled_outputs)                                        

        sen2_outputs = self.bert(input_ids.select(1, 2),
                            attention_mask=attention_mask.select(1, 2),
                            token_type_ids=token_type_ids.select(1, 2),               
                            )
        sen2_pooled_outputs = sen2_outputs[1]
        sen2_pooled_outputs = self.dropout(sen2_pooled_outputs)                                        
        similarity = self.cos(sen1_pooled_outputs, sen2_pooled_outputs)
        similarity_loss = 1 / torch.exp(similarity - 1)
        similarity_loss = torch.mean(similarity_loss)

        # Task1: classify sen pair, is paraphrase or not
        senpair_outputs = self.bert(input_ids.select(1, 0),
                            attention_mask=attention_mask.select(1, 0),
                            token_type_ids=token_type_ids.select(1, 0),               
                            )
        senpair_pooled_outputs = senpair_outputs[1]
        senpair_pooled_outputs = self.dropout(senpair_pooled_outputs)                                        

        logits = self.paraphrase_classifier(senpair_pooled_outputs)

        outputs = (logits,)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            #loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            classify_loss = loss_fct(logits.view(-1, 2), labels.view(-1))
            loss = classify_loss * similarity_loss
            outputs = (loss,) + outputs

        return outputs  # (loss, logits)



@add_start_docstrings("""Pretrain for paraphrase with a two-gram classification task. """,
    BERT_START_DOCSTRING, BERT_INPUTS_DOCSTRING)
class BertForParaphraseRcMultiTask(BertPreTrainedModel):
    r"""
        TODO multi-task learning
    """
    def __init__(self, config):
        super(BertForParaphraseRcMultiTask, self).__init__(config)
        #self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # classify [CLS] sen1 [SEP] sen2 [SEP]
        # assert self.config.num_labels == 2
        self.paraphrase_classifier = nn.Linear(config.hidden_size, self.config.num_labels)        
        # for calculate the similarity between sen1's [CLS] and sen2's [CLS]
        self.cos = torch.nn.CosineSimilarity()
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        # Task2: similarity of two sentences
        sen1_outputs = self.bert(input_ids.select(1, 1),
                            attention_mask=attention_mask.select(1, 1),
                            token_type_ids=token_type_ids.select(1, 1),               
                            )
        sen1_pooled_outputs = sen1_outputs[1]
        sen1_pooled_outputs = self.dropout(sen1_pooled_outputs)                                        

        sen2_outputs = self.bert(input_ids.select(1, 2),
                            attention_mask=attention_mask.select(1, 2),
                            token_type_ids=token_type_ids.select(1, 2),               
                            )
        sen2_pooled_outputs = sen2_outputs[1]
        sen2_pooled_outputs = self.dropout(sen2_pooled_outputs)                                        
        similarity = self.cos(sen1_pooled_outputs, sen2_pooled_outputs)
        similarity_loss = 1 / torch.exp(similarity - 1)
        similarity_loss = torch.mean(similarity_loss)

        # Task1: classify sen pair, is paraphrase or not
        senpair_outputs = self.bert(input_ids.select(1, 0),
                            attention_mask=attention_mask.select(1, 0),
                            token_type_ids=token_type_ids.select(1, 0),               
                            )
        senpair_pooled_outputs = senpair_outputs[1]
        senpair_pooled_outputs = self.dropout(senpair_pooled_outputs)                                        

        logits = self.paraphrase_classifier(senpair_pooled_outputs)

        outputs = (logits,)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            #loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            classify_loss = loss_fct(logits.view(-1, 2), labels.view(-1))
            loss = classify_loss * similarity_loss
            outputs = (loss,) + outputs

        return outputs  # (loss, logits)



@add_start_docstrings("""Bert Model transformer with a relation classification head on top (a linear layer on top of
    the pooled output) e.g. for SemEval tasks. """,
    BERT_START_DOCSTRING, BERT_INPUTS_DOCSTRING)
class BertForRelationClassificationWithCLS(BertPreTrainedModel):
    r"""
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for computing the sequence classification/regression loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.
            If ``config.num_labels == 1`` a regression loss is computed (Mean-Square loss),
            If ``config.num_labels > 1`` a classification loss is computed (Cross-Entropy).

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Classification (or regression if config.num_labels==1) loss.
        **logits**: ``torch.FloatTensor`` of shape ``(batch_size, config.num_labels)``
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForRelationClassification.from_pretrained('bert-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, logits = outputs[:2]

    """
    def __init__(self, config):
        super(BertForRelationClassificationWithCLS, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size*3, self.config.num_labels)

        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, labels=None, entity_position=None):

        cls_outputs, entity_outputs = self.bert(input_ids,
                                    attention_mask=attention_mask,
                                    token_type_ids=token_type_ids,
                                    position_ids=position_ids, 
                                    head_mask=head_mask,
                                    entity_position=entity_position)
        # pooled_output=[batch_size, hidden_size*2]
                        
        pooled_outputs = torch.cat([cls_outputs, entity_outputs], 1)
        pooled_outputs = self.dropout(pooled_outputs)
        logits = self.classifier(pooled_outputs)

        outputs = (logits,)

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits



@add_start_docstrings("""Bert Model transformer with a relation classification head on top (a linear layer on top of
    the pooled output) e.g. for SemEval tasks. """,
    BERT_START_DOCSTRING, BERT_INPUTS_DOCSTRING)
class BertForRelationClassificationWithRawSenCLS(BertPreTrainedModel):
    r"""
        input: raw sen
        output: [CLS]
        model is pretrained BERT used for outputting whether two sentences express a same meaning
    """
    def __init__(self, config):
        super(BertForRelationClassificationWithRawSenCLS, self).__init__(config)        
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        #self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)

        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, labels=None, entity_position=None):

        cls_pooled_output, _ = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids, 
                            head_mask=head_mask,
                            entity_position=None)
        # dropout will be did in next model
        # cls_pooled_output = self.dropout(cls_pooled_output)                    

        return cls_pooled_output


@add_start_docstrings("""Bert Model transformer with a relation classification with share BERT 
for incorporating with translated data
head on top (a linear layer on top of the pooled output) e.g. for SemEval tasks. """,
    BERT_START_DOCSTRING, BERT_INPUTS_DOCSTRING)
class BertForRelationClassificationWithShareBertShareClassifier(BertPreTrainedModel):
    r"""
    output = loss1 + alpha * (loss2)
    Use two machine translations results to save cuda memory
    Since we need to input with bag
    """
    def __init__(self, config):
        super(BertForRelationClassificationWithShareBertShareClassifier, self).__init__(config)
        self.num_labels = config.num_labels
        # Need to modify the parameters's name in pretrained model, 
        # We add a new classmethod from_pretrained_for_independent_bert in modeling_utils         
        self.bert = BertModel(config)        
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size*2, self.config.num_labels)
        
        #self.alpha_1 = nn.Parameter(torch.Tensor([1.0]))
        #self.alpha_2 = nn.Parameter(torch.Tensor([1.0]))
        #self.alpha_3 = nn.Parameter(torch.Tensor([1.0]))
        #self.alpha_4 = nn.Parameter(torch.Tensor([1.0]))
        self.init_weights()

    def forward(self, mt_system, mode, loss_weight, input_ids, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, labels=None, entity_position=None):
        """
        mt_system=google, baidu, xiaoniu, all
        mode = train/test
        combine_mode = "plus", "concate"
        Input is [batch_size, 4, *], 4 for human, google, baidu and xiaoniu, we need to select
        """        

        _, human_e1e2_outputs = self.bert(input_ids.select(1, 0),
                                        attention_mask=attention_mask.select(1, 0),
                                        token_type_ids=token_type_ids.select(1, 0),
                                        position_ids=position_ids, 
                                        head_mask=head_mask,
                                        entity_position=entity_position.select(1, 0))
        # pooled_output=[batch_size, hidden_size*2]
        human_pooled_output = human_e1e2_outputs
        human_pooled_output = self.dropout(human_pooled_output)                  
        human_logits = self.classifier(human_pooled_output)              
        if mode == "train":
            # For training
            if mt_system == "google":
                _, google_e1e2_outputs = self.bert(input_ids.select(1, 1),
                                    attention_mask=attention_mask.select(1, 1),
                                    token_type_ids=token_type_ids.select(1, 1),
                                    position_ids=position_ids, 
                                    head_mask=head_mask,
                                    entity_position=entity_position.select(1, 1))            
                google_pooled_output = google_e1e2_outputs
                google_pooled_output = self.dropout(google_pooled_output)                              
                
                trans_logits = self.classifier(google_pooled_output)
            elif mt_system == "baidu":
                _, baidu_e1e2_outputs = self.bert(input_ids.select(1, 2),
                                                attention_mask=attention_mask.select(1, 2),
                                                token_type_ids=token_type_ids.select(1, 2),
                                                position_ids=position_ids, 
                                                head_mask=head_mask,
                                                entity_position=entity_position.select(1, 2))            
                baidu_pooled_output = baidu_e1e2_outputs
                baidu_pooled_output = self.dropout(baidu_pooled_output)  
                trans_logits = self.classifier(baidu_pooled_output)
            elif mt_system == "xiaoniu":
                _, xiaoniu_e1e2_outputs = self.bert(input_ids.select(1, 3),
                                                attention_mask=attention_mask.select(1, 3),
                                                token_type_ids=token_type_ids.select(1, 3),
                                                position_ids=position_ids, 
                                                head_mask=head_mask,
                                                entity_position=entity_position.select(1, 3))            
                xiaoniu_pooled_output = xiaoniu_e1e2_outputs
                xiaoniu_pooled_output = self.dropout(xiaoniu_pooled_output)     
                trans_logits = self.classifier(xiaoniu_pooled_output)              
            else:
                raise ValueError(f"Error MT system {mt_system}")

        outputs = (human_logits,) #  + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(human_logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                human_loss = loss_fct(human_logits.view(-1, self.num_labels), labels.view(-1))
                if mode == "train":                    
                    trans_loss = loss_fct(trans_logits.view(-1, self.num_labels), labels.view(-1))                    
                    #xiaoniu_loss = loss_fct(xiaoniu_logits.view(-1, self.num_labels), labels.view(-1))
                    loss = human_loss + loss_weight * trans_loss
                else:
                    loss = human_loss        
            outputs = (loss,) + outputs
        
        # Check the final alpha value        
        return outputs  # (loss), logits




@add_start_docstrings("""Bert Model transformer with a relation classification with share BERT 
for incorporating with translated data
head on top (a linear layer on top of the pooled output) e.g. for SemEval tasks. """,
    BERT_START_DOCSTRING, BERT_INPUTS_DOCSTRING)
class BertForRelationClassificationWithShareBertShareClassifierMaxTrans(BertPreTrainedModel):
    r"""
    trans_loss = cross_entropy( max{google_repre, baidu_repre, xiaoniu_repre} )
    output = loss1 + alpha * (trans_loss)
    """
    def __init__(self, config):
        super(BertForRelationClassificationWithShareBertShareClassifierMaxTrans, self).__init__(config)
        self.num_labels = config.num_labels
        # Need to modify the parameters's name in pretrained model, 
        # We add a new classmethod from_pretrained_for_independent_bert in modeling_utils         
        self.bert = BertModel(config)        
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size*2, self.config.num_labels)
        
        #self.alpha_1 = nn.Parameter(torch.Tensor([1.0]))
        #self.alpha_2 = nn.Parameter(torch.Tensor([1.0]))
        #self.alpha_3 = nn.Parameter(torch.Tensor([1.0]))
        #self.alpha_4 = nn.Parameter(torch.Tensor([1.0]))
        self.init_weights()

    def forward(self, mt_system, mode, loss_weight, input_ids, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, labels=None, entity_position=None, select_mode=None):
        """
        select_mode=max/ave
        mt_system=google, baidu, xiaoniu, all
        mode = train/test
        combine_mode = "plus", "concate"
        Input is [batch_size, 4, *], 4 for human, google, baidu and xiaoniu, we need to select
        TODO
        """        

        _, human_e1e2_outputs = self.bert(input_ids.select(1, 0),
                                        attention_mask=attention_mask.select(1, 0),
                                        token_type_ids=token_type_ids.select(1, 0),
                                        position_ids=position_ids, 
                                        head_mask=head_mask,
                                        entity_position=entity_position.select(1, 0))
        # pooled_output=[batch_size, hidden_size*2]
        human_pooled_output = human_e1e2_outputs
        human_pooled_output = self.dropout(human_pooled_output)                  
        human_logits = self.classifier(human_pooled_output)              
        if mode == "train":
            # For training
            _, google_e1e2_outputs = self.bert(input_ids.select(1, 1),
                                attention_mask=attention_mask.select(1, 1),
                                token_type_ids=token_type_ids.select(1, 1),
                                position_ids=position_ids, 
                                head_mask=head_mask,
                                entity_position=entity_position.select(1, 1))            
            google_pooled_output = google_e1e2_outputs
            google_pooled_output = self.dropout(google_pooled_output)
            
            _, baidu_e1e2_outputs = self.bert(input_ids.select(1, 2),
                                            attention_mask=attention_mask.select(1, 2),
                                            token_type_ids=token_type_ids.select(1, 2),
                                            position_ids=position_ids, 
                                            head_mask=head_mask,
                                            entity_position=entity_position.select(1, 2))            
            baidu_pooled_output = baidu_e1e2_outputs
            baidu_pooled_output = self.dropout(baidu_pooled_output)  
            
            _, xiaoniu_e1e2_outputs = self.bert(input_ids.select(1, 3),
                                            attention_mask=attention_mask.select(1, 3),
                                            token_type_ids=token_type_ids.select(1, 3),
                                            position_ids=position_ids, 
                                            head_mask=head_mask,
                                            entity_position=entity_position.select(1, 3))            
            xiaoniu_pooled_output = xiaoniu_e1e2_outputs
            xiaoniu_pooled_output = self.dropout(xiaoniu_pooled_output)  
            bag_trans_pooled_output = torch.cat([google_pooled_output.unsqueeze(0), baidu_pooled_output.unsqueeze(0), xiaoniu_pooled_output.unsqueeze(0)], 0)
            bag_max_trans_pooled_output, _ = torch.max(bag_trans_pooled_output, dim=0)        
            bag_trans_logits = self.classifier(bag_max_trans_pooled_output)
            """
            google_logits = self.classifier(google_pooled_output)
            baidu_logits = self.classifier(baidu_pooled_output)
            xiaoniu_logits = self.classifier(xiaoniu_pooled_output)
            """

        outputs = (human_logits,) #  + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(human_logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                human_loss = loss_fct(human_logits.view(-1, self.num_labels), labels.view(-1))
                if mode == "train":                    
                    bag_trans_loss = loss_fct(bag_trans_logits.view(-1, self.num_labels), labels.view(-1))
                    loss = human_loss + loss_weight * (bag_trans_loss)
                else:
                    loss = human_loss        
            outputs = (loss,) + outputs
        
        # Check the final alpha value        
        return outputs  # (loss), logits



@add_start_docstrings("""Bert Model transformer with a relation classification with share BERT 
for incorporating with translated data
head on top (a linear layer on top of the pooled output) e.g. for SemEval tasks. """,
    BERT_START_DOCSTRING, BERT_INPUTS_DOCSTRING)
class BertForRelationClassificationWithShareBertShareClassifierAveTrans(BertPreTrainedModel):
    r"""
    trans_loss = cross_entropy( ave{google_repre, baidu_repre, xiaoniu_repre} )
    output = loss1 + alpha * (trans_loss)}
    """
    def __init__(self, config):
        super(BertForRelationClassificationWithShareBertShareClassifierAveTrans, self).__init__(config)
        self.num_labels = config.num_labels
        # Need to modify the parameters's name in pretrained model, 
        # We add a new classmethod from_pretrained_for_independent_bert in modeling_utils         
        self.bert = BertModel(config)        
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size*2, self.config.num_labels)
        
        #self.alpha_1 = nn.Parameter(torch.Tensor([1.0]))
        #self.alpha_2 = nn.Parameter(torch.Tensor([1.0]))
        #self.alpha_3 = nn.Parameter(torch.Tensor([1.0]))
        #self.alpha_4 = nn.Parameter(torch.Tensor([1.0]))
        self.init_weights()

    def forward(self, mt_system, mode, loss_weight, input_ids, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, labels=None, entity_position=None, select_mode=None):
        """
        select_mode=max/ave
        mt_system=google, baidu, xiaoniu, all
        mode = train/test
        combine_mode = "plus", "concate"
        Input is [batch_size, 4, *], 4 for human, google, baidu and xiaoniu, we need to select
        TODO
        """        

        _, human_e1e2_outputs = self.bert(input_ids.select(1, 0),
                                        attention_mask=attention_mask.select(1, 0),
                                        token_type_ids=token_type_ids.select(1, 0),
                                        position_ids=position_ids, 
                                        head_mask=head_mask,
                                        entity_position=entity_position.select(1, 0))
        # pooled_output=[batch_size, hidden_size*2]
        human_pooled_output = human_e1e2_outputs
        human_pooled_output = self.dropout(human_pooled_output)                  
        human_logits = self.classifier(human_pooled_output)              
        if mode == "train":
            # For training
            _, google_e1e2_outputs = self.bert(input_ids.select(1, 1),
                                attention_mask=attention_mask.select(1, 1),
                                token_type_ids=token_type_ids.select(1, 1),
                                position_ids=position_ids, 
                                head_mask=head_mask,
                                entity_position=entity_position.select(1, 1))            
            google_pooled_output = google_e1e2_outputs
            google_pooled_output = self.dropout(google_pooled_output)
            
            _, baidu_e1e2_outputs = self.bert(input_ids.select(1, 2),
                                            attention_mask=attention_mask.select(1, 2),
                                            token_type_ids=token_type_ids.select(1, 2),
                                            position_ids=position_ids, 
                                            head_mask=head_mask,
                                            entity_position=entity_position.select(1, 2))            
            baidu_pooled_output = baidu_e1e2_outputs
            baidu_pooled_output = self.dropout(baidu_pooled_output)  
            
            _, xiaoniu_e1e2_outputs = self.bert(input_ids.select(1, 3),
                                            attention_mask=attention_mask.select(1, 3),
                                            token_type_ids=token_type_ids.select(1, 3),
                                            position_ids=position_ids, 
                                            head_mask=head_mask,
                                            entity_position=entity_position.select(1, 3))            
            xiaoniu_pooled_output = xiaoniu_e1e2_outputs
            xiaoniu_pooled_output = self.dropout(xiaoniu_pooled_output)  
            bag_trans_pooled_output = torch.cat([google_pooled_output.unsqueeze(0), baidu_pooled_output.unsqueeze(0), xiaoniu_pooled_output.unsqueeze(0)], 0)
            bag_ave_trans_pooled_output = torch.mean(bag_trans_pooled_output, dim=0)        
            bag_trans_logits = self.classifier(bag_ave_trans_pooled_output)
            """
            google_logits = self.classifier(google_pooled_output)
            baidu_logits = self.classifier(baidu_pooled_output)
            xiaoniu_logits = self.classifier(xiaoniu_pooled_output)
            """

        outputs = (human_logits,) #  + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(human_logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                human_loss = loss_fct(human_logits.view(-1, self.num_labels), labels.view(-1))
                if mode == "train":                    
                    bag_trans_loss = loss_fct(bag_trans_logits.view(-1, self.num_labels), labels.view(-1))
                    loss = human_loss + loss_weight * (bag_trans_loss)
                else:
                    loss = human_loss        
            outputs = (loss,) + outputs
        
        # Check the final alpha value        
        return outputs  # (loss), logits




@add_start_docstrings("""Bert Model transformer with a relation classification with share BERT 
for incorporating with translated data
head on top (a linear layer on top of the pooled output) e.g. for SemEval tasks. """,
    BERT_START_DOCSTRING, BERT_INPUTS_DOCSTRING)
class BertForRelationClassificationWithShareBertShareClassifierOneTrans(BertPreTrainedModel):
    r"""
    trans_loss = cross_entropy(one{google_repre, baidu_repre, xiaoniu_repre} )
    output = loss1 + one * (trans_loss)}
    select the sentence with max-prob on labeled relation during training
    """
    def __init__(self, config):
        super(BertForRelationClassificationWithShareBertShareClassifierOneTrans, self).__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        # Need to modify the parameters's name in pretrained model, 
        # We add a new classmethod from_pretrained_for_independent_bert in modeling_utils         
        self.bert = BertModel(config)        
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size*2, self.config.num_labels)
        #self.attention = nn.Parameter(torch.ones(config.hidden_size*2)) # will been changed into a diag during training
        
        self.init_weights()

    def forward(self, mt_system, mode, loss_weight, input_ids, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, labels=None, entity_position=None, select_mode=None):
        """
        select_mode=max/ave
        mt_system=google, baidu, xiaoniu, all
        mode = train/test
        combine_mode = "plus", "concate"
        Input is [batch_size, 4, *], 4 for human, google, baidu and xiaoniu, we need to select
        TODO
        """        

        _, human_e1e2_outputs = self.bert(input_ids.select(1, 0),
                                        attention_mask=attention_mask.select(1, 0),
                                        token_type_ids=token_type_ids.select(1, 0),
                                        position_ids=position_ids, 
                                        head_mask=head_mask,
                                        entity_position=entity_position.select(1, 0))
        # pooled_output=[batch_size, hidden_size*2]
        human_pooled_output = human_e1e2_outputs
        human_pooled_output = self.dropout(human_pooled_output)                  
        human_logits = self.classifier(human_pooled_output)              
        if mode == "train":
            # For training
            _, google_e1e2_outputs = self.bert(input_ids.select(1, 1),
                                attention_mask=attention_mask.select(1, 1),
                                token_type_ids=token_type_ids.select(1, 1),
                                position_ids=position_ids, 
                                head_mask=head_mask,
                                entity_position=entity_position.select(1, 1))            
            google_pooled_output = google_e1e2_outputs
            google_pooled_output = self.dropout(google_pooled_output)
            
            _, baidu_e1e2_outputs = self.bert(input_ids.select(1, 2),
                                            attention_mask=attention_mask.select(1, 2),
                                            token_type_ids=token_type_ids.select(1, 2),
                                            position_ids=position_ids, 
                                            head_mask=head_mask,
                                            entity_position=entity_position.select(1, 2))            
            baidu_pooled_output = baidu_e1e2_outputs
            baidu_pooled_output = self.dropout(baidu_pooled_output)  
            
            _, xiaoniu_e1e2_outputs = self.bert(input_ids.select(1, 3),
                                            attention_mask=attention_mask.select(1, 3),
                                            token_type_ids=token_type_ids.select(1, 3),
                                            position_ids=position_ids, 
                                            head_mask=head_mask,
                                            entity_position=entity_position.select(1, 3))            
            xiaoniu_pooled_output = xiaoniu_e1e2_outputs
            xiaoniu_pooled_output = self.dropout(xiaoniu_pooled_output)  


            # This should be [batch_size, 3, hidden_size*2]
            # [batch, hidden] -> [batch, 1, hidden] after cat -> [batch, 3, hidden]

            bag_trans_pooled_output = torch.cat([google_pooled_output.unsqueeze(1), baidu_pooled_output.unsqueeze(1), xiaoniu_pooled_output.unsqueeze(1)], 1)            
            
            # [batch, 3, hidden] - > [batch*3, hidden] * [hidden, num_label] = [batch*3, num_label] -> [batch, 3, num_label]
            # if we use google/baidu/xiaoniu, then it is 3
            batch_bag_trans_logits = self.classifier(bag_trans_pooled_output.view(-1, self.config.hidden_size*2)).view(-1, 3, self.config.num_labels)
            # [batch, 3, num_label] -> [batch, 3, 1] -> [batch, 3]
            batch_bag_probs = torch.softmax(batch_bag_trans_logits, -1)            
            #print(f"batch_bag_probs: {batch_bag_probs.size()}")
            #print(f"labels: {labels.size()}")
            batch_bag_probs_on_gold = []
            for i in range(batch_bag_probs.size()[0]):    # for each bag
                bag_probs = batch_bag_probs[i]  #[bag_size, num_labels]
                gold_label = labels[i]
                bag_probs_on_gold = bag_probs[:, gold_label]    #[bag_size, 1]
                batch_bag_probs_on_gold.append(bag_probs_on_gold)
            #[batch, bag_size]
            batch_bag_probs_on_gold = torch.stack(batch_bag_probs_on_gold, 0)
            #print(f"batch_bag_probs_on_gold: {batch_bag_probs_on_gold.size()}")
            # return which sentence: [batch]
            _, max_index = torch.max(batch_bag_probs_on_gold, -1)
            batch_one_trans_logits = []
            # use logits not probs because we use CrossEntropy whcih alerady combined with softmax
            for i in range(batch_bag_trans_logits.size()[0]):    # for each bag
                max_sen_index = max_index[i]
                one_trans_logits = batch_bag_trans_logits[i][max_sen_index]
                batch_one_trans_logits.append(one_trans_logits)
            batch_one_trans_logits = torch.stack(batch_one_trans_logits, 0)
            #print(f"batch_one_trans_logits: {batch_one_trans_logits.size()}")            
        outputs = (human_logits,) #  + outputs[2:]  # add hidden states and attention if they are here        

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(human_logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                human_loss = loss_fct(human_logits.view(-1, self.num_labels), labels.view(-1))
                if mode == "train":                    
                    bag_trans_loss = loss_fct(batch_one_trans_logits.view(-1, self.num_labels), labels.view(-1))
                    loss = human_loss + loss_weight * (bag_trans_loss)
                else:
                    loss = human_loss        
            outputs = (loss,) + outputs        
        # Check the final alpha value        
        return outputs  # (loss), logits




@add_start_docstrings("""Bert Model transformer with a relation classification with share BERT 
for incorporating with translated data
head on top (a linear layer on top of the pooled output) e.g. for SemEval tasks. """,
    BERT_START_DOCSTRING, BERT_INPUTS_DOCSTRING)
class BertForRelationClassificationWithShareBertShareClassifierAttTrans(BertPreTrainedModel):
    r"""
    trans_loss = cross_entropy( att{google_repre, baidu_repre, xiaoniu_repre} )
    output = loss1 + alpha * (trans_loss)}
    """
    def __init__(self, config):
        super(BertForRelationClassificationWithShareBertShareClassifierAttTrans, self).__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        # Need to modify the parameters's name in pretrained model, 
        # We add a new classmethod from_pretrained_for_independent_bert in modeling_utils         
        self.bert = BertModel(config)        
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size*2, self.config.num_labels)
        self.attention = nn.Parameter(torch.ones(config.hidden_size*2)) # will been changed into a diag during training
        #self.alpha_1 = nn.Parameter(torch.Tensor([1.0]))
        #self.alpha_2 = nn.Parameter(torch.Tensor([1.0]))
        #self.alpha_3 = nn.Parameter(torch.Tensor([1.0]))
        #self.alpha_4 = nn.Parameter(torch.Tensor([1.0]))
        self.init_weights()

    def forward(self, mt_system, mode, loss_weight, input_ids, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, labels=None, entity_position=None, select_mode=None):
        """
        select_mode=max/ave
        mt_system=google, baidu, xiaoniu, all
        mode = train/test
        combine_mode = "plus", "concate"
        Input is [batch_size, 4, *], 4 for human, google, baidu and xiaoniu, we need to select
        TODO
        """        

        _, human_e1e2_outputs = self.bert(input_ids.select(1, 0),
                                        attention_mask=attention_mask.select(1, 0),
                                        token_type_ids=token_type_ids.select(1, 0),
                                        position_ids=position_ids, 
                                        head_mask=head_mask,
                                        entity_position=entity_position.select(1, 0))
        # pooled_output=[batch_size, hidden_size*2]
        human_pooled_output = human_e1e2_outputs
        human_pooled_output = self.dropout(human_pooled_output)                  
        human_logits = self.classifier(human_pooled_output)              
        if mode == "train":
            # For training
            _, google_e1e2_outputs = self.bert(input_ids.select(1, 1),
                                attention_mask=attention_mask.select(1, 1),
                                token_type_ids=token_type_ids.select(1, 1),
                                position_ids=position_ids, 
                                head_mask=head_mask,
                                entity_position=entity_position.select(1, 1))            
            google_pooled_output = google_e1e2_outputs
            google_pooled_output = self.dropout(google_pooled_output)
            
            
            _, baidu_e1e2_outputs = self.bert(input_ids.select(1, 2),
                                            attention_mask=attention_mask.select(1, 2),
                                            token_type_ids=token_type_ids.select(1, 2),
                                            position_ids=position_ids, 
                                            head_mask=head_mask,
                                            entity_position=entity_position.select(1, 2))            
            baidu_pooled_output = baidu_e1e2_outputs
            baidu_pooled_output = self.dropout(baidu_pooled_output)  
            
            _, xiaoniu_e1e2_outputs = self.bert(input_ids.select(1, 3),
                                            attention_mask=attention_mask.select(1, 3),
                                            token_type_ids=token_type_ids.select(1, 3),
                                            position_ids=position_ids, 
                                            head_mask=head_mask,
                                            entity_position=entity_position.select(1, 3))            
            xiaoniu_pooled_output = xiaoniu_e1e2_outputs
            xiaoniu_pooled_output = self.dropout(xiaoniu_pooled_output)  


            # This should be [batch_size, 3, hidden_size*2]
            # [batch, hidden] -> [batch, 1, hidden] after cat -> [batch, 3, hidden]
            bag_trans_pooled_output = torch.cat([google_pooled_output.unsqueeze(1), baidu_pooled_output.unsqueeze(1), xiaoniu_pooled_output.unsqueeze(1)], 1)            
#            print("bag_trans_pooled_output")
#            print(bag_trans_pooled_output.size())
            diag_attention = torch.diag(self.attention)
#            print(f"diag attention={diag_attention.size()}")
            # [batch*bag_size, hidden*2] [hidden*2, hidden*2]
            xA = torch.matmul(bag_trans_pooled_output.view(-1, self.config.hidden_size * 2), diag_attention)
            # [batch, bag_size, hidden_size*2]
            #xA = xA.view(self.config.per_gpu_train_batch_size, 3, self.config.hidden_size)
#            print(f"xA:{xA.size()}")
            xA = xA.view(-1, 3, self.config.hidden_size*2) #[batch, bag_size, hidden*2]            

#            print(f"xA:{xA.size()}")
            # build gold-label's r for this batch= [batch*bag_size, hidden*2]'s transpose
            #label is based on bag, first change to based on sentence, each bag has 3 sentences, this 3 sens share a same bag label
            #e.g.: [1,2] ->[[1],[2]] -> [[1,1,1], [2,2,2]] -> [1,1,1,2,2,2]
            #Error ! labels = torch.flatten(labels.unsqueeze(-1).expand(-1, 3))   #[batch] -> [batch, 1] -> [batch, 3] -> [batch*3]                
            r = self.classifier.weight[labels, :].unsqueeze(-1)   # [hidden, num_labels] -> [batch, hidden, 1] -> [batch, hidden]
#            print(f"r: {r.size()}")
            xAr = torch.matmul(xA, r)
            # xAr=[batch_size, 3, 1]
#            print(f"xAr: {xAr.size()}")
            xAr = xAr.squeeze()
            # Then calculate alpha by softmax
            alpha = torch.softmax(xAr, -1)  # [batch,3]
#            print(alpha)
#            print(alpha.size())
            # Finally, weighted sum of hidden states
            # [batch, 1, 3] mm [batch, 3, hidden] = [batch, hidden]
            alpha = alpha.unsqueeze(1)
#            print(f"alpha {alpha.size()}")
#            print(f"bag_trans_pooled_output: {bag_trans_pooled_output.size()}")
            #[batch, 1, hidden] - > [batch, hidden]
            att_trans_pooled_output = torch.matmul(alpha, bag_trans_pooled_output).squeeze()
#            print(f"att bag output: {att_trans_pooled_output.size()}")
             
            bag_trans_logits = self.classifier(att_trans_pooled_output)

        outputs = (human_logits,) #  + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(human_logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                human_loss = loss_fct(human_logits.view(-1, self.num_labels), labels.view(-1))
                if mode == "train":                    
                    bag_trans_loss = loss_fct(bag_trans_logits.view(-1, self.num_labels), labels.view(-1))
                    loss = human_loss + loss_weight * (bag_trans_loss)
                else:
                    loss = human_loss        
            outputs = (loss,) + outputs
        
        # Check the final alpha value        
        return outputs  # (loss), logits




@add_start_docstrings("""Bert Model transformer with a relation classification with share BERT 
for incorporating with translated data
head on top (a linear layer on top of the pooled output) e.g. for SemEval tasks. """,
    BERT_START_DOCSTRING, BERT_INPUTS_DOCSTRING)
class BertForRelationClassificationWithShareBertShareClassifierAttTransWithSimilarity(BertPreTrainedModel):
    r"""
    trans_loss = cross_entropy( att{google_repre, baidu_repre, xiaoniu_repre} )
    output = loss1 + alpha * (similarity * trans_loss)}
    """
    def __init__(self, config):
        super(BertForRelationClassificationWithShareBertShareClassifierAttTransWithSimilarity, self).__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        # Need to modify the parameters's name in pretrained model, 
        # We add a new classmethod from_pretrained_for_independent_bert in modeling_utils         
        self.bert = BertModel(config)        
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size*2, self.config.num_labels)
        self.attention = nn.Parameter(torch.ones(config.hidden_size*2)) # will been changed into a diag during training        
        self.cos = torch.nn.CosineSimilarity()
        self.init_weights()

    def forward(self, mt_system, mode, loss_weight, input_ids, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, labels=None, entity_position=None, select_mode=None):
        """
        select_mode=max/ave
        mt_system=google, baidu, xiaoniu, all
        mode = train/test
        combine_mode = "plus", "concate"
        Input is [batch_size, 4, *], 4 for human, google, baidu and xiaoniu, we need to select
        TODO
        """        

        _, human_e1e2_outputs = self.bert(input_ids.select(1, 0),
                                        attention_mask=attention_mask.select(1, 0),
                                        token_type_ids=token_type_ids.select(1, 0),
                                        position_ids=position_ids, 
                                        head_mask=head_mask,
                                        entity_position=entity_position.select(1, 0))
        # pooled_output=[batch_size, hidden_size*2]
        human_pooled_output = human_e1e2_outputs
        human_pooled_output = self.dropout(human_pooled_output)                  
        human_logits = self.classifier(human_pooled_output)              
        if mode == "train":
            # For training
            _, google_e1e2_outputs = self.bert(input_ids.select(1, 1),
                                attention_mask=attention_mask.select(1, 1),
                                token_type_ids=token_type_ids.select(1, 1),
                                position_ids=position_ids, 
                                head_mask=head_mask,
                                entity_position=entity_position.select(1, 1))            
            google_pooled_output = google_e1e2_outputs
            google_pooled_output = self.dropout(google_pooled_output)
            
            
            _, baidu_e1e2_outputs = self.bert(input_ids.select(1, 2),
                                            attention_mask=attention_mask.select(1, 2),
                                            token_type_ids=token_type_ids.select(1, 2),
                                            position_ids=position_ids, 
                                            head_mask=head_mask,
                                            entity_position=entity_position.select(1, 2))            
            baidu_pooled_output = baidu_e1e2_outputs
            baidu_pooled_output = self.dropout(baidu_pooled_output)  
            
            _, xiaoniu_e1e2_outputs = self.bert(input_ids.select(1, 3),
                                            attention_mask=attention_mask.select(1, 3),
                                            token_type_ids=token_type_ids.select(1, 3),
                                            position_ids=position_ids, 
                                            head_mask=head_mask,
                                            entity_position=entity_position.select(1, 3))            
            xiaoniu_pooled_output = xiaoniu_e1e2_outputs
            xiaoniu_pooled_output = self.dropout(xiaoniu_pooled_output)  


            # This should be [batch_size, 3, hidden_size*2]
            # [batch, hidden] -> [batch, 1, hidden] after cat -> [batch, 3, hidden]
            bag_trans_pooled_output = torch.cat([google_pooled_output.unsqueeze(1), baidu_pooled_output.unsqueeze(1), xiaoniu_pooled_output.unsqueeze(1)], 1)            
#            print("bag_trans_pooled_output")
#            print(bag_trans_pooled_output.size())
            diag_attention = torch.diag(self.attention)
#            print(f"diag attention={diag_attention.size()}")
            # [batch*bag_size, hidden*2] [hidden*2, hidden*2]
            xA = torch.matmul(bag_trans_pooled_output.view(-1, self.config.hidden_size * 2), diag_attention)
            # [batch, bag_size, hidden_size*2]
            #xA = xA.view(self.config.per_gpu_train_batch_size, 3, self.config.hidden_size)
#            print(f"xA:{xA.size()}")
            xA = xA.view(-1, 3, self.config.hidden_size*2) #[batch, bag_size, hidden*2]            

#            print(f"xA:{xA.size()}")
            # build gold-label's r for this batch= [batch*bag_size, hidden*2]'s transpose
            #label is based on bag, first change to based on sentence, each bag has 3 sentences, this 3 sens share a same bag label
            #e.g.: [1,2] ->[[1],[2]] -> [[1,1,1], [2,2,2]] -> [1,1,1,2,2,2]
            #Error ! labels = torch.flatten(labels.unsqueeze(-1).expand(-1, 3))   #[batch] -> [batch, 1] -> [batch, 3] -> [batch*3]                
            r = self.classifier.weight[labels, :].unsqueeze(-1)   # [hidden, num_labels] -> [batch, hidden, 1] -> [batch, hidden]
#            print(f"r: {r.size()}")
            xAr = torch.matmul(xA, r)
            # xAr=[batch_size, 3, 1]
#            print(f"xAr: {xAr.size()}")
            xAr = xAr.squeeze()
            # Then calculate alpha by softmax
            alpha = torch.softmax(xAr, -1)  # [batch,3]
#            print(alpha)
#            print(alpha.size())
            # Finally, weighted sum of hidden states
            # [batch, 1, 3] mm [batch, 3, hidden] = [batch, hidden]
            alpha = alpha.unsqueeze(1)
#            print(f"alpha {alpha.size()}")
#            print(f"bag_trans_pooled_output: {bag_trans_pooled_output.size()}")
            #[batch, 1, hidden] - > [batch, hidden]
            att_trans_pooled_output = torch.matmul(alpha, bag_trans_pooled_output).squeeze()
#            print(f"att bag output: {att_trans_pooled_output.size()}")
            # add similarity
            trans_similarity = self.cos(human_pooled_output, att_trans_pooled_output)            
            trans_similarity = 1 / torch.exp(trans_similarity - 1)
            trans_similarity_loss = torch.mean(trans_similarity)
            
            bag_trans_logits = self.classifier(att_trans_pooled_output)

        outputs = (human_logits,) #  + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(human_logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                human_loss = loss_fct(human_logits.view(-1, self.num_labels), labels.view(-1))
                if mode == "train":                    
                    bag_trans_loss = loss_fct(bag_trans_logits.view(-1, self.num_labels), labels.view(-1))                                                            
                    loss = trans_similarity_loss * (human_loss + loss_weight * bag_trans_loss)                    
                else:
                    loss = human_loss        
            outputs = (loss,) + outputs
        
        # Check the final alpha value        
        return outputs  # (loss), logits



@add_start_docstrings("""Bert Model transformer with a relation classification with share BERT 
for incorporating with translated data
head on top (a linear layer on top of the pooled output) e.g. for SemEval tasks. """,
    BERT_START_DOCSTRING, BERT_INPUTS_DOCSTRING)
class BertForRelationClassificationWithShareBertShareClassifierSoftminAttTrans(BertPreTrainedModel):
    r"""
    trans_loss = cross_entropy( att{google_repre, baidu_repre, xiaoniu_repre} )
    output = loss1 + alpha * (trans_loss)}
    alpha is reversed, use softmin
    """
    def __init__(self, config):
        super(BertForRelationClassificationWithShareBertShareClassifierSoftminAttTrans, self).__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        # Need to modify the parameters's name in pretrained model, 
        # We add a new classmethod from_pretrained_for_independent_bert in modeling_utils         
        self.bert = BertModel(config)        
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size*2, self.config.num_labels)
        self.attention = nn.Parameter(torch.ones(config.hidden_size*2)) # will been changed into a diag during training
        #self.alpha_1 = nn.Parameter(torch.Tensor([1.0]))
        #self.alpha_2 = nn.Parameter(torch.Tensor([1.0]))
        #self.alpha_3 = nn.Parameter(torch.Tensor([1.0]))
        #self.alpha_4 = nn.Parameter(torch.Tensor([1.0]))
        self.init_weights()

    def forward(self, mt_system, mode, loss_weight, input_ids, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, labels=None, entity_position=None, select_mode=None):
        """
        select_mode=max/ave
        mt_system=google, baidu, xiaoniu, all
        mode = train/test
        combine_mode = "plus", "concate"
        Input is [batch_size, 4, *], 4 for human, google, baidu and xiaoniu, we need to select
        TODO
        """        

        _, human_e1e2_outputs = self.bert(input_ids.select(1, 0),
                                        attention_mask=attention_mask.select(1, 0),
                                        token_type_ids=token_type_ids.select(1, 0),
                                        position_ids=position_ids, 
                                        head_mask=head_mask,
                                        entity_position=entity_position.select(1, 0))
        # pooled_output=[batch_size, hidden_size*2]
        human_pooled_output = human_e1e2_outputs
        human_pooled_output = self.dropout(human_pooled_output)                  
        human_logits = self.classifier(human_pooled_output)              
        if mode == "train":
            # For training
            _, google_e1e2_outputs = self.bert(input_ids.select(1, 1),
                                attention_mask=attention_mask.select(1, 1),
                                token_type_ids=token_type_ids.select(1, 1),
                                position_ids=position_ids, 
                                head_mask=head_mask,
                                entity_position=entity_position.select(1, 1))            
            google_pooled_output = google_e1e2_outputs
            google_pooled_output = self.dropout(google_pooled_output)
            
            _, baidu_e1e2_outputs = self.bert(input_ids.select(1, 2),
                                            attention_mask=attention_mask.select(1, 2),
                                            token_type_ids=token_type_ids.select(1, 2),
                                            position_ids=position_ids, 
                                            head_mask=head_mask,
                                            entity_position=entity_position.select(1, 2))            
            baidu_pooled_output = baidu_e1e2_outputs
            baidu_pooled_output = self.dropout(baidu_pooled_output)  
            
            _, xiaoniu_e1e2_outputs = self.bert(input_ids.select(1, 3),
                                            attention_mask=attention_mask.select(1, 3),
                                            token_type_ids=token_type_ids.select(1, 3),
                                            position_ids=position_ids, 
                                            head_mask=head_mask,
                                            entity_position=entity_position.select(1, 3))            
            xiaoniu_pooled_output = xiaoniu_e1e2_outputs
            xiaoniu_pooled_output = self.dropout(xiaoniu_pooled_output)  


            # This should be [batch_size, 3, hidden_size*2]
            # [batch, hidden] -> [batch, 1, hidden] after cat -> [batch, 3, hidden]
            bag_trans_pooled_output = torch.cat([google_pooled_output.unsqueeze(1), baidu_pooled_output.unsqueeze(1), xiaoniu_pooled_output.unsqueeze(1)], 1)
#            print("bag_trans_pooled_output")
#            print(bag_trans_pooled_output.size())
            diag_attention = torch.diag(self.attention)
#            print(f"diag attention={diag_attention.size()}")
            # [batch*bag_size, hidden*2] [hidden*2, hidden*2]
            xA = torch.matmul(bag_trans_pooled_output.view(-1, self.config.hidden_size * 2), diag_attention)
            # [batch, bag_size, hidden_size*2]
            #xA = xA.view(self.config.per_gpu_train_batch_size, 3, self.config.hidden_size)
#            print(f"xA:{xA.size()}")
            xA = xA.view(-1, 3, self.config.hidden_size*2) #[batch, bag_size, hidden*2]
#            print(f"xA:{xA.size()}")
            # build gold-label's r for this batch= [batch*bag_size, hidden*2]'s transpose
            #label is based on bag, first change to based on sentence, each bag has 3 sentences, this 3 sens share a same bag label
            #e.g.: [1,2] ->[[1],[2]] -> [[1,1,1], [2,2,2]] -> [1,1,1,2,2,2]
            #Error ! labels = torch.flatten(labels.unsqueeze(-1).expand(-1, 3))   #[batch] -> [batch, 1] -> [batch, 3] -> [batch*3]                
            r = self.classifier.weight[labels, :].unsqueeze(-1)   # [batch, hidden*2, 1]            
#            print(f"r: {r.size()}")            
            xAr = torch.matmul(xA, r)
            # xAr=[batch_size, 3, 1]
#            print(f"xAr: {xAr.size()}")
            xAr = xAr.squeeze()
            # Then calculate alpha by softmax
            #alpha = torch.softmax(xAr, -1)  # [batch,3]
            alpha = F.softmin(xAr, -1)  # [batch,3]
#            print(alpha)
#            print(alpha.size())
            # Finally, weighted sum of hidden states
            # [batch, 1, 3] mm [batch, 3, hidden] = [batch, hidden]
            alpha = alpha.unsqueeze(1)
#            print(f"alpha {alpha.size()}")
#            print(f"bag_trans_pooled_output: {bag_trans_pooled_output.size()}")
            #[batch, 1, hidden] - > [batch, hidden]
            att_trans_pooled_output = torch.matmul(alpha, bag_trans_pooled_output).squeeze()
#            print(f"att bag output: {att_trans_pooled_output.size()}")
             
            bag_trans_logits = self.classifier(att_trans_pooled_output)

        outputs = (human_logits,) #  + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(human_logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                human_loss = loss_fct(human_logits.view(-1, self.num_labels), labels.view(-1))
                if mode == "train":                    
                    bag_trans_loss = loss_fct(bag_trans_logits.view(-1, self.num_labels), labels.view(-1))
                    loss = human_loss + loss_weight * (bag_trans_loss)
                else:
                    loss = human_loss        
            outputs = (loss,) + outputs
        
        # Check the final alpha value        
        return outputs  # (loss), logits



@add_start_docstrings("""Bert Model transformer with a relation classification with share BERT 
for incorporating with translated data
head on top (a linear layer on top of the pooled output) e.g. for SemEval tasks. """,
    BERT_START_DOCSTRING, BERT_INPUTS_DOCSTRING)
class BertForRelationClassificationWithShareBertShareClassifierBak(BertPreTrainedModel):
    r"""
    output = loss1 + alpha * (loss2 + loss3 + loss4)
    """
    def __init__(self, config):
        super(BertForRelationClassificationWithShareBertShareClassifierBak, self).__init__(config)
        self.num_labels = config.num_labels
        # Need to modify the parameters's name in pretrained model, 
        # We add a new classmethod from_pretrained_for_independent_bert in modeling_utils         
        self.bert = BertModel(config)        
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size*2, self.config.num_labels)
        
        #self.alpha_1 = nn.Parameter(torch.Tensor([1.0]))
        #self.alpha_2 = nn.Parameter(torch.Tensor([1.0]))
        #self.alpha_3 = nn.Parameter(torch.Tensor([1.0]))
        #self.alpha_4 = nn.Parameter(torch.Tensor([1.0]))
        self.init_weights()

    def forward(self, mt_system, mode, loss_weight, input_ids, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, labels=None, entity_position=None):
        """
        mt_system=google, baidu, xiaoniu, all
        mode = train/test
        combine_mode = "plus", "concate"
        Input is [batch_size, 4, *], 4 for human, google, baidu and xiaoniu, we need to select
        """        

        _, human_e1e2_outputs = self.bert(input_ids.select(1, 0),
                                        attention_mask=attention_mask.select(1, 0),
                                        token_type_ids=token_type_ids.select(1, 0),
                                        position_ids=position_ids, 
                                        head_mask=head_mask,
                                        entity_position=entity_position.select(1, 0))
        # pooled_output=[batch_size, hidden_size*2]
        human_pooled_output = human_e1e2_outputs
        human_pooled_output = self.dropout(human_pooled_output)                  
        human_logits = self.classifier(human_pooled_output)              
        if mode == "train":
            # For training
            _, google_e1e2_outputs = self.bert(input_ids.select(1, 1),
                                attention_mask=attention_mask.select(1, 1),
                                token_type_ids=token_type_ids.select(1, 1),
                                position_ids=position_ids, 
                                head_mask=head_mask,
                                entity_position=entity_position.select(1, 1))            
            google_pooled_output = google_e1e2_outputs
            google_pooled_output = self.dropout(google_pooled_output)
            
            _, baidu_e1e2_outputs = self.bert(input_ids.select(1, 2),
                                            attention_mask=attention_mask.select(1, 2),
                                            token_type_ids=token_type_ids.select(1, 2),
                                            position_ids=position_ids, 
                                            head_mask=head_mask,
                                            entity_position=entity_position.select(1, 2))            
            baidu_pooled_output = baidu_e1e2_outputs
            baidu_pooled_output = self.dropout(baidu_pooled_output)  
            
            _, xiaoniu_e1e2_outputs = self.bert(input_ids.select(1, 3),
                                            attention_mask=attention_mask.select(1, 3),
                                            token_type_ids=token_type_ids.select(1, 3),
                                            position_ids=position_ids, 
                                            head_mask=head_mask,
                                            entity_position=entity_position.select(1, 3))            
            xiaoniu_pooled_output = xiaoniu_e1e2_outputs
            xiaoniu_pooled_output = self.dropout(xiaoniu_pooled_output)             
                    
            
            google_logits = self.classifier(google_pooled_output)
            baidu_logits = self.classifier(baidu_pooled_output)
            xiaoniu_logits = self.classifier(xiaoniu_pooled_output)

        outputs = (human_logits,) #  + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(human_logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                human_loss = loss_fct(human_logits.view(-1, self.num_labels), labels.view(-1))
                if mode == "train":                    
                    google_loss = loss_fct(google_logits.view(-1, self.num_labels), labels.view(-1))
                    baidu_loss = loss_fct(baidu_logits.view(-1, self.num_labels), labels.view(-1))
                    xiaoniu_loss = loss_fct(xiaoniu_logits.view(-1, self.num_labels), labels.view(-1))
                    loss = human_loss + loss_weight * (google_loss + baidu_loss + xiaoniu_loss)
                else:
                    loss = human_loss        
            outputs = (loss,) + outputs
        
        # Check the final alpha value        
        return outputs  # (loss), logits


@add_start_docstrings("""Bert Model transformer with a relation classification with share BERT 
for incorporating with translated data
head on top (a linear layer on top of the pooled output) e.g. for SemEval tasks. """,
    BERT_START_DOCSTRING, BERT_INPUTS_DOCSTRING)
class BertForRelationClassificationWithShareBertOwnClassifier(BertPreTrainedModel):
    r"""
    output = loss1 + loss2 + loss3 + loss4
    """
    def __init__(self, config):
        super(BertForRelationClassificationWithShareBertOwnClassifier, self).__init__(config)
        self.num_labels = config.num_labels
        # Need to modify the parameters's name in pretrained model, 
        # We add a new classmethod from_pretrained_for_independent_bert in modeling_utils         
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.human_classifier = nn.Linear(config.hidden_size*2, self.config.num_labels)
        self.google_classifier = nn.Linear(config.hidden_size*2, self.config.num_labels)
        self.baidu_classifier = nn.Linear(config.hidden_size*2, self.config.num_labels)
        self.xiaoniu_classifier = nn.Linear(config.hidden_size*2, self.config.num_labels)
        
        #self.alpha_1 = nn.Parameter(torch.Tensor([1.0]))
        #self.alpha_2 = nn.Parameter(torch.Tensor([1.0]))
        #self.alpha_3 = nn.Parameter(torch.Tensor([1.0]))
        #self.alpha_4 = nn.Parameter(torch.Tensor([1.0]))
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, labels=None, entity_position=None):
        """
        mt_system=google, baidu, xiaoniu, all
        combine_mode = "plus", "concate"
        Input is [batch_size, 4, *], 4 for human, google, baidu and xiaoniu, we need to select
        """        

        human_outputs = self.bert(input_ids.select(1, 0),
                                        attention_mask=attention_mask.select(1, 0),
                                        token_type_ids=token_type_ids.select(1, 0),
                                        position_ids=position_ids, 
                                        head_mask=head_mask,
                                        entity_position=entity_position.select(1, 0))
        # pooled_output=[batch_size, hidden_size*2]
        human_pooled_output = human_outputs[1]
        human_pooled_output = self.dropout(human_pooled_output)
        human_logits = self.human_classifier(human_pooled_output)
    
        google_outputs = self.bert(input_ids.select(1, 1),
                            attention_mask=attention_mask.select(1, 1),
                            token_type_ids=token_type_ids.select(1, 1),
                            position_ids=position_ids, 
                            head_mask=head_mask,
                            entity_position=entity_position.select(1, 1))            
        google_pooled_output = google_outputs[1]
        google_pooled_output = self.dropout(google_pooled_output)
        google_logits = self.google_classifier(google_pooled_output)
    
        baidu_outputs = self.bert(input_ids.select(1, 2),
                                        attention_mask=attention_mask.select(1, 2),
                                        token_type_ids=token_type_ids.select(1, 2),
                                        position_ids=position_ids, 
                                        head_mask=head_mask,
                                        entity_position=entity_position.select(1, 2))            
        baidu_pooled_output = baidu_outputs[1]
        baidu_pooled_output = self.dropout(baidu_pooled_output)
        baidu_logits = self.baidu_classifier(baidu_pooled_output)
    
        xiaoniu_outputs = self.bert(input_ids.select(1, 3),
                                        attention_mask=attention_mask.select(1, 3),
                                        token_type_ids=token_type_ids.select(1, 3),
                                        position_ids=position_ids, 
                                        head_mask=head_mask,
                                        entity_position=entity_position.select(1, 3))            
        xiaoniu_pooled_output = xiaoniu_outputs[1]
        xiaoniu_pooled_output = self.dropout(xiaoniu_pooled_output)
        xiaoniu_logits = self.xiaoniu_classifier(xiaoniu_pooled_output)

        pool_logits = F.softmax(human_logits) + F.softmax(google_logits) + F.softmax(baidu_logits) + F.softmax(xiaoniu_logits)
        
        #outputs = (human_logits, google_logits, baidu_logits, xiaoniu_logits, ) #  + outputs[2:]  # add hidden states and attention if they are here
        outputs = (pool_logits, )
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                human_loss = loss_fct(human_logits.view(-1, self.num_labels), labels.view(-1))
                google_loss = loss_fct(google_logits.view(-1, self.num_labels), labels.view(-1))
                baidu_loss = loss_fct(baidu_logits.view(-1, self.num_labels), labels.view(-1))
                xiaoniu_loss = loss_fct(xiaoniu_logits.view(-1, self.num_labels), labels.view(-1))
                loss = human_loss + google_loss + baidu_loss + xiaoniu_loss
            outputs = (loss,) + outputs                    

        return outputs  # (loss), logits, (hidden_states), (attentions)


@add_start_docstrings("""Bert Model transformer with a relation classification with share BERT 
for incorporating with translated data
head on top (a linear layer on top of the pooled output) e.g. for SemEval tasks. """,
    BERT_START_DOCSTRING, BERT_INPUTS_DOCSTRING)
class BertForRelationClassificationWithOwnBertOwnClassifier(BertPreTrainedModel):
    r"""
    output = loss1 + loss2 + loss3 + loss4
    """
    def __init__(self, config):
        super(BertForRelationClassificationWithOwnBertOwnClassifier, self).__init__(config)
        self.num_labels = config.num_labels
        # Need to modify the parameters's name in pretrained model, 
        # We add a new classmethod from_pretrained_for_independent_bert in modeling_utils                 
        self.human_bert = BertModel(config)
        self.google_bert = BertModel(config)
        self.baidu_bert = BertModel(config)
        self.xiaoniu_bert = BertModel(config)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.human_classifier = nn.Linear(config.hidden_size*2, self.config.num_labels)
        self.google_classifier = nn.Linear(config.hidden_size*2, self.config.num_labels)
        self.baidu_classifier = nn.Linear(config.hidden_size*2, self.config.num_labels)
        self.xiaoniu_classifier = nn.Linear(config.hidden_size*2, self.config.num_labels)
        
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, labels=None, entity_position=None):
        """
        mt_system=google, baidu, xiaoniu, all
        combine_mode = "plus", "concate"
        Input is [batch_size, 4, *], 4 for human, google, baidu and xiaoniu, we need to select
        """        

        human_outputs = self.human_bert(input_ids.select(1, 0),
                                        attention_mask=attention_mask.select(1, 0),
                                        token_type_ids=token_type_ids.select(1, 0),
                                        position_ids=position_ids, 
                                        head_mask=head_mask,
                                        entity_position=entity_position.select(1, 0))
        # pooled_output=[batch_size, hidden_size*2]
        human_pooled_output = human_outputs[1]
        human_pooled_output = self.dropout(human_pooled_output)
        human_logits = self.human_classifier(human_pooled_output)
    
        google_outputs = self.google_bert(input_ids.select(1, 1),
                            attention_mask=attention_mask.select(1, 1),
                            token_type_ids=token_type_ids.select(1, 1),
                            position_ids=position_ids, 
                            head_mask=head_mask,
                            entity_position=entity_position.select(1, 1))            
        google_pooled_output = google_outputs[1]
        google_pooled_output = self.dropout(google_pooled_output)
        google_logits = self.google_classifier(google_pooled_output)
    
        baidu_outputs = self.baidu_bert(input_ids.select(1, 2),
                                        attention_mask=attention_mask.select(1, 2),
                                        token_type_ids=token_type_ids.select(1, 2),
                                        position_ids=position_ids, 
                                        head_mask=head_mask,
                                        entity_position=entity_position.select(1, 2))            
        baidu_pooled_output = baidu_outputs[1]
        baidu_pooled_output = self.dropout(baidu_pooled_output)
        baidu_logits = self.baidu_classifier(baidu_pooled_output)
    
        xiaoniu_outputs = self.xiaoniu_bert(input_ids.select(1, 3),
                                        attention_mask=attention_mask.select(1, 3),
                                        token_type_ids=token_type_ids.select(1, 3),
                                        position_ids=position_ids, 
                                        head_mask=head_mask,
                                        entity_position=entity_position.select(1, 3))            
        xiaoniu_pooled_output = xiaoniu_outputs[1]
        xiaoniu_pooled_output = self.dropout(xiaoniu_pooled_output)
        xiaoniu_logits = self.xiaoniu_classifier(xiaoniu_pooled_output)

        
        outputs = (human_logits, google_logits, baidu_logits, xiaoniu_logits, ) #  + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:        
            loss_fct = CrossEntropyLoss()
            human_loss = loss_fct(human_logits.view(-1, self.num_labels), labels.view(-1))
            google_loss = loss_fct(google_logits.view(-1, self.num_labels), labels.view(-1))
            baidu_loss = loss_fct(baidu_logits.view(-1, self.num_labels), labels.view(-1))
            xiaoniu_loss = loss_fct(xiaoniu_logits.view(-1, self.num_labels), labels.view(-1))
            loss = human_loss + google_loss + baidu_loss + xiaoniu_loss

            outputs = (loss,) + outputs                    

        return outputs  # (loss), logits, (hidden_states), (attentions)



@add_start_docstrings("""Bert Model transformer with a relation classification head on top (a linear layer on top of
    the pooled output) e.g. for SemEval tasks. """,
    BERT_START_DOCSTRING, BERT_INPUTS_DOCSTRING)
class BertForRelationClassification(BertPreTrainedModel):
    r"""
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for computing the sequence classification/regression loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.
            If ``config.num_labels == 1`` a regression loss is computed (Mean-Square loss),
            If ``config.num_labels > 1`` a classification loss is computed (Cross-Entropy).

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Classification (or regression if config.num_labels==1) loss.
        **logits**: ``torch.FloatTensor`` of shape ``(batch_size, config.num_labels)``
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForRelationClassification.from_pretrained('bert-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, logits = outputs[:2]

    """
    def __init__(self, config):
        super(BertForRelationClassification, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size*2, self.config.num_labels)

        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, labels=None, entity_position=None,
                cls_outputs=None):
        
        cls_outputs, e1_e2_outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids, 
                            head_mask=head_mask,
                            entity_position=entity_position)
        # pooled_output=[batch_size, hidden_size*2]        
        
        # pooled_output = torch.cat([cls_outputs, e1_e2_outputs], -1)
        pooled_output = e1_e2_outputs
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        outputs = (logits, pooled_output)

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)



@add_start_docstrings("""Bert Model transformer with a relation classification head on top (a linear layer on top of
    the pooled output) e.g. for SemEval tasks. """,
    BERT_START_DOCSTRING, BERT_INPUTS_DOCSTRING)
class BertForSenPairRelationClassification(BertPreTrainedModel):
    r"""
        Input = [CLS] sen with entity wrapped with tag [SEP] e1 [E] e2 [SEP]
        Output = [CLS]

    """
    def __init__(self, config):
        super(BertForSenPairRelationClassification, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)

        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, labels=None, entity_position=None):

        pooled_output, _ = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids, 
                            head_mask=head_mask,
                            entity_position=None)
        #[CLS] = pooled_output=[batch_size, hidden_size]        

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        outputs = (logits,)

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits



@add_start_docstrings("""Bert Model transformer with a relation classification head on top (a linear layer on top of
    the pooled output) e.g. for SemEval tasks. """,
    BERT_START_DOCSTRING, BERT_INPUTS_DOCSTRING)
class BertForSenPairRelationClassificationOutE1E2(BertPreTrainedModel):
    r"""
        Input = [CLS] ... [E1] ... [E2] .. [SEP] mask entity copy [SEP]
        Output = [E1] cat [E2]

    """
    def __init__(self, config):
        super(BertForSenPairRelationClassificationOutE1E2, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size * 2, self.config.num_labels)

        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, labels=None, entity_position=None):

        cls_pooled_output, e1e2_pooled_output = self.bert(input_ids,
                                                attention_mask=attention_mask,
                                                token_type_ids=token_type_ids,
                                                position_ids=position_ids, 
                                                head_mask=head_mask,
                                                entity_position=entity_position)
        #[CLS] = pooled_output=[batch_size, hidden_size]        
        pooled_output = e1e2_pooled_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        outputs = (logits,)

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits


@add_start_docstrings("""Bert Model transformer with a relation classification head on top (a linear layer on top of
    the pooled output) e.g. for SemEval tasks. """,
    BERT_START_DOCSTRING, BERT_INPUTS_DOCSTRING)
class BertForSenPairRelationClassificationOutClsE1E2(BertPreTrainedModel):
    r"""
        Input = [CLS] ... [E1] ... [E2] .. [SEP] mask entity copy [SEP]
        Output = [CLS] cat [E1] cat [E2]

    """
    def __init__(self, config):
        super(BertForSenPairRelationClassificationOutClsE1E2, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size * 3, self.config.num_labels)

        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, labels=None, entity_position=None):

        cls_pooled_output, e1e2_pooled_output = self.bert(input_ids,
                                                attention_mask=attention_mask,
                                                token_type_ids=token_type_ids,
                                                position_ids=position_ids, 
                                                head_mask=head_mask,
                                                entity_position=entity_position)
        #[CLS] = pooled_output=[batch_size, hidden_size]        
        pooled_output = torch.cat([cls_pooled_output, e1e2_pooled_output], -1)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        outputs = (logits,)

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits


@add_start_docstrings("""Bert Model transformer with a relation classification head on top (a linear layer on top of
    the pooled output) e.g. for SemEval tasks. """,
    BERT_START_DOCSTRING, BERT_INPUTS_DOCSTRING)
class BertForSenPairWithCosForTransRelationClassificationOutClsE1E2(BertPreTrainedModel):
    r"""
        Input = [CLS] ... [E1] ... [E2] .. [SEP] mask entity copy [SEP]
        Output = [CLS] cat [E1] cat [E2]
        Add similarity model

    """
    def __init__(self, config):
        super(BertForSenPairWithCosForTransRelationClassificationOutClsE1E2, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size * 3, self.config.num_labels)
        self.cos = torch.nn.CosineSimilarity()

        self.init_weights()

    def forward(self, mode, input_ids, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, labels=None, entity_position=None):
        """
        mode = "train" use similarity
        mode="test" not use similarity
        """

        human_cls_pooled_output, human_e1e2_pooled_output = self.bert(input_ids.select(1, 0),
                                                attention_mask=attention_mask.select(1, 0),
                                                token_type_ids=token_type_ids.select(1, 0),
                                                position_ids=position_ids, 
                                                head_mask=head_mask,
                                                entity_position=entity_position.select(1, 0))
        #[CLS] = pooled_output=[batch_size, hidden_size]        
        human_pooled_output = torch.cat([human_cls_pooled_output, human_e1e2_pooled_output], -1)
        human_pooled_output = self.dropout(human_pooled_output)
        logits = self.classifier(human_pooled_output)
        outputs = (logits,)
        loss_similarity = 1.0
        if mode == "train":
            google_cls_pooled_output, google_e1e2_pooled_output = self.bert(input_ids.select(1, 1),
                                                    attention_mask=attention_mask.select(1, 1),
                                                    token_type_ids=token_type_ids.select(1, 1),
                                                    position_ids=position_ids, 
                                                    head_mask=head_mask,
                                                    entity_position=entity_position.select(1, 1))
            google_pooled_output = torch.cat([google_cls_pooled_output, google_e1e2_pooled_output], -1)            
            google_pooled_output = self.dropout(google_pooled_output)
            """
            baidu_cls_pooled_output, baidu_e1e2_pooled_output = self.bert(input_ids.select(1, 2),
                                                    attention_mask=attention_mask.select(1, 2),
                                                    token_type_ids=token_type_ids.select(1, 2),
                                                    position_ids=position_ids, 
                                                    head_mask=head_mask,
                                                    entity_position=entity_position.select(1, 2))
            baidu_pooled_output = torch.cat([baidu_cls_pooled_output, baidu_e1e2_pooled_output], -1)
            baidu_pooled_output = self.dropout(baidu_pooled_output)
            xiaoniu_cls_pooled_output, xiaoniu_e1e2_pooled_output = self.bert(input_ids.select(1, 3),
                                                    attention_mask=attention_mask.select(1, 3),
                                                    token_type_ids=token_type_ids.select(1, 3),
                                                    position_ids=position_ids, 
                                                    head_mask=head_mask,
                                                    entity_position=entity_position.select(1, 3))                                                
            xiaoniu_pooled_output = torch.cat([xiaoniu_cls_pooled_output, xiaoniu_e1e2_pooled_output], -1)
            xiaoniu_pooled_output = self.dropout(xiaoniu_pooled_output)
            """
            google_similarity = self.cos(human_pooled_output, google_pooled_output)
            #baidu_similarity = self.cos(baidu_pooled_output, baidu_pooled_output)
            #xiaoniu_similarity = self.cos(xiaoniu_pooled_output, xiaoniu_pooled_output)
            google_similarity = 1 / torch.exp(google_similarity - 1)
            #baidu_similarity = 1 / torch.exp(baidu_similarity - 1)
            #xiaoniu_similarity = 1 / torch.exp(xiaoniu_similarity - 1)
            product_similarity = google_similarity# * baidu_similarity * xiaoniu_similarity
            loss_similarity = torch.mean(product_similarity)

            outputs = (logits, loss_similarity,)

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            
            loss = loss * loss_similarity
            outputs = (loss,) + outputs

        return outputs  # (loss), logits



@add_start_docstrings("""Bert Model transformer with a relation classification head on top (a linear layer on top of
    the pooled output) e.g. for SemEval tasks. """,
    BERT_START_DOCSTRING, BERT_INPUTS_DOCSTRING)
class BertForSenPairBinaryRelationClassification(BertPreTrainedModel):
    r"""
        Input = [CLS] sen with entity wrapped with tag [SEP] e1 rel_N e2 [SEP]
        Output = [CLS]
        0/1

    """
    def __init__(self, config):
        super(BertForSenPairBinaryRelationClassification, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 2)#self.config.num_labels)
        self.softmax = nn.Softmax(dim=-1)

        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, binary_labels=None, labels=None, entity_position=None):        
        input_ids = input_ids.squeeze()
        attention_mask = attention_mask.squeeze()
        token_type_ids = token_type_ids.squeeze()        
        pooled_output, _ = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids, 
                            head_mask=head_mask,
                            entity_position=None)
        #[CLS] = pooled_output=[batch_size, hidden_size]                
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        outputs = (logits,)

        if binary_labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), binary_labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), binary_labels.view(-1))
            outputs = (loss,) + outputs

        # output relation label by these binary outputs
        if labels is not None:
            a = self.softmax(logits)                        
            max_prob, max_index = torch.max(a[:, 1], 0)
            # 返回所有标签为1的porb，组成一个伪logits            
            outputs = outputs + (torch.unsqueeze(a[:, 1], 0),)

        return outputs  # (loss), logits



@add_start_docstrings("""Bert Model transformer with a relation classification with independent BERT 
for incorporating with translated data
head on top (a linear layer on top of the pooled output) e.g. for SemEval tasks. """,
    BERT_START_DOCSTRING, BERT_INPUTS_DOCSTRING)
class BertForRelationClassificationWithIndependentBert(BertPreTrainedModel):
    r"""
    ABC
    """
    def __init__(self, config):
        super(BertForRelationClassificationWithIndependentBert, self).__init__(config)
        self.num_labels = config.num_labels
        # Need to modify the parameters's name in pretrained model, 
        # We add a new classmethod from_pretrained_for_independent_bert in modeling_utils         
        self.human_bert = BertModel(config)
        self.google_bert = BertModel(config)
        self.baidu_bert = BertModel(config)
        self.xiaoniu_bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size*2, self.config.num_labels)

        self.init_weights()

    def forward(self, mt_system, combine_mode, input_ids, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, labels=None, entity_position=None):
        """
        mt_system=google, baidu, xiaoniu, all
        combine_mode = "plus", "concate"
        Input is [batch_size, 4, *], 4 for human, google, baidu and xiaoniu, we need to select
        """        

        human_outputs = self.human_bert(input_ids.select(1, 0),
                                        attention_mask=attention_mask.select(1, 0),
                                        token_type_ids=token_type_ids.select(1, 0),
                                        position_ids=position_ids, 
                                        head_mask=head_mask,
                                        entity_position=entity_position.select(1, 0))
        # pooled_output=[batch_size, hidden_size*2]
        human_pooled_output = human_outputs[1]

        human_pooled_output = self.dropout(human_pooled_output)

        pooled_output = None
        if mt_system == "google":
            google_outputs = self.google_bert(input_ids.select(1, 1),
                                            attention_mask=attention_mask.select(1, 1),
                                            token_type_ids=token_type_ids.select(1, 1),
                                            position_ids=position_ids, 
                                            head_mask=head_mask,
                                            entity_position=entity_position.select(1, 1))
            # pooled_output=[batch_size, hidden_size*2]
            google_pooled_output = google_outputs[1]
            google_pooled_output = self.dropout(google_pooled_output)
            if combine_mode == "plus":
                pooled_output = human_pooled_output + google_pooled_output
            elif combine_mode == "concate":
                pooled_output = torch.cat([human_pooled_output, google_pooled_output], -1)
                print(pooled_output.size())
                exit()
            else:
                raise ValueError(f"Error combine_mode {combine_mode}")
        elif mt_system == "baidu":
            baidu_outputs = self.baidu_bert(input_ids.select(1, 2),
                                            attention_mask=attention_mask.select(1, 2),
                                            token_type_ids=token_type_ids.select(1, 2),
                                            position_ids=position_ids, 
                                            head_mask=head_mask,
                                            entity_position=entity_position.select(1, 2))
            # pooled_output=[batch_size, hidden_size*2]
            baidu_pooled_output = baidu_outputs[1]
            baidu_pooled_output = self.dropout(baidu_pooled_output)  
            if combine_mode == "plus":
                pooled_output = human_pooled_output + baidu_pooled_output
            elif combine_mode == "concate":
                pooled_output = torch.cat([human_pooled_output, baidu_pooled_output], -1)                
            else:
                raise ValueError(f"Error combine_mode {combine_mode}")            
        elif mt_system == "xiaoniu":          
            xiaoniu_outputs = self.xiaoniu_bert(input_ids.select(1, 3),
                                            attention_mask=attention_mask.select(1, 3),
                                            token_type_ids=token_type_ids.select(1, 3),
                                            position_ids=position_ids, 
                                            head_mask=head_mask,
                                            entity_position=entity_position.select(1, 3))
            # pooled_output=[batch_size, hidden_size*2]
            xiaoniu_pooled_output = xiaoniu_outputs[1]
            xiaoniu_pooled_output = self.dropout(xiaoniu_pooled_output)        
            if combine_mode == "plus":
                pooled_output = human_pooled_output + xiaoniu_pooled_output
            elif combine_mode == "concate":
                pooled_output = torch.cat([human_pooled_output, xiaoniu_pooled_output], -1)                
            else:
                raise ValueError(f"Error combine_mode {combine_mode}")                        
        elif mt_system == "all":
            google_outputs = self.google_bert(input_ids.select(1, 1),
                                attention_mask=attention_mask.select(1, 1),
                                token_type_ids=token_type_ids.select(1, 1),
                                position_ids=position_ids, 
                                head_mask=head_mask,
                                entity_position=entity_position.select(1, 1))            
            google_pooled_output = google_outputs[1]
            google_pooled_output = self.dropout(google_pooled_output)
        
            baidu_outputs = self.baidu_bert(input_ids.select(1, 2),
                                            attention_mask=attention_mask.select(1, 2),
                                            token_type_ids=token_type_ids.select(1, 2),
                                            position_ids=position_ids, 
                                            head_mask=head_mask,
                                            entity_position=entity_position.select(1, 2))            
            baidu_pooled_output = baidu_outputs[1]
            baidu_pooled_output = self.dropout(baidu_pooled_output)  
        
            xiaoniu_outputs = self.xiaoniu_bert(input_ids.select(1, 3),
                                            attention_mask=attention_mask.select(1, 3),
                                            token_type_ids=token_type_ids.select(1, 3),
                                            position_ids=position_ids, 
                                            head_mask=head_mask,
                                            entity_position=entity_position.select(1, 3))            
            xiaoniu_pooled_output = xiaoniu_outputs[1]
            xiaoniu_pooled_output = self.dropout(xiaoniu_pooled_output)     
            if combine_mode == "plus":
                pooled_output = human_pooled_output + google_pooled_output + baidu_pooled_output + xiaoniu_pooled_output
            elif combine_mode == "concate":
                pooled_output = torch.cat([human_pooled_output, google_pooled_output, baidu_pooled_output, xiaoniu_pooled_output], -1)                
            else:
                raise ValueError(f"Error combine_mode {combine_mode}")                        
        else:
            raise ValueError(f"Error mt_system {mt_system}")   
        
        assert pooled_output is not None
        logits = self.classifier(pooled_output)

        outputs = (logits,) #  + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)



@add_start_docstrings("""Bert Model transformer with a relation classification with independent BERT 
for incorporating with translated data
cos(h_human, h_trans)
head on top (a linear layer on top of the pooled output) e.g. for SemEval tasks. """,
    BERT_START_DOCSTRING, BERT_INPUTS_DOCSTRING)
class BertForRelationClassificationWithSimilarityTask(BertPreTrainedModel):
    r"""
    ABC
    """
    def __init__(self, config):
        super(BertForRelationClassificationWithSimilarityTask, self).__init__(config)
        self.num_labels = config.num_labels
        # Need to modify the parameters's name in pretrained model, 
        # We add a new classmethod from_pretrained_for_independent_bert in modeling_utils
        self.bert = BertModel(config)        
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size*2, self.config.num_labels)

        self.cos = torch.nn.CosineSimilarity()

        self.init_weights()

    def forward(self, mt_system, input_ids, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, labels=None, entity_position=None):
        """
        mt_system=google, baidu, xiaoniu, all
        combine_mode = "plus", "concate"
        Input is [batch_size, 4, *], 4 for human, google, baidu and xiaoniu, we need to select
        """        

        _, human_pooled_output = self.bert(input_ids.select(1, 0),
                                        attention_mask=attention_mask.select(1, 0),
                                        token_type_ids=token_type_ids.select(1, 0),
                                        position_ids=position_ids, 
                                        head_mask=head_mask,
                                        entity_position=entity_position.select(1, 0))
        # pooled_output=[batch_size, hidden_size*2]

        human_pooled_output = self.dropout(human_pooled_output)

        pooled_output = None
        if mt_system == "google":
            _,  google_pooled_output = self.bert(input_ids.select(1, 1),
                                            attention_mask=attention_mask.select(1, 1),
                                            token_type_ids=token_type_ids.select(1, 1),
                                            position_ids=position_ids, 
                                            head_mask=head_mask,
                                            entity_position=entity_position.select(1, 1))
            # pooled_output=[batch_size, hidden_size*2]
            google_pooled_output = self.dropout(google_pooled_output)
            # Add similarity task
            similarity = self.cos(human_pooled_output, google_pooled_output)
            # now loss_similarity=[batch_size]            
            loss_similarity = 1 / torch.exp(similarity - 1)
            loss_similarity = torch.mean(loss_similarity)                        
            
        elif mt_system == "baidu":
            _, baidu_pooled_output = self.bert(input_ids.select(1, 2),
                                            attention_mask=attention_mask.select(1, 2),
                                            token_type_ids=token_type_ids.select(1, 2),
                                            position_ids=position_ids, 
                                            head_mask=head_mask,
                                            entity_position=entity_position.select(1, 2))
            # pooled_output=[batch_size, hidden_size*2]
            baidu_pooled_output = self.dropout(baidu_pooled_output)
            # Add similarity task
            similarity = self.cos(human_pooled_output, baidu_pooled_output)
            # now loss_similarity=[batch_size]            
            loss_similarity = 1 / torch.exp(similarity - 1)
            loss_similarity = torch.mean(loss_similarity)
        elif mt_system == "xiaoniu":          
            _, xiaoniu_pooled_output = self.bert(input_ids.select(1, 3),
                                            attention_mask=attention_mask.select(1, 3),
                                            token_type_ids=token_type_ids.select(1, 3),
                                            position_ids=position_ids, 
                                            head_mask=head_mask,
                                            entity_position=entity_position.select(1, 3))
            # pooled_output=[batch_size, hidden_size*2]
            xiaoniu_pooled_output = self.dropout(xiaoniu_pooled_output)        
            # Add similarity task
            similarity = self.cos(human_pooled_output, xiaoniu_pooled_output)
            # now loss_similarity=[batch_size]
            loss_similarity = 1 / torch.exp(similarity - 1)
            loss_similarity = torch.mean(loss_similarity)
        elif mt_system == "all":
            _, google_pooled_output = self.bert(input_ids.select(1, 1),
                                attention_mask=attention_mask.select(1, 1),
                                token_type_ids=token_type_ids.select(1, 1),
                                position_ids=position_ids, 
                                head_mask=head_mask,
                                entity_position=entity_position.select(1, 1))
            google_pooled_output = self.dropout(google_pooled_output)
        
            _, baidu_pooled_output = self.bert(input_ids.select(1, 2),
                                            attention_mask=attention_mask.select(1, 2),
                                            token_type_ids=token_type_ids.select(1, 2),
                                            position_ids=position_ids, 
                                            head_mask=head_mask,
                                            entity_position=entity_position.select(1, 2))
            baidu_pooled_output = self.dropout(baidu_pooled_output)  
        
            _, xiaoniu_pooled_output = self.bert(input_ids.select(1, 3),
                                            attention_mask=attention_mask.select(1, 3),
                                            token_type_ids=token_type_ids.select(1, 3),
                                            position_ids=position_ids, 
                                            head_mask=head_mask,
                                            entity_position=entity_position.select(1, 3))            
            xiaoniu_pooled_output = self.dropout(xiaoniu_pooled_output)
            # Add similarity task
            similarity_google = self.cos(human_pooled_output, google_pooled_output)
            similarity_baidu = self.cos(human_pooled_output, baidu_pooled_output)
            similarity_xiaoniu = self.cos(human_pooled_output, xiaoniu_pooled_output)
            similarity_google = 1 / torch.exp(similarity_google - 1)
            similarity_baidu = 1 / torch.exp(similarity_baidu - 1)
            similarity_xiaoniu = 1 / torch.exp(similarity_xiaoniu - 1)
            product_similarity = similarity_google * similarity_baidu * similarity_xiaoniu
            #print(similarity_google)
            #print(similarity_google.size())
            #print(product_similarity)
            #print(product_similarity.size())
            # now loss_similarity=[batch_size]                        
            loss_similarity = torch.mean(product_similarity)            
            #exit()
            

        else:
            raise ValueError(f"Error mt_system {mt_system}")   
        
        
        logits = self.classifier(human_pooled_output)

        outputs = (loss_similarity, logits,) #  + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                        
            loss = loss * loss_similarity
            outputs = (loss,) + outputs

        return outputs  # (loss, loss_similarity, logits, (hidden_states), (attentions)




@add_start_docstrings("""Bert Model transformer with a relation classification with independent BERT 
for incorporating with translated data
cos(h_human, h_trans)
head on top (a linear layer on top of the pooled output) e.g. for SemEval tasks. """,
    BERT_START_DOCSTRING, BERT_INPUTS_DOCSTRING)
class BertForRelationClassificationWithSimilarityTaskAddCLS(BertPreTrainedModel):
    r"""
    human_cls cat human_e1 cat human_e2 with translated sens' similarity task
    """
    def __init__(self, config):
        super(BertForRelationClassificationWithSimilarityTaskAddCLS, self).__init__(config)
        self.num_labels = config.num_labels
        # Need to modify the parameters's name in pretrained model, 
        # We add a new classmethod from_pretrained_for_independent_bert in modeling_utils
        self.bert = BertModel(config)        
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size*3, self.config.num_labels)

        self.cos = torch.nn.CosineSimilarity()

        self.init_weights()

    def forward(self, mt_system, input_ids, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, labels=None, entity_position=None):
        """
        mt_system=google, baidu, xiaoniu, all
        combine_mode = "plus", "concate"
        Input is [batch_size, 4, *], 4 for human, google, baidu and xiaoniu, we need to select
        """        

        human_cls_outputs, human_entity_outputs = self.bert(input_ids.select(1, 0),
                                        attention_mask=attention_mask.select(1, 0),
                                        token_type_ids=token_type_ids.select(1, 0),
                                        position_ids=position_ids, 
                                        head_mask=head_mask,
                                        entity_position=entity_position.select(1, 0))
        # pooled_output=[batch_size, hidden_size*2]        
        
        human_pooled_output = torch.cat([human_cls_outputs, human_entity_outputs], -1)

        human_pooled_output = self.dropout(human_pooled_output)

        pooled_output = None
        if mt_system == "google":
            google_outputs = self.bert(input_ids.select(1, 1),
                                            attention_mask=attention_mask.select(1, 1),
                                            token_type_ids=token_type_ids.select(1, 1),
                                            position_ids=position_ids, 
                                            head_mask=head_mask,
                                            entity_position=entity_position.select(1, 1))
            # pooled_output=[batch_size, hidden_size*2]
            exit()
            google_pooled_output = self.dropout(google_pooled_output)
            # Add similarity task
            similarity = self.cos(human_pooled_output, google_pooled_output)
            # now loss_similarity=[batch_size]            
            loss_similarity = 1 / torch.exp(similarity - 1)
            loss_similarity = torch.mean(loss_similarity)                        
            
        elif mt_system == "baidu":
            baidu_outputs = self.bert(input_ids.select(1, 2),
                                            attention_mask=attention_mask.select(1, 2),
                                            token_type_ids=token_type_ids.select(1, 2),
                                            position_ids=position_ids, 
                                            head_mask=head_mask,
                                            entity_position=entity_position.select(1, 2))
            # pooled_output=[batch_size, hidden_size*2]
            baidu_pooled_output = baidu_outputs[1]
            baidu_pooled_output = self.dropout(baidu_pooled_output)
            # Add similarity task
            similarity = self.cos(human_pooled_output, baidu_pooled_output)
            # now loss_similarity=[batch_size]            
            loss_similarity = 1 / torch.exp(similarity - 1)
            loss_similarity = torch.mean(loss_similarity)
        elif mt_system == "xiaoniu":          
            xiaoniu_outputs = self.bert(input_ids.select(1, 3),
                                            attention_mask=attention_mask.select(1, 3),
                                            token_type_ids=token_type_ids.select(1, 3),
                                            position_ids=position_ids, 
                                            head_mask=head_mask,
                                            entity_position=entity_position.select(1, 3))
            # pooled_output=[batch_size, hidden_size*2]
            xiaoniu_pooled_output = xiaoniu_outputs[1]
            xiaoniu_pooled_output = self.dropout(xiaoniu_pooled_output)        
            # Add similarity task
            similarity = self.cos(human_pooled_output, xiaoniu_pooled_output)
            # now loss_similarity=[batch_size]
            loss_similarity = 1 / torch.exp(similarity - 1)
            loss_similarity = torch.mean(loss_similarity)
        elif mt_system == "all":
            google_cls_outputs, google_entity_outputs = self.bert(input_ids.select(1, 1),
                                attention_mask=attention_mask.select(1, 1),
                                token_type_ids=token_type_ids.select(1, 1),
                                position_ids=position_ids, 
                                head_mask=head_mask,
                                entity_position=entity_position.select(1, 1))                                    
            google_pooled_output = torch.cat([google_cls_outputs, google_entity_outputs], -1)
            google_pooled_output = self.dropout(google_pooled_output)
        
            baidu_cls_outputs, baidu_entity_outputs = self.bert(input_ids.select(1, 2),
                                            attention_mask=attention_mask.select(1, 2),
                                            token_type_ids=token_type_ids.select(1, 2),
                                            position_ids=position_ids, 
                                            head_mask=head_mask,
                                            entity_position=entity_position.select(1, 2))                        
            baidu_pooled_output = torch.cat([baidu_cls_outputs, baidu_entity_outputs], -1)
            baidu_pooled_output = self.dropout(baidu_pooled_output)  
        
            xiaoniu_cls_outputs, xiaoniu_entity_outputs = self.bert(input_ids.select(1, 3),
                                            attention_mask=attention_mask.select(1, 3),
                                            token_type_ids=token_type_ids.select(1, 3),
                                            position_ids=position_ids, 
                                            head_mask=head_mask,
                                            entity_position=entity_position.select(1, 3))                        
            xiaoniu_pooled_output = torch.cat([xiaoniu_cls_outputs, xiaoniu_entity_outputs], -1)
            xiaoniu_pooled_output = self.dropout(xiaoniu_pooled_output)
            # Add similarity task
            similarity_google = self.cos(human_pooled_output, google_pooled_output)
            similarity_baidu = self.cos(human_pooled_output, baidu_pooled_output)
            similarity_xiaoniu = self.cos(human_pooled_output, xiaoniu_pooled_output)
            similarity_google = 1 / torch.exp(similarity_google - 1)
            similarity_baidu = 1 / torch.exp(similarity_baidu - 1)
            similarity_xiaoniu = 1 / torch.exp(similarity_xiaoniu - 1)
            product_similarity = similarity_google * similarity_baidu * similarity_xiaoniu
            #print(similarity_google)
            #print(similarity_google.size())
            #print(product_similarity)
            #print(product_similarity.size())
            # now loss_similarity=[batch_size]                        
            loss_similarity = torch.mean(product_similarity)            
            #exit()
            

        else:
            raise ValueError(f"Error mt_system {mt_system}")   
        
        
        logits = self.classifier(human_pooled_output)

        outputs = (loss_similarity, logits,) #  + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                        
            loss = loss * loss_similarity
            outputs = (loss,) + outputs

        return outputs  # (loss, loss_similarity, logits, (hidden_states), (attentions)


@add_start_docstrings("""Bert Model transformer with a relation classification task and similarity task,
Similarity: cos(hi, hj)
head on top (a linear layer on top of the pooled output) e.g. for SemEval tasks. """,
    BERT_START_DOCSTRING, BERT_INPUTS_DOCSTRING)
class BertForRelationClassificationWithSimilarityAllTask(BertPreTrainedModel):
    r"""
    ABC
    """
    def __init__(self, config):
        super(BertForRelationClassificationWithSimilarityAllTask, self).__init__(config)
        self.num_labels = config.num_labels
        # Need to modify the parameters's name in pretrained model, 
        # We add a new classmethod from_pretrained_for_independent_bert in modeling_utils
        self.bert = BertModel(config)        
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size*2, self.config.num_labels)

        self.cos = torch.nn.CosineSimilarity()

        self.init_weights()

    def forward(self, mt_system, input_ids, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, labels=None, entity_position=None):
        """
        mt_system=google, baidu, xiaoniu, all
        combine_mode = "plus", "concate"
        Input is [batch_size, 4, *], 4 for human, google, baidu and xiaoniu, we need to select
        """        

        human_outputs = self.bert(input_ids.select(1, 0),
                                        attention_mask=attention_mask.select(1, 0),
                                        token_type_ids=token_type_ids.select(1, 0),
                                        position_ids=position_ids, 
                                        head_mask=head_mask,
                                        entity_position=entity_position.select(1, 0))
        # pooled_output=[batch_size, hidden_size*2]
        human_pooled_output = human_outputs[1]

        human_pooled_output = self.dropout(human_pooled_output)

        pooled_output = None
        if mt_system == "google":
            google_outputs = self.bert(input_ids.select(1, 1),
                                            attention_mask=attention_mask.select(1, 1),
                                            token_type_ids=token_type_ids.select(1, 1),
                                            position_ids=position_ids, 
                                            head_mask=head_mask,
                                            entity_position=entity_position.select(1, 1))
            # pooled_output=[batch_size, hidden_size*2]
            google_pooled_output = google_outputs[1]
            google_pooled_output = self.dropout(google_pooled_output)
            # Add similarity task
            similarity = self.cos(human_pooled_output, google_pooled_output)
            # now loss_similarity=[batch_size]            
            loss_similarity = 1 / torch.exp(similarity - 1)
            loss_similarity = torch.mean(loss_similarity)                        
            
        elif mt_system == "baidu":
            baidu_outputs = self.bert(input_ids.select(1, 2),
                                            attention_mask=attention_mask.select(1, 2),
                                            token_type_ids=token_type_ids.select(1, 2),
                                            position_ids=position_ids, 
                                            head_mask=head_mask,
                                            entity_position=entity_position.select(1, 2))
            # pooled_output=[batch_size, hidden_size*2]
            baidu_pooled_output = baidu_outputs[1]
            baidu_pooled_output = self.dropout(baidu_pooled_output)
            # Add similarity task
            similarity = self.cos(human_pooled_output, baidu_pooled_output)
            # now loss_similarity=[batch_size]            
            loss_similarity = 1 / torch.exp(similarity - 1)
            loss_similarity = torch.mean(loss_similarity)
        elif mt_system == "xiaoniu":          
            xiaoniu_outputs = self.bert(input_ids.select(1, 3),
                                            attention_mask=attention_mask.select(1, 3),
                                            token_type_ids=token_type_ids.select(1, 3),
                                            position_ids=position_ids, 
                                            head_mask=head_mask,
                                            entity_position=entity_position.select(1, 3))
            # pooled_output=[batch_size, hidden_size*2]
            xiaoniu_pooled_output = xiaoniu_outputs[1]
            xiaoniu_pooled_output = self.dropout(xiaoniu_pooled_output)        
            # Add similarity task
            similarity = self.cos(human_pooled_output, xiaoniu_pooled_output)
            # now loss_similarity=[batch_size]
            loss_similarity = 1 / torch.exp(similarity - 1)
            loss_similarity = torch.mean(loss_similarity)
        elif mt_system == "all":
            google_outputs = self.bert(input_ids.select(1, 1),
                                attention_mask=attention_mask.select(1, 1),
                                token_type_ids=token_type_ids.select(1, 1),
                                position_ids=position_ids, 
                                head_mask=head_mask,
                                entity_position=entity_position.select(1, 1))            
            google_pooled_output = google_outputs[1]
            google_pooled_output = self.dropout(google_pooled_output)
        
            baidu_outputs = self.bert(input_ids.select(1, 2),
                                            attention_mask=attention_mask.select(1, 2),
                                            token_type_ids=token_type_ids.select(1, 2),
                                            position_ids=position_ids, 
                                            head_mask=head_mask,
                                            entity_position=entity_position.select(1, 2))            
            baidu_pooled_output = baidu_outputs[1]
            baidu_pooled_output = self.dropout(baidu_pooled_output)  
        
            xiaoniu_outputs = self.bert(input_ids.select(1, 3),
                                            attention_mask=attention_mask.select(1, 3),
                                            token_type_ids=token_type_ids.select(1, 3),
                                            position_ids=position_ids, 
                                            head_mask=head_mask,
                                            entity_position=entity_position.select(1, 3))            
            xiaoniu_pooled_output = xiaoniu_outputs[1]
            xiaoniu_pooled_output = self.dropout(xiaoniu_pooled_output)
            # Add similarity task
            similarity_human_google = self.cos(human_pooled_output, google_pooled_output)
            similarity_human_baidu = self.cos(human_pooled_output, baidu_pooled_output)
            similarity_human_xiaoniu = self.cos(human_pooled_output, xiaoniu_pooled_output)
            similarity_google_baidu = self.cos(google_pooled_output, baidu_pooled_output)
            similarity_google_xiaoniu = self.cos(google_pooled_output, xiaoniu_pooled_output)
            similarity_baidu_xiaoniu = self.cos(baidu_pooled_output, xiaoniu_pooled_output)
            similarity_human_google = 1 / torch.exp(similarity_human_google - 1)
            similarity_human_baidu = 1 / torch.exp(similarity_human_baidu - 1)
            similarity_human_xiaoniu = 1 / torch.exp(similarity_human_xiaoniu - 1)
            similarity_google_baidu = 1 / torch.exp(similarity_google_baidu - 1)
            similarity_google_xiaoniu = 1 / torch.exp(similarity_google_xiaoniu - 1)
            similarity_baidu_xiaoniu = 1 / torch.exp(similarity_baidu_xiaoniu - 1)

            product_similarity = similarity_human_google * similarity_human_baidu * similarity_human_xiaoniu * similarity_google_baidu * similarity_google_xiaoniu * similarity_baidu_xiaoniu
            #print(similarity_google)
            #print(similarity_google.size())
            #print(product_similarity)
            #print(product_similarity.size())
            # now loss_similarity=[batch_size]                        
            loss_similarity = torch.mean(product_similarity)            
            #exit()
            

        else:
            raise ValueError(f"Error mt_system {mt_system}")   
        
        
        logits = self.classifier(human_pooled_output)

        outputs = (loss_similarity, logits,) #  + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                        
            loss = loss * loss_similarity
            outputs = (loss,) + outputs

        return outputs  # (loss, loss_similarity, logits, (hidden_states), (attentions)


@add_start_docstrings("""Bert Model transformer with a relation classification with independent BERT 
for incorporating with translated data
head on top (a linear layer on top of the pooled output) e.g. for SemEval tasks. """,
    BERT_START_DOCSTRING, BERT_INPUTS_DOCSTRING)
class BertForRelationClassificationWithSimilarityPlusTask(BertPreTrainedModel):
    r"""
    Plus the similarity in a bag
    """
    def __init__(self, config):
        super(BertForRelationClassificationWithSimilarityPlusTask, self).__init__(config)
        self.num_labels = config.num_labels
        # Need to modify the parameters's name in pretrained model, 
        # We add a new classmethod from_pretrained_for_independent_bert in modeling_utils
        self.bert = BertModel(config)        
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size*2, self.config.num_labels)

        self.cos = torch.nn.CosineSimilarity()

        self.init_weights()

    def forward(self, mt_system, input_ids, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, labels=None, entity_position=None):
        """
        mt_system=google, baidu, xiaoniu, all
        combine_mode = "plus", "concate"
        Input is [batch_size, 4, *], 4 for human, google, baidu and xiaoniu, we need to select
        """        

        human_outputs = self.bert(input_ids.select(1, 0),
                                        attention_mask=attention_mask.select(1, 0),
                                        token_type_ids=token_type_ids.select(1, 0),
                                        position_ids=position_ids, 
                                        head_mask=head_mask,
                                        entity_position=entity_position.select(1, 0))
        # pooled_output=[batch_size, hidden_size*2]
        human_pooled_output = human_outputs[1]

        human_pooled_output = self.dropout(human_pooled_output)

        pooled_output = None
        if mt_system == "google":
            google_outputs = self.bert(input_ids.select(1, 1),
                                            attention_mask=attention_mask.select(1, 1),
                                            token_type_ids=token_type_ids.select(1, 1),
                                            position_ids=position_ids, 
                                            head_mask=head_mask,
                                            entity_position=entity_position.select(1, 1))
            # pooled_output=[batch_size, hidden_size*2]
            google_pooled_output = google_outputs[1]
            google_pooled_output = self.dropout(google_pooled_output)
            # Add similarity task
            similarity = self.cos(human_pooled_output, google_pooled_output)
            # now loss_similarity=[batch_size]            
            loss_similarity = 1 / torch.exp(similarity)
            loss_similarity = torch.mean(loss_similarity)                        
            
        elif mt_system == "baidu":
            baidu_outputs = self.bert(input_ids.select(1, 2),
                                            attention_mask=attention_mask.select(1, 2),
                                            token_type_ids=token_type_ids.select(1, 2),
                                            position_ids=position_ids, 
                                            head_mask=head_mask,
                                            entity_position=entity_position.select(1, 2))
            # pooled_output=[batch_size, hidden_size*2]
            baidu_pooled_output = baidu_outputs[1]
            baidu_pooled_output = self.dropout(baidu_pooled_output)
            # Add similarity task
            similarity = self.cos(human_pooled_output, baidu_pooled_output)
            # now loss_similarity=[batch_size]            
            loss_similarity = 1 / torch.exp(similarity)
            loss_similarity = torch.mean(loss_similarity)
        elif mt_system == "xiaoniu":          
            xiaoniu_outputs = self.bert(input_ids.select(1, 3),
                                            attention_mask=attention_mask.select(1, 3),
                                            token_type_ids=token_type_ids.select(1, 3),
                                            position_ids=position_ids, 
                                            head_mask=head_mask,
                                            entity_position=entity_position.select(1, 3))
            # pooled_output=[batch_size, hidden_size*2]
            xiaoniu_pooled_output = xiaoniu_outputs[1]
            xiaoniu_pooled_output = self.dropout(xiaoniu_pooled_output)        
            # Add similarity task
            similarity = self.cos(human_pooled_output, xiaoniu_pooled_output)
            # now loss_similarity=[batch_size]
            loss_similarity = 1 / torch.exp(similarity)
            loss_similarity = torch.mean(loss_similarity)
        elif mt_system == "all":
            google_outputs = self.bert(input_ids.select(1, 1),
                                attention_mask=attention_mask.select(1, 1),
                                token_type_ids=token_type_ids.select(1, 1),
                                position_ids=position_ids, 
                                head_mask=head_mask,
                                entity_position=entity_position.select(1, 1))            
            google_pooled_output = google_outputs[1]
            google_pooled_output = self.dropout(google_pooled_output)
        
            baidu_outputs = self.bert(input_ids.select(1, 2),
                                            attention_mask=attention_mask.select(1, 2),
                                            token_type_ids=token_type_ids.select(1, 2),
                                            position_ids=position_ids, 
                                            head_mask=head_mask,
                                            entity_position=entity_position.select(1, 2))            
            baidu_pooled_output = baidu_outputs[1]
            baidu_pooled_output = self.dropout(baidu_pooled_output)  
        
            xiaoniu_outputs = self.bert(input_ids.select(1, 3),
                                            attention_mask=attention_mask.select(1, 3),
                                            token_type_ids=token_type_ids.select(1, 3),
                                            position_ids=position_ids, 
                                            head_mask=head_mask,
                                            entity_position=entity_position.select(1, 3))            
            xiaoniu_pooled_output = xiaoniu_outputs[1]
            xiaoniu_pooled_output = self.dropout(xiaoniu_pooled_output)
            # Add similarity task
            similarity_google = self.cos(human_pooled_output, google_pooled_output)
            similarity_baidu = self.cos(human_pooled_output, baidu_pooled_output)
            similarity_xiaoniu = self.cos(human_pooled_output, xiaoniu_pooled_output)
            similarity_google = 1 / torch.exp(similarity_google)
            similarity_baidu = 1 / torch.exp(similarity_baidu)
            similarity_xiaoniu = 1 / torch.exp(similarity_xiaoniu)
            plus_similarity = similarity_google + similarity_baidu + similarity_xiaoniu
            #print(similarity_google)
            #print(similarity_google.size())
            #print(product_similarity)
            #print(product_similarity.size())
            # now loss_similarity=[batch_size]                        
            loss_similarity = torch.mean(plus_similarity)            
            #exit()
            

        else:
            raise ValueError(f"Error mt_system {mt_system}")   
        
        
        logits = self.classifier(human_pooled_output)

        outputs = (loss_similarity, logits,) #  + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                        
            loss = loss * loss_similarity
            outputs = (loss,) + outputs

        return outputs  # (loss, loss_similarity, logits, (hidden_states), (attentions)


@add_start_docstrings("""Bert Model with a multiple choice classification head on top (a linear layer on top of
    the pooled output and a softmax) e.g. for RocStories/SWAG tasks. """,
    BERT_START_DOCSTRING, BERT_INPUTS_DOCSTRING)
class BertForMultipleChoice(BertPreTrainedModel):
    r"""
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for computing the multiple choice classification loss.
            Indices should be in ``[0, ..., num_choices]`` where `num_choices` is the size of the second dimension
            of the input tensors. (see `input_ids` above)

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Classification loss.
        **classification_scores**: ``torch.FloatTensor`` of shape ``(batch_size, num_choices)`` where `num_choices` is the size of the second dimension
            of the input tensors. (see `input_ids` above).
            Classification scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForMultipleChoice.from_pretrained('bert-base-uncased')
        choices = ["Hello, my dog is cute", "Hello, my cat is amazing"]
        input_ids = torch.tensor([tokenizer.encode(s) for s in choices]).unsqueeze(0)  # Batch size 1, 2 choices
        labels = torch.tensor(1).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, classification_scores = outputs[:2]

    """
    def __init__(self, config):
        super(BertForMultipleChoice, self).__init__(config)

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)

        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, labels=None):
        num_choices = input_ids.shape[1]

        input_ids = input_ids.view(-1, input_ids.size(-1))
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask)

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, num_choices)

        outputs = (reshaped_logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)
            outputs = (loss,) + outputs

        return outputs  # (loss), reshaped_logits, (hidden_states), (attentions)


@add_start_docstrings("""Bert Model with a token classification head on top (a linear layer on top of
    the hidden-states output) e.g. for Named-Entity-Recognition (NER) tasks. """,
    BERT_START_DOCSTRING, BERT_INPUTS_DOCSTRING)
class BertForTokenClassification(BertPreTrainedModel):
    r"""
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Labels for computing the token classification loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Classification loss.
        **scores**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, config.num_labels)``
            Classification scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForTokenClassification.from_pretrained('bert-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1] * input_ids.size(1)).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, scores = outputs[:2]

    """
    def __init__(self, config):
        super(BertForTokenClassification, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, labels=None):

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids, 
                            head_mask=head_mask)

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), scores, (hidden_states), (attentions)


@add_start_docstrings("""Bert Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear layers on top of
    the hidden-states output to compute `span start logits` and `span end logits`). """,
    BERT_START_DOCSTRING, BERT_INPUTS_DOCSTRING)
class BertForQuestionAnswering(BertPreTrainedModel):
    r"""
        **start_positions**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`).
            Position outside of the sequence are not taken into account for computing the loss.
        **end_positions**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`).
            Position outside of the sequence are not taken into account for computing the loss.

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Total span extraction loss is the sum of a Cross-Entropy for the start and end positions.
        **start_scores**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length,)``
            Span-start scores (before SoftMax).
        **end_scores**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length,)``
            Span-end scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForQuestionAnswering.from_pretrained('bert-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        start_positions = torch.tensor([1])
        end_positions = torch.tensor([3])
        outputs = model(input_ids, start_positions=start_positions, end_positions=end_positions)
        loss, start_scores, end_scores = outputs[:2]

    """
    def __init__(self, config):
        super(BertForQuestionAnswering, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                start_positions=None, end_positions=None):

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids, 
                            head_mask=head_mask)

        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        outputs = (start_logits, end_logits,) + outputs[2:]
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            outputs = (total_loss,) + outputs

        return outputs  # (loss), start_logits, end_logits, (hidden_states), (attentions)


