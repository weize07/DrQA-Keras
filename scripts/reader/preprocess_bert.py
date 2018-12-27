# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
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
"""Extract pre-computed feature vectors from BERT."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import codecs
import collections
import json
import re

import modeling
import tokenization
import tensorflow as tf

flags = tf.flags

FLAGS = flags.FLAGS

flags.DEFINE_string("input_file", None, "")

flags.DEFINE_string("output_file", None, "")

flags.DEFINE_string("layers", "-1,-2,-3,-4", "")

flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer("batch_size", 32, "Batch size for predictions.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

flags.DEFINE_string("master", None,
                    "If using a TPU, the address of the master.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")

flags.DEFINE_bool(
    "use_one_hot_embeddings", False,
    "If True, tf.one_hot will be used for embedding lookups, otherwise "
    "tf.nn.embedding_lookup will be used. On TPUs, this should be True "
    "since it is much faster.")

BERT_DIR = '/Users/weize/Workspace/VENV-3.6/workspace/bert/uncased_L-12_H-768_A-12'


class InputFeatures(object):
  """A single set of features of data."""

  def __init__(self, unique_id, tokens, input_ids, input_mask, input_type_ids):
    self.unique_id = unique_id
    self.tokens = tokens
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.input_type_ids = input_type_ids


def input_fn_builder(features, seq_length):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  all_unique_ids = []
  all_input_ids = []
  all_input_mask = []
  all_input_type_ids = []

  for feature in features:
    all_unique_ids.append(feature.unique_id)
    all_input_ids.append(feature.input_ids)
    all_input_mask.append(feature.input_mask)
    all_input_type_ids.append(feature.input_type_ids)

  def input_fn(params):
    """The actual input function."""
    batch_size = params["batch_size"]

    num_examples = len(features)

    # This is for demo purposes and does NOT scale to large data sets. We do
    # not use Dataset.from_generator() because that uses tf.py_func which is
    # not TPU compatible. The right way to load data is with TFRecordReader.
    d = tf.data.Dataset.from_tensor_slices({
        "unique_ids":
            tf.constant(all_unique_ids, shape=[num_examples], dtype=tf.int32),
        "input_ids":
            tf.constant(
                all_input_ids, shape=[num_examples, seq_length],
                dtype=tf.int32),
        "input_mask":
            tf.constant(
                all_input_mask,
                shape=[num_examples, seq_length],
                dtype=tf.int32),
        "input_type_ids":
            tf.constant(
                all_input_type_ids,
                shape=[num_examples, seq_length],
                dtype=tf.int32),
    })

    d = d.batch(batch_size=batch_size, drop_remainder=False)
    return d

  return input_fn


def model_fn_builder(bert_config, init_checkpoint, layer_indexes, use_tpu,
                     use_one_hot_embeddings):
  """Returns `model_fn` closure for TPUEstimator."""

  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for TPUEstimator."""

    unique_ids = features["unique_ids"]
    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    input_type_ids = features["input_type_ids"]

    model = modeling.BertModel(
        config=bert_config,
        is_training=False,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=input_type_ids,
        use_one_hot_embeddings=use_one_hot_embeddings)

    if mode != tf.estimator.ModeKeys.PREDICT:
      raise ValueError("Only PREDICT modes are supported: %s" % (mode))

    tvars = tf.trainable_variables()
    scaffold_fn = None
    (assignment_map,
     initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(
         tvars, init_checkpoint)
    if use_tpu:

      def tpu_scaffold():
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
        return tf.train.Scaffold()

      scaffold_fn = tpu_scaffold
    else:
      tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    tf.logging.info("**** Trainable Variables ****")
    for var in tvars:
      init_string = ""
      if var.name in initialized_variable_names:
        init_string = ", *INIT_FROM_CKPT*"
      tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                      init_string)

    all_layers = model.get_all_encoder_layers()

    predictions = {
        "unique_id": unique_ids,
    }

    for (i, layer_index) in enumerate(layer_indexes):
      predictions["layer_output_%d" % i] = all_layers[layer_index]

    output_spec = tf.contrib.tpu.TPUEstimatorSpec(
        mode=mode, predictions=predictions, scaffold_fn=scaffold_fn)
    return output_spec

  return model_fn


def convert_dataset_to_features(dataset, seq_length, tokenizer):
  """Loads a data file into a list of `InputBatch`s."""

  doc_features = []
  que_features = []
  for drqa_id in dataset:
    example = dataset[drqa_id]
    unique_id = example['uid']
    doc_tokens = []
    doc_input_type_ids = []
    doc_tokens.append("[CLS]")
    doc_input_type_ids.append(0)
    for line in example['document']:
      tokens = tokenizer.tokenize(line)
      for token in tokens:
        doc_tokens.append(token)
        doc_input_type_ids.append(0)
      doc_tokens.append("[SEP]")
      doc_input_type_ids.append(0)
    doc_tokens = doc_tokens[:seq_length]
    doc_input_type_ids = doc_input_type_ids[:seq_length]

    que_tokens = tokenizer.tokenize(example['question'])
    que_tokens = ['[CLS]'] + que_tokens + ['[SEP]']
    que_input_type_ids = [ 0 for t in que_tokens ] 

    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
    doc_input_ids = tokenizer.convert_tokens_to_ids(doc_tokens)
    que_input_ids = tokenizer.convert_tokens_to_ids(que_tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    doc_input_mask = [1] * len(doc_input_type_ids)
    que_input_mask = [1] * len(que_input_type_ids)

    # Zero-pad up to the sequence length.
    while len(doc_input_ids) < seq_length:
      doc_input_ids.append(0)
      doc_input_mask.append(0)
      doc_input_type_ids.append(0)

    while len(que_input_ids) < seq_length:
      que_input_ids.append(0)
      que_input_mask.append(0)
      que_input_type_ids.append(0)

    assert len(doc_input_ids) == seq_length
    assert len(doc_input_mask) == seq_length
    assert len(doc_input_type_ids) == seq_length

    assert len(que_input_ids) == seq_length
    assert len(que_input_mask) == seq_length
    assert len(que_input_type_ids) == seq_length

    doc_features.append(
        InputFeatures(
            unique_id=unique_id,
            tokens=tokens,
            input_ids=doc_input_ids,
            input_mask=doc_input_mask,
            input_type_ids=doc_input_type_ids))
    que_features.append(
        InputFeatures(
            unique_id=unique_id,
            tokens=tokens,
            input_ids=que_input_ids,
            input_mask=que_input_mask,
            input_type_ids=que_input_type_ids))
  return (doc_features, que_features)


def read_dataset(input_file):
  """Read a list of `InputExample`s from an input file."""
  dataset = {}
  uid = 1
  with open(input_file, "r") as reader:
    for line in reader:
        json_obj = json.loads(line)
        document = []
        cur_line = ''
        for token in json_obj['document']:
            cur_line = cur_line + ' ' + token
            if token in ['.', '?', '!']:
                document.append(cur_line)
                cur_line = ''
        if cur_line != '':
            document.append(cur_line)
        dataset[json_obj['id']] = {'uid': uid, 'question': ' '.join(json_obj['question']), 'document': document}
        uid += 1
  return dataset


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)

  layer_indexes = [int(x) for x in FLAGS.layers.split(",")]

  bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

  tokenizer = tokenization.FullTokenizer(
      vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

  is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
  run_config = tf.contrib.tpu.RunConfig(
      master=FLAGS.master,
      tpu_config=tf.contrib.tpu.TPUConfig(
          num_shards=FLAGS.num_tpu_cores,
          per_host_input_for_training=is_per_host))

  dataset = read_dataset(FLAGS.input_file)
  unique_id_to_drqa_id = {}
  for d in dataset:
    unique_id_to_drqa_id[dataset[d]['uid']] = d

  (doc_features, que_features) = convert_dataset_to_features(
      dataset=dataset, seq_length=FLAGS.max_seq_length, tokenizer=tokenizer)

  doc_unique_id_to_feature = {}
  for feature in doc_features:
    doc_unique_id_to_feature[feature.unique_id] = feature

  que_unique_id_to_feature = {}
  for feature in que_features:
    que_unique_id_to_feature[feature.unique_id] = feature

  model_fn = model_fn_builder(
      bert_config=bert_config,
      init_checkpoint=FLAGS.init_checkpoint,
      layer_indexes=layer_indexes,
      use_tpu=FLAGS.use_tpu,
      use_one_hot_embeddings=FLAGS.use_one_hot_embeddings)

  # If TPU is not available, this will fall back to normal Estimator on CPU
  # or GPU.
  estimator = tf.contrib.tpu.TPUEstimator(
      use_tpu=FLAGS.use_tpu,
      model_fn=model_fn,
      config=run_config,
      predict_batch_size=FLAGS.batch_size)

  fn_and_output(estimator, layer_indexes, doc_features, doc_unique_id_to_feature, unique_id_to_drqa_id, 'doc')
  fn_and_output(estimator, layer_indexes, que_features, que_unique_id_to_feature, unique_id_to_drqa_id, 'question')
  
def fn_and_output(estimator, layer_indexes, features, unique_id_to_feature, unique_id_to_drqa_id, feature_type):
  input_fn = input_fn_builder(
      features=features, seq_length=FLAGS.max_seq_length)

  with codecs.getwriter("utf-8")(tf.gfile.Open(feature_type + '_' + FLAGS.output_file,
                                               "w")) as writer:
    for result in estimator.predict(input_fn, yield_single_examples=True):
      unique_id = int(result["unique_id"])
      feature = unique_id_to_feature[unique_id]
      output_json = collections.OrderedDict()
      output_json["linex_index"] = unique_id
      output_json["id"] = unique_id_to_drqa_id[unique_id]
      all_features = []
      for (i, token) in enumerate(feature.tokens):
        all_layers = []
        for (j, layer_index) in enumerate(layer_indexes):
          layer_output = result["layer_output_%d" % j]
          layers = collections.OrderedDict()
          layers["index"] = layer_index
          layers["values"] = [
              round(float(x), 6) for x in layer_output[i:(i + 1)].flat
          ]
          all_layers.append(layers)
        features = collections.OrderedDict()
        features["token"] = token
        features["layers"] = all_layers
        all_features.append(features)
      output_json["features"] = all_features
      writer.write(json.dumps(output_json) + "\n")


if __name__ == "__main__":
  flags.mark_flag_as_required("input_file")
  flags.mark_flag_as_required("vocab_file")
  flags.mark_flag_as_required("bert_config_file")
  flags.mark_flag_as_required("init_checkpoint")
  flags.mark_flag_as_required("output_file")
  tf.app.run()
