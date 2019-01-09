import modeling
import tokenization
import os

import tensorflow as tf

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


class BERTEmbedder(object):
    def __init__(self, bert_config_path):
        bert_config_file = os.path.join(bert_config_path, 'bert_config.json')
        bert_vocab_file = os.path.join(bert_config_path, 'vocab.txt')
        bert_checkpoint_file = os.path.join(bert_config_path, 'bert_model.ckpt')
        
        self.bert_config = modeling.BertConfig.from_json_file(bert_config_file)
        self.tokenizer = tokenization.FullTokenizer(
            vocab_file=bert_vocab_file, do_lower_case=True)
        model_fn = model_fn_builder(
            bert_config=self.bert_config,
            init_checkpoint=bert_checkpoint_file,
            layer_indexes=[-1],
            use_tpu=False,
            use_one_hot_embeddings=False)
        is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
        run_config = tf.contrib.tpu.RunConfig(
            master=None,
            tpu_config=tf.contrib.tpu.TPUConfig(
                num_shards=8,
                per_host_input_for_training=is_per_host)
            )
        # If TPU is not available, this will fall back to normal Estimator on CPU
        # or GPU.
        self.estimator = tf.contrib.tpu.TPUEstimator(
            use_tpu=False,
            model_fn=model_fn,
            config=run_config,
            predict_batch_size=32)

    def _prepare_docs(self, documents):
        doc_features = []
        max_seq_length = -1
        for doc in documents:
            max_seq_length = max(max_seq_length, len(doc))
        for i in range(len(documents)):
            doc = documents[i]
            doc_tokens = []
            doc_input_type_ids = []
            doc_tokens.append('[CLS]')
            doc_input_type_ids.append(0)
            tokens = self.tokenizer.tokenize(doc)
            for token in tokens:
                doc_tokens.append(token)
                doc_input_type_ids.append(0)
                if token in ['.', '?', '!']:
                    doc_tokens.append('[SEP]')
                    doc_input_type_ids.append(0)
            doc_input_ids = self.tokenizer.convert_tokens_to_ids(doc_tokens)
            doc_input_mask = [1] * len(doc_input_type_ids)
            # Zero-pad up to the sequence length.
            while len(doc_input_ids) < max_seq_length:
                doc_input_ids.append(0)
                doc_input_mask.append(0)
                doc_input_type_ids.append(0)

            doc_features.append(
                InputFeatures(
                    unique_id=i,
                    tokens=doc_tokens,
                    input_ids=doc_input_ids,
                    input_mask=doc_input_mask,
                    input_type_ids=doc_input_type_ids))
        return (doc_features, max_seq_length)

    def _prepare_question(self, question):
        que_features = []
        que_tokens = ['[CLS]'] + self.tokenizer.tokenize(question) + ['[SEP]']
        que_input_ids = self.tokenizer.convert_tokens_to_ids(que_tokens)
        que_input_type_ids = [ 0 for t in que_tokens ]
        que_input_mask = [1] * len(que_input_type_ids)
        que_features.append(
            InputFeatures(
                unique_id=0,
                tokens=que_tokens,
                input_ids=que_input_ids,
                input_mask=que_input_mask,
                input_type_ids=que_input_type_ids))
        return (que_features, len(que_tokens))

    def embed_document(self, document):
        (doc_features, max_seq_length) = self._prepare_docs([document])
        doc_input_fn = input_fn_builder(features=doc_features, seq_length=max_seq_length)
        
        doc_res = self.estimator.predict(doc_input_fn, yield_single_examples=True)
        return (doc_res, doc_features)

    def embed_question(self, question):
        (que_features, max_seq_length2) = self._prepare_question(question)
        que_input_fn = input_fn_builder(features=que_features, seq_length=max_seq_length2)
        
        que_res = self.estimator.predict(que_input_fn, yield_single_examples=True)
        return (que_res, que_features)
    

