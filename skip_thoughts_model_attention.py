# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""Skip-Thoughts model for learning sentence vectors.

The model is based on the paper:

  "Skip-Thought Vectors"
  Ryan Kiros, Yukun Zhu, Ruslan Salakhutdinov, Richard S. Zemel,
  Antonio Torralba, Raquel Urtasun, Sanja Fidler.
  https://papers.nips.cc/paper/5950-skip-thought-vectors.pdf

Layer normalization is applied based on the paper:

  "Layer Normalization"
  Jimmy Lei Ba, Jamie Ryan Kiros, Geoffrey E. Hinton
  https://arxiv.org/abs/1607.06450
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf

from ops import gru_cell
from ops import input_ops
from modules import *



class SkipThoughtsModel(object):
  """Skip-thoughts model."""

  def __init__(self, config, mode="train", input_reader=None):
    """Basic setup. The actual TensorFlow graph is constructed in build().

    Args:
      config: Object containing configuration parameters.
      mode: "train", "eval" or "encode".
      input_reader: Subclass of tf.ReaderBase for reading the input serialized
        tf.Example protocol buffers. Defaults to TFRecordReader.

    Raises:
      ValueError: If mode is invalid.
    """
    if mode not in ["train", "eval", "encode"]:
      raise ValueError("Unrecognized mode: %s" % mode)

    self.config = config
    self.mode = mode
    self.reader = input_reader if input_reader else tf.TFRecordReader()

    # Initializer used for non-recurrent weights.
    self.uniform_initializer = tf.random_uniform_initializer(
        minval=-self.config.uniform_init_scale,
        maxval=self.config.uniform_init_scale)

    # Input sentences represented as sequences of word ids. "encode" is the
    # source sentence, "decode_pre" is the previous sentence and "decode_post"
    # is the next sentence.
    # Each is an int64 Tensor with  shape [batch_size, padded_length].
    self.encode_ids = None
    self.decode_pre_ids = None
    self.decode_post_ids = None

    # Boolean masks distinguishing real words (1) from padded words (0).
    # Each is an int32 Tensor with shape [batch_size, padded_length].
    self.encode_mask = None
    self.decode_pre_mask = None
    self.decode_post_mask = None

    # Input sentences represented as sequences of word embeddings.
    # Each is a float32 Tensor with shape [batch_size, padded_length, emb_dim].
    self.encode_emb = None
    self.decode_pre_emb = None
    self.decode_post_emb = None

    # The output from the sentence encoder.
    # A float32 Tensor with shape [batch_size, num_gru_units].
    self.thought_vectors = None

    # The cross entropy losses and corresponding weights of the decoders. Used
    # for evaluation.
    self.target_cross_entropy_losses = []
    self.target_cross_entropy_loss_weights = []

    # The total loss to optimize.
    self.total_loss = None

  def build_inputs(self):
    """Builds the ops for reading input data.

    Outputs:
      self.encode_ids
      self.decode_pre_ids
      self.decode_post_ids
      self.encode_mask
      self.decode_pre_mask
      self.decode_post_mask
    """
    if self.mode == "encode":
      # Word embeddings are fed from an external vocabulary which has possibly
      # been expanded (see vocabulary_expansion.py).
      encode_ids = None
      decode_pre_ids = None
      decode_post_ids = None
      encode_mask = tf.placeholder(tf.int8, (None, None), name="encode_mask")
      decode_pre_mask = None
      decode_post_mask = None
    else:
      # Prefetch serialized tf.Example protos.
      input_queue = input_ops.prefetch_input_data(
          self.reader,
          self.config.input_file_pattern,
          shuffle=self.config.shuffle_input_data,
          capacity=self.config.input_queue_capacity,
          num_reader_threads=self.config.num_input_reader_threads)
      # Deserialize a batch.
      serialized = input_queue.dequeue_many(self.config.batch_size)
      encode, decode_pre, decode_post = input_ops.parse_example_batch(
          serialized)

      encode_ids = encode.ids
      decode_pre_ids = decode_pre.ids
      decode_post_ids = decode_post.ids

      encode_mask = encode.mask
      decode_pre_mask = decode_pre.mask
      decode_post_mask = decode_post.mask

    self.encode_ids = encode_ids
    self.decode_pre_ids = decode_pre_ids
    self.decode_post_ids = decode_post_ids

    self.encode_mask = encode_mask
    self.decode_pre_mask = decode_pre_mask
    self.decode_post_mask = decode_post_mask



  def build_word_embeddings(self):
    """Builds the word embeddings.

    Inputs:
      self.encode_ids
      self.decode_pre_ids
      self.decode_post_ids

    Outputs:
      self.encode_emb
      self.decode_pre_emb
      self.decode_post_emb
    """
    if self.mode == "encode":
      # Word embeddings are fed from an external vocabulary which has possibly
      # been expanded (see vocabulary_expansion.py).
      encode_emb = tf.placeholder(tf.float32, (
          None, None, self.config.word_embedding_dim), "encode_emb")
      # No sequences to decode.
      decode_pre_emb = None
      decode_post_emb = None
    else:
      word_emb = tf.get_variable(
          name="word_embedding",
          shape=[self.config.vocab_size, self.config.word_embedding_dim],
          initializer=self.uniform_initializer)

      encode_emb = tf.nn.embedding_lookup(word_emb, self.encode_ids)
      decode_pre_emb = tf.nn.embedding_lookup(word_emb, self.decode_pre_ids)
      decode_post_emb = tf.nn.embedding_lookup(word_emb, self.decode_post_ids)


    self.encode_emb = encode_emb
    self.decode_pre_emb = decode_pre_emb
    self.decode_post_emb = decode_post_emb

  def build_encoder(self):
      """Builds the sentence encoder.

      Inputs:
        self.encode_emb
        self.encode_mask

      Outputs:
        self.thought_vectors

      Raises:
        ValueError: if config.bidirectional_encoder is True and config.encoder_dim
          is odd.
      """
      with tf.variable_scope("encoder"):

          ## Blocks
          for i in range(self.config.num_blocks):
              with tf.variable_scope("enc_num_blocks_{}".format(i)):
                  ### Multihead Attention
                  enc = multihead_attention(queries=self.encode_emb,
                                             keys=self.encode_emb,
                                             mask=self.encode_mask,
                                             num_units=self.config.word_embedding_dim,
                                             num_heads=self.config.nb_head,
                                             dropout_rate=self.config.dropout_keep_prob,
                                             is_training=True,
                                             causality=False,
                                             scope='encode')

                  ### Feed Forward
                  enc = feedforward(enc,
                                         num_units=[4 * self.config.word_embedding_dim, self.config.word_embedding_dim],scope='encode')

          self.thought_vectors = enc
          print("self.thought_vectors", self.thought_vectors)

  def _build_decoder(self, name, embeddings, targets, mask, initial_state,
                     reuse_logits):
      """Builds a sentence decoder.

      Args:
        name: Decoder name.
        embeddings: Batch of sentences to decode; a float32 Tensor with shape
          [batch_size, padded_length, emb_dim].
        targets: Batch of target word ids; an int64 Tensor with shape
          [batch_size, padded_length].
        mask: A 0/1 Tensor with shape [batch_size, padded_length].
        initial_state: Initial state of the GRU. A float32 Tensor with shape
          [batch_size, num_gru_cells].
        reuse_logits: Whether to reuse the logits weights.
      """
      # Decoder RNN.
      with tf.variable_scope("decoder") :

      ## Blocks
          for i in range(self.config.num_blocks):
              with tf.variable_scope("dec_num_blocks_{}".format(i)):
                  ## Multihead Attention ( self-attention)
                  dec = multihead_attention(queries=embeddings,
                                            keys=embeddings,
                                            mask=mask,
                                            num_units=self.config.word_embedding_dim,
                                            num_heads=self.config.nb_head,
                                            dropout_rate= self.config.dropout_keep_prob,
                                            is_training=True,
                                            causality=True,
                                            scope="self_attention"+name
                                            )

              ## Multihead Attention ( vanilla attention)
                  dec = multihead_attention(queries=dec,
                                            keys=initial_state,
                                            num_units=self.config.word_embedding_dim,
                                            num_heads=self.config.nb_head,
                                            dropout_rate=self.config.dropout_keep_prob,
                                            is_training=True,
                                            # causality=False,
                                            scope="vanilla_attention"+name
                                            # reuse=tf.AUTO_REUSE
                                            )

              # ## Feed Forward
                  dec = feedforward(dec, num_units=[4 * self.config.word_embedding_dim, self.config.word_embedding_dim], scope=name)

      # Stack batch vertically.
      decoder_output = tf.reshape(dec, [-1, self.config.word_embedding_dim])
      targets = tf.reshape(targets, [-1])
      weights = tf.to_float(tf.reshape(mask, [-1]))

    # Logits.
      with tf.variable_scope("logits", reuse=reuse_logits) as scope:
          logits = tf.contrib.layers.fully_connected(
                   inputs=decoder_output,
                   num_outputs=self.config.vocab_size,
                   activation_fn=None,
                   weights_initializer=self.uniform_initializer,
                   scope=scope)

      losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
               labels=targets, logits=logits)
      batch_loss = tf.reduce_mean(losses * weights)
      tf.losses.add_loss(batch_loss)

      tf.summary.scalar("losses/" + name, batch_loss)

      self.target_cross_entropy_losses.append(losses)
      self.target_cross_entropy_loss_weights.append(weights)

  def build_decoders(self):
    """Builds the sentence decoders.

    Inputs:
      self.decode_pre_emb
      self.decode_post_emb
      self.decode_pre_ids
      self.decode_post_ids
      self.decode_pre_mask
      self.decode_post_mask
      self.thought_vectors

    Outputs:
      self.target_cross_entropy_losses
      self.target_cross_entropy_loss_weights
    """
    if self.mode != "encode":
      # Pre-sentence decoder.
      self._build_decoder("decoder_pre", self.decode_pre_emb,
                          self.decode_pre_ids, self.decode_pre_mask,
                          self.thought_vectors, False)

      # Post-sentence decoder. Logits weights are reused.
      self._build_decoder("decoder_post", self.decode_post_emb,
                          self.decode_post_ids, self.decode_post_mask,
                          self.thought_vectors, True)

  def build_loss(self):
    """Builds the loss Tensor.

    Outputs:
      self.total_loss
    """
    if self.mode != "encode":
      total_loss = tf.losses.get_total_loss()
      tf.summary.scalar("losses/total", total_loss)

      self.total_loss = total_loss/2

  def build_global_step(self):
    """Builds the global step Tensor.

    Outputs:
      self.global_step
    """
    self.global_step = tf.contrib.framework.create_global_step()

  def build(self):
    """Creates all ops for training, evaluation or encoding."""
    self.build_inputs()
    self.build_word_embeddings()
    self.build_encoder()
    self.build_decoders()
    self.build_loss()
    self.build_global_step()
