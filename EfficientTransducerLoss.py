#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright 2021 Lucky Wong
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""Efficient Transducer Loss definition."""

from warprnnt_tensorflow import rnnt_loss

class EfficientTransducerLoss():
    """ Transducer loss utterance by utterance.
        Ref: IMPROVING RNN TRANSDUCER MODELING FOR END-TO-END SPEECH RECOGNITION

        Instead, we implement the combination sequence by sequence. 
        Then we concatenate all zn instead of paralleling them, which means we convert z into a two-dimension tensor (􏰀Nn=1 Tn ∗ Un), D). 

        Computes the RNNT loss between a sequence of activations and a
        ground truth labeling.

        Args:
            blank_label: int, the label value/index that the RNNT
                         calculation should use as the blank label
    """

    def __init__(self, blank_index=0, name="EfficientTransducerLoss"):
        self.blank_index = blank_index

    def __call__(self, logits, labels, label_length, logit_length):
        """
        Computes the RNNT loss between a sequence of activations and a
        ground truth labeling.

        Args:
            logits: A 2-D Tensor of floats.  The dimensions should be (Sum(T_i*U_i), V),
                         T is the time index, U is the prediction network sequence
                         length, and V indexes over activations for each
                         symbol in the alphabet.
            labels: A 2-D Tensor of ints, a padded label sequences to make sure
                         labels for the minibatch are same length.
            label_lengths: A 1-D Tensor of ints, the length of each label
                           for each example in the minibatch.
            logit_length: A 1-D Tensor of ints, the number of time steps
                           for each sequence in the minibatch.
        Returns:
            1-D float Tensor, the cost of each example in the minibatch
            (as negative log probabilities).

        * This class performs the softmax operation internally.
        * The label reserved for the blank symbol should be label 0.
        """
        batch_size, = shape_list(logit_length)
        i = tf.constant(0)
        start = tf.constant(0)
        losses = tf.constant(0.0, dtype=tf.float32)

        def body(i, start, losses):
            def expand_batch_axis(x):
                return tf.expand_dims(x, 0)
            t = logit_length[i]
            u = label_length[i]
            tu = t*(u+1)  # +1 means blank label prepanded

            # Slice and reshape
            # [t*u, V] => [t, u, V]
            logit_i = tf.reshape(
                tf.slice(logits, [start, 0], [tu, -1]),
                shape=[t, u+1, shape_list(logits)[-1]]
            )
            if logit_i.dtype.base_dtype == tf.float16:
                logit_i = tf.cast(logit_i, dtype=tf.float32)
            label_i = labels[i, :u+1]

            loss = rnnt_loss(
                acts=expand_batch_axis(logit_i),
                labels=expand_batch_axis(label_i),
                input_lengths=expand_batch_axis(t),
                label_lengths=expand_batch_axis(u),
                blank_label=self.blank_index)[0]

            return (tf.add(i, 1), tf.add(start, tu), tf.add(losses, loss))

        _, start, losses = tf.while_loop(
            lambda i, a, b: tf.less(i, batch_size),
            body,
            [i, start, losses]
        )

        return losses/tf.cast(batch_size, dtype=tf.float32)
