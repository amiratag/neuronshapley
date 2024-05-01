# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Contains common code shared by all inception models.

Usage of arg scope:
    with slim.arg_scope(inception_arg_scope()):
        logits, end_points = inception.inception_v3(images, num_classes,
                                                                                                is_training=is_training)

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from multiprocessing import dummy as multiprocessing
from PIL import Image
import tensorflow as tf
import numpy as np
slim = tf.contrib.slim


def inception_arg_scope(weight_decay=0.00004,
                        use_batch_norm=True,
                        batch_norm_decay=0.9997,
                        batch_norm_epsilon=0.001,
                        activation_fn=tf.nn.relu,
                        batch_norm_updates_collections=tf.GraphKeys.UPDATE_OPS):
    """Defines the default arg scope for inception models.

    Args:
        weight_decay: The weight decay to use for regularizing the model.
        use_batch_norm: "If `True`, batch_norm is applied after each convolution.
        batch_norm_decay: Decay for batch norm moving average.
        batch_norm_epsilon: Small float added to variance to avoid dividing by zero
            in batch norm.
        activation_fn: Activation function for conv2d.
        batch_norm_updates_collections: Collection for the update ops for
            batch norm.

    Returns:
        An `arg_scope` to use for the inception models.
    """
    batch_norm_params = {
            # Decay for the moving averages.
            'decay': batch_norm_decay,
            # epsilon to prevent 0s in variance.
            'epsilon': batch_norm_epsilon,
            # collection containing update_ops.
            'updates_collections': batch_norm_updates_collections,
            # use fused batch norm if possible.
            'fused': None,
    }
    if use_batch_norm:
        normalizer_fn = slim.batch_norm
        normalizer_params = batch_norm_params
    else:
        normalizer_fn = None
        normalizer_params = {}
    # Set weight_decay for weights in Conv and FC layers.
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                                            weights_regularizer=slim.l2_regularizer(weight_decay)):
        with slim.arg_scope(
                [slim.conv2d],
                weights_initializer=slim.variance_scaling_initializer(),
                activation_fn=activation_fn,
                normalizer_fn=normalizer_fn,
                normalizer_params=normalizer_params) as sc:
            return sc

        
def load_image_from_file(filename, shape=None):
    if not tf.gfile.Exists(filename):
        tf.logging.error('Cannot find file: {}'.format(filename))
        return None
    try:
        if shape is None:
            img = np.array(Image.open(filename))
        else:
            img = np.array(Image.open(filename).resize(
                    shape, Image.BILINEAR))
        # Normalize pixel values to between 0 and 1.
        img = np.float32(img) / 255.0
        if len(img.shape) == 2:
            return np.stack([img, img, img], axis=-1)
        if len(img.shape) == 3:
            if img.shape[-1] == 4:
                return img[:, :, :3]
            return img
        else:
            return None
    except Exception as e:
        tf.logging.info(e)
        return None
    return img


def load_images_from_files(filenames, max_imgs=500, return_filenames=False,
                           do_shuffle=False, run_parallel=True, shape=(299, 299),
                           num_workers=10):
    imgs = []
    filenames = filenames[:]
    if do_shuffle:
        np.random.shuffle(filenames)
    if return_filenames:
        final_filenames = []
    if run_parallel:
        pool = multiprocessing.Pool(num_workers)
        imgs = pool.map(lambda filename: load_image_from_file(filename, shape),
                                        filenames[:max_imgs])
        if return_filenames:
            final_filenames = [f for i, f in enumerate(filenames[:max_imgs])
                                                 if imgs[i] is not None]
        imgs = [img for img in imgs if img is not None]
        pool.close()
    else:
        for filename in filenames:
            img = load_image_from_file(filename, shape)
            if img is not None:
                imgs.append(img)
                if return_filenames:
                    final_filenames.append(filename)
            if len(imgs) >= max_imgs:
                break

    if return_filenames:
        return np.array(imgs), final_filenames
    else:
        return np.array(imgs)