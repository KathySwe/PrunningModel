import tensorboard
import tensorflow as tf
import tempfile
import zipfile
import os
import numpy as np

from tensorflow_model_optimization.sparsity import keras as sparsity
import tensorflow_model_optimization as tfm

#tf.enable_eager_execution()
tf.compat.v1.enable_eager_execution()
#//////////////////////////////////////////////////////////////////////////


def relu(x): return tf.keras.layers.Activation('relu')(x)


def conv(x, nf, ks, name, weight_decay):

    kernel_reg = tf.keras.regularizers.l2(weight_decay[0]) if weight_decay else None
    bias_reg = tf.keras.regularizers.l2(weight_decay[1]) if weight_decay else None

    epochs = 100
    num_train_samples = 118287
    end_step = np.ceil(1.0 * num_train_samples / 2).astype(np.int32) * epochs
    # print('End step: ' + str(end_step))

    pruning_params = {
        'pruning_schedule': tfm.sparsity.keras.PolynomialDecay(initial_sparsity=0.50,
                                                     final_sparsity=0.90,
                                                     begin_step=10,
                                                     end_step=end_step,
                                                     frequency=100)
    }

    x = tfm.sparsity.keras.prune_low_magnitude(tf.keras.layers.Conv2D(nf, (ks, ks), padding='same', name=name,
               kernel_regularizer=kernel_reg,
               bias_regularizer=bias_reg,
               kernel_initializer=tf.initializers.random_normal(stddev=0.01),
               bias_initializer=tf.initializers.constant(0.0)),**pruning_params)(x)
    return x


def pooling(x, ks, st, name):
    x = tf.keras.layers.MaxPool2D((ks, ks), strides=(st, st), name=name)(x)
    return x


def vgg_block(x, weight_decay):
    # Block 1
    x = conv(x, 64, 3, "conv1_1", (weight_decay, 0))
    x = relu(x)
    x = conv(x, 64, 3, "conv1_2", (weight_decay, 0))
    x = relu(x)
    x = pooling(x, 2, 2, "pool1_1")

    # Block 2
    x = conv(x, 128, 3, "conv2_1", (weight_decay, 0))
    x = relu(x)
    x = conv(x, 128, 3, "conv2_2", (weight_decay, 0))
    x = relu(x)
    x = pooling(x, 2, 2, "pool2_1")

    # Block 3
    x = conv(x, 256, 3, "conv3_1", (weight_decay, 0))
    x = relu(x)
    x = conv(x, 256, 3, "conv3_2", (weight_decay, 0))
    x = relu(x)
    x = conv(x, 256, 3, "conv3_3", (weight_decay, 0))
    x = relu(x)
    x = conv(x, 256, 3, "conv3_4", (weight_decay, 0))
    x = relu(x)
    x = pooling(x, 2, 2, "pool3_1")

    # Block 4
    x = conv(x, 512, 3, "conv4_1", (weight_decay, 0))
    x = relu(x)
    x = conv(x, 512, 3, "conv4_2", (weight_decay, 0))
    x = relu(x)

    # Additional non vgg layers
    x = conv(x, 256, 3, "conv4_3_CPM", (weight_decay, 0))
    x = relu(x)
    x = conv(x, 128, 3, "conv4_4_CPM", (weight_decay, 0))
    x = relu(x)

    return x


def stage1_block(x, num_p, branch, weight_decay):

    # Block 1


    x = conv(x, 128, 3, "Mconv1_stage1_L%d" % branch, (weight_decay, 0))
    x = relu(x)

    x = conv(x, 128, 3, "Mconv2_stage1_L%d" % branch, (weight_decay, 0))
    x = relu(x)
    x = conv(x, 128, 3, "Mconv3_stage1_L%d" % branch, (weight_decay, 0))
    x = relu(x)
    x = conv(x, 512, 1, "Mconv4_stage1_L%d" % branch, (weight_decay, 0))
    x = relu(x)
    x = conv(x, num_p, 1, "Mconv5_stage1_L%d" % branch, (weight_decay, 0))

    return x


def stageT_block(x, num_p, stage, branch, weight_decay):
    # Block 1
    x = conv(x, 128, 7, "Mconv1_stage%d_L%d" % (stage, branch), (weight_decay, 0))
    x = relu(x)
    x = conv(x, 128, 7, "Mconv2_stage%d_L%d" % (stage, branch), (weight_decay, 0))
    x = relu(x)
    x = conv(x, 128, 7, "Mconv3_stage%d_L%d" % (stage, branch), (weight_decay, 0))
    x = relu(x)
    x = conv(x, 128, 7, "Mconv4_stage%d_L%d" % (stage, branch), (weight_decay, 0))
    x = relu(x)
    x = conv(x, 128, 7, "Mconv5_stage%d_L%d" % (stage, branch), (weight_decay, 0))
    x = relu(x)
    x = conv(x, 128, 1, "Mconv6_stage%d_L%d" % (stage, branch), (weight_decay, 0))
    x = relu(x)
    x = conv(x, num_p, 1, "Mconv7_stage%d_L%d" % (stage, branch), (weight_decay, 0))

    return x


def apply_mask(x, mask1, mask2, num_p, stage, branch):
    w_name = "weight_stage%d_L%d" % (stage, branch)
    if num_p == 38:
        w = tf.keras.layers.Multiply(name=w_name)([x, mask1]) # vec_weight

    else:
        w = tf.keras.layers.Multiply(name=w_name)([x, mask2])  # vec_heat
    return w


def get_training_model(weight_decay):

    stages = 6
    np_branch1 = 38
    np_branch2 = 19

    img_input_shape = (None, None, 3)
    vec_input_shape = (None, None, 38)
    heat_input_shape = (None, None, 19)

    inputs = []
    outputs = []

    img_input = tf.keras.layers.Input(shape=img_input_shape)
    vec_weight_input = tf.keras.layers.Input(shape=vec_input_shape)
    heat_weight_input = tf.keras.layers.Input(shape=heat_input_shape)

    inputs.append(img_input)
    inputs.append(vec_weight_input)
    inputs.append(heat_weight_input)

    img_normalized = tf.keras.layers.Lambda(lambda x: x / 256 - 0.5)(img_input) # [-0.5, 0.5]

    # VGG
    stage0_out = vgg_block(img_normalized, weight_decay)  #!!!!! to create feature maps for the input image.

    # stage 1 - branch 1 (PAF)
    stage1_branch1_out = stage1_block(stage0_out, np_branch1, 1, weight_decay) #!!!!!predicts a set of 2D Confidence Maps (S) of body part locations
    w1 = apply_mask(stage1_branch1_out, vec_weight_input, heat_weight_input, np_branch1, 1, 1)

    # stage 1 - branch 2 (confidence maps) #!!!!!predicts a set of 2D vector fields (L) of Part Affinities (PAF), which encode the degree of association between parts (keypoints)
    stage1_branch2_out = stage1_block(stage0_out, np_branch2, 2, weight_decay)
    w2 = apply_mask(stage1_branch2_out, vec_weight_input, heat_weight_input, np_branch2, 1, 2)

    x = tf.keras.layers.Concatenate()([stage1_branch1_out, stage1_branch2_out, stage0_out])

    outputs.append(w1)
    outputs.append(w2)

    # stage sn >= 2
    for sn in range(2, stages + 1):

        # stage SN - branch 1 (PAF)
        stageT_branch1_out = stageT_block(x, np_branch1, sn, 1, weight_decay)
        w1 = apply_mask(stageT_branch1_out, vec_weight_input, heat_weight_input, np_branch1, sn, 1)

        # stage SN - branch 2 (confidence maps)
        stageT_branch2_out = stageT_block(x, np_branch2, sn, 2, weight_decay)
        w2 = apply_mask(stageT_branch2_out, vec_weight_input, heat_weight_input, np_branch2, sn, 2)

        outputs.append(w1)
        outputs.append(w2)

        if (sn < stages):
            x = tf.keras.layers.Concatenate()([stageT_branch1_out, stageT_branch2_out, stage0_out])

    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)

    return model


def get_testing_model():
    stages = 6
    np_branch1 = 38
    np_branch2 = 19

    img_input_shape = (None, None, 3)

    img_input = tf.keras.layers.Input(shape=img_input_shape)

    img_normalized = tf.keras.layers.Lambda(lambda x: x / 256 - 0.5)(img_input) # [-0.5, 0.5]

    # VGG
    stage0_out = vgg_block(img_normalized, None)

    # stage 1 - branch 1 (PAF)
    stage1_branch1_out = stage1_block(stage0_out, np_branch1, 1, None)

    # stage 1 - branch 2 (confidence maps)
    stage1_branch2_out = stage1_block(stage0_out, np_branch2, 2, None)

    x = tf.keras.layers.Concatenate()([stage1_branch1_out, stage1_branch2_out, stage0_out])

    # stage t >= 2
    stageT_branch1_out = None
    stageT_branch2_out = None
    for sn in range(2, stages + 1):
        stageT_branch1_out = stageT_block(x, np_branch1, sn, 1, None)
        stageT_branch2_out = stageT_block(x, np_branch2, sn, 2, None)

        if (sn < stages):
            x = tf.keras.layers.Concatenate()([stageT_branch1_out, stageT_branch2_out, stage0_out])

    model = tf.keras.models.Model(inputs=[img_input], outputs=[stageT_branch1_out, stageT_branch2_out])

    return model