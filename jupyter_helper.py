import tensorflow as tf
import os
import glob
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def make_single_dataset(image_size=[256, 128], tfrecords_path="/home/salehe/siamese/mars_test/mars_test_validation_00000-of-00001.tfrecord", shuffle_buffer_size=2000, repeat=True, train=True):
    """
	Input:
		image_size: size of input images to network
		tfrecords_path: address to tfrecords file containing all image data
		shuffle_buffer_size: number of images to load into a memory for a shuffling 			operation.
		repeat (boolean): repeat dataset
		train (boolean): use in training
	Features:
		image: image tensor
		label: label tensor
		height: original image height
		width: original image width
		addr: image address in file system
	Returns:
		Dataset
	"""

    image_size = tf.cast(image_size, tf.int32)

    def _parse_function(example_proto):

        features = {'image/class/label': tf.FixedLenFeature((), tf.int64, default_value=1),
	    	'image/encoded': tf.FixedLenFeature((), tf.string, default_value=""),
	    	'image/height': tf.FixedLenFeature([], tf.int64),
	    	'image/width': tf.FixedLenFeature([], tf.int64),
	    	'image/format': tf.FixedLenFeature((), tf.string, default_value="")}

        parsed_features = tf.parse_single_example(example_proto, features)
        image_buffer = parsed_features['image/encoded']

        image = tf.image.decode_jpeg(image_buffer,channels=3)
        image = tf.cast(image, tf.float32)

        S = tf.stack([tf.cast(parsed_features['image/height'], tf.int32),
    		tf.cast(parsed_features['image/width'], tf.int32), 3])
        image = tf.reshape(image, S)

        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize_images(image, [256, 128])

        return image, parsed_features['image/class/label'], parsed_features['image/format']

    filenames = [tfrecords_path]

    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.shuffle(shuffle_buffer_size)
    dataset = dataset.map(_parse_function, num_parallel_calls=8)
    return dataset


def combine_dataset(batch_size, image_size, same_prob, diff_prob, repeat=True, train=True):
    """
	Input:
		image size (int)
		batch_size (int)
		same_prob (float): probability of retaining images in same class
		diff_prob (float): probability of retaining images in different class
		train (boolean): train or validation
		repeat (boolean): repeat elements in dataset
	Return:
		zipped dataset

	"""
    dataset_left = make_single_dataset(image_size, repeat=repeat, train=train)
    dataset_right = make_single_dataset(image_size, repeat=repeat, train=train)

    dataset = tf.data.Dataset.zip((dataset_left, dataset_right))

    if train:
        filter_func = create_filter_func(same_prob, diff_prob)
        dataset = dataset.filter(filter_func)
    if repeat:
        dataset = dataset.repeat()

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(1)
    return dataset


def create_filter_func(same_prob, diff_prob):
    def filter_func(left, right):
        _, right_label, _ = left
        _, left_label, _ = right

        label_cond = tf.equal(right_label, left_label)

        different_labels = tf.fill(tf.shape(label_cond), diff_prob)
        same_labels = tf.fill(tf.shape(label_cond), same_prob)

        weights = tf.where(label_cond, same_labels, different_labels)
        random_tensor = tf.random_uniform(shape=tf.shape(weights))

        return weights > random_tensor

    return filter_func


def conv2d_old(prev, filters, kernel_size, strides=1, padding="SAME", name="conv2d", reuse=False, batch_norm=True, maxpool=True):

    layer = tf.layers.conv2d(prev, filters, [kernel_size, kernel_size], strides=strides, padding=padding,
        kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1), name=name, reuse=reuse)
    if maxpool:
        # layer = tf.layers.max_pooling2d(layer, pool_size=2, strides=2, padding='valid')
        layer = tf.contrib.layers.max_pool2d(layer, [2, 2], padding='SAME')
    if batch_norm:
        layer = tf.layers.batch_normalization(layer, fused=True)
    layer = tf.nn.relu(layer)
    return layer


def conv2d(prev, filters, kernel_size, strides=1, padding='SAME', name='conv2d', reuse=False, batch_norm=True, maxpool=True):

    # layer = tf.layers.conv2d(prev, filters, [kernel_size, kernel_size], strides=1, padding='SAME',
    #                          kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1), name=name, reuse=reuse)

    # layer = tf.contrib.layers.conv2d(prev, filters, [kernel_size, kernel_size], activation_fn=tf.nn.relu, padding='SAME',
    #                                        weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(), reuse=reuse)
    layer = tf.contrib.layers.conv2d(prev, filters, [kernel_size, kernel_size], activation_fn=None,
                                     padding='SAME',
                                     weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(), reuse=reuse)

    layer = tf.contrib.layers.max_pool2d(layer, [2, 2], padding='SAME')

    layer = tf.layers.batch_normalization(layer, fused=True)

    layer = tf.nn.relu(layer)

    return layer


def model_old(input_image):
    """
	Input:
		input_image: Tensor of shape (batch_size, image_size, image_size, 3)
	Returns:
		Feature Vector of shape
	"""
    prev = input_image
    # filters = [256, 128, 64, 32]
    filters = [32, 64, 128, 256, 2]
    # kernel_size = [[7, 7], [5, 5], [3, 3], [1, 1], [1, 1]]
    kernel_size = [7, 5, 3, 1, 1]
    print(np.shape(prev))
    for i in range(5): # 4
        prev = conv2d(prev=prev, filters=filters[i], kernel_size=kernel_size[i]) # 3
        print(np.shape(prev))
    flatten = tf.layers.flatten(prev)
    print(np.shape(flatten))

    # single_output = tf.layers.dense(flatten, 1024, activation=tf.sigmoid)
    # print(np.shape(single_output))


    # return single_output
    return flatten


def model_correct(input, reuse=False):
    print(np.shape(input))
    with tf.name_scope("model"):
        with tf.variable_scope("conv1") as scope:
            net = tf.contrib.layers.conv2d(input, 256, [3, 3], activation_fn=None, padding='SAME',
                                           weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                           scope=scope, reuse=reuse)
            print(np.shape(net))
            net = tf.contrib.layers.max_pool2d(net, [2, 2], padding='SAME')
            net = tf.layers.batch_normalization(net, fused=True)
            net = tf.nn.relu(net)
            print(np.shape(net))

        with tf.variable_scope("conv2") as scope:
            net = tf.contrib.layers.conv2d(net, 128, [3, 3], activation_fn=None, padding='SAME',
                                           weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                           scope=scope, reuse=reuse)
            print(np.shape(net))
            net = tf.contrib.layers.max_pool2d(net, [2, 2], padding='SAME')
            net = tf.layers.batch_normalization(net, fused=True)
            net = tf.nn.relu(net)
            print(np.shape(net))

        with tf.variable_scope("conv3") as scope:
            net = tf.contrib.layers.conv2d(net, 64, [3, 3], activation_fn=None, padding='SAME',
                                           weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                           scope=scope, reuse=reuse)
            print(np.shape(net))
            net = tf.contrib.layers.max_pool2d(net, [2, 2], padding='SAME')
            net = tf.layers.batch_normalization(net, fused=True)
            net = tf.nn.relu(net)
            print(np.shape(net))

        with tf.variable_scope("conv4") as scope:
            net = tf.contrib.layers.conv2d(net, 32, [3, 3], activation_fn=None, padding='SAME',
                                           weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                           scope=scope, reuse=reuse)
            print(np.shape(net))
            net = tf.contrib.layers.max_pool2d(net, [2, 2], padding='SAME')
            net = tf.layers.batch_normalization(net, fused=True)
            net = tf.nn.relu(net)
            print(np.shape(net))

        with tf.variable_scope("conv5") as scope:
            net = tf.contrib.layers.conv2d(net, 16, [3, 3], activation_fn=None, padding='SAME',
                                           weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                           scope=scope, reuse=reuse)
            print(np.shape(net))
            net = tf.contrib.layers.max_pool2d(net, [2, 2], padding='SAME')
            net = tf.layers.batch_normalization(net, fused=True)
            net = tf.nn.relu(net)
            print(np.shape(net))

        # with tf.variable_scope("conv6") as scope:
        #     net = tf.contrib.layers.conv2d(net, 1, [2, 2], activation_fn=None, padding='SAME',
        #                                    weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
        #                                    scope=scope, reuse=reuse)
        #     print(np.shape(net))
        #     net = tf.contrib.layers.max_pool2d(net, [2, 2], padding='SAME')
        #     print(np.shape(net))
        #
        # with tf.variable_scope("conv7") as scope:
        #     net = tf.contrib.layers.conv2d(net, 1, [2, 2], activation_fn=None, padding='SAME',
        #                                    weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
        #                                    scope=scope, reuse=reuse)
        #     print(np.shape(net))
        #     net = tf.contrib.layers.max_pool2d(net, [2, 2], padding='SAME')
        #     print(np.shape(net))
        #
        # with tf.variable_scope("conv7") as scope:
        #     net = tf.contrib.layers.conv2d(net, 1, [2, 2], activation_fn=None, padding='SAME',
        #                                    weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
        #                                    scope=scope, reuse=reuse)
        #     print(np.shape(net))
        #     net = tf.contrib.layers.max_pool2d(net, [2, 2], padding='SAME')
        #     print(np.shape(net))

        net = tf.contrib.layers.flatten(net)
        print(np.shape(net))

        # net = tf.nn.sigmoid(net)
        # print(np.shape(net))

        # net = tf.layers.dense(net, 512, activation=tf.sigmoid)
        net = tf.layers.dense(net, 512, activation=None)
        print(np.shape(net))

    return net


def model(input, reuse=False):
    print(np.shape(input))
    with tf.name_scope("model"):
        with tf.variable_scope("conv1") as scope:
            net = tf.contrib.layers.conv2d(input, 256, [3, 3], activation_fn=None, padding='SAME',
                                           weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                           scope=scope, reuse=reuse)
            print(np.shape(net))
            net = tf.layers.max_pooling2d(net, pool_size=2, strides=2, padding='valid')
            net = tf.layers.batch_normalization(net, fused=True)
            net = tf.nn.relu(net)
            print(np.shape(net))

        with tf.variable_scope("conv2") as scope:
            net = tf.contrib.layers.conv2d(net, 128, [3, 3], activation_fn=None, padding='SAME',
                                           weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                           scope=scope, reuse=reuse)
            print(np.shape(net))
            net = tf.layers.max_pooling2d(net, pool_size=2, strides=2, padding='valid')
            net = tf.layers.batch_normalization(net, fused=True)
            net = tf.nn.relu(net)
            print(np.shape(net))

        with tf.variable_scope("conv3") as scope:
            net = tf.contrib.layers.conv2d(net, 64, [3, 3], activation_fn=None, padding='SAME',
                                           weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                           scope=scope, reuse=reuse)
            print(np.shape(net))
            net = tf.layers.max_pooling2d(net, pool_size=2, strides=2, padding='valid')
            net = tf.layers.batch_normalization(net, fused=True)
            net = tf.nn.relu(net)
            print(np.shape(net))

        with tf.variable_scope("conv4") as scope:
            net = tf.contrib.layers.conv2d(net, 32, [3, 3], activation_fn=None, padding='SAME',
                                           weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                           scope=scope, reuse=reuse)
            print(np.shape(net))
            net = tf.layers.max_pooling2d(net, pool_size=2, strides=2, padding='valid')
            net = tf.layers.batch_normalization(net, fused=True)
            net = tf.nn.relu(net)
            print(np.shape(net))

        # with tf.variable_scope("conv5") as scope:
        #     net = tf.contrib.layers.conv2d(net, 16, [3, 3], activation_fn=None, padding='SAME',
        #                                    weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
        #                                    scope=scope, reuse=reuse)
        #     print(np.shape(net))
        #     net = tf.contrib.layers.max_pool2d(net, [2, 2], padding='SAME')
        #     net = tf.layers.batch_normalization(net, fused=True)
        #     net = tf.nn.relu(net)
        #     print(np.shape(net))

        # with tf.variable_scope("conv6") as scope:
        #     net = tf.contrib.layers.conv2d(net, 1, [2, 2], activation_fn=None, padding='SAME',
        #                                    weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
        #                                    scope=scope, reuse=reuse)
        #     print(np.shape(net))
        #     net = tf.contrib.layers.max_pool2d(net, [2, 2], padding='SAME')
        #     print(np.shape(net))
        #
        # with tf.variable_scope("conv7") as scope:
        #     net = tf.contrib.layers.conv2d(net, 1, [2, 2], activation_fn=None, padding='SAME',
        #                                    weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
        #                                    scope=scope, reuse=reuse)
        #     print(np.shape(net))
        #     net = tf.contrib.layers.max_pool2d(net, [2, 2], padding='SAME')
        #     print(np.shape(net))
        #
        # with tf.variable_scope("conv7") as scope:
        #     net = tf.contrib.layers.conv2d(net, 1, [2, 2], activation_fn=None, padding='SAME',
        #                                    weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
        #                                    scope=scope, reuse=reuse)
        #     print(np.shape(net))
        #     net = tf.contrib.layers.max_pool2d(net, [2, 2], padding='SAME')
        #     print(np.shape(net))

        net = tf.contrib.layers.flatten(net)
        print(np.shape(net))

        # net = tf.nn.sigmoid(net)
        # print(np.shape(net))

        net = tf.layers.dense(net, 4096, activation=tf.sigmoid)
        # net = tf.layers.dense(net, 512, activation=None)
        print(np.shape(net))

    return net


def model_orig(input, reuse=False):
    print(np.shape(input))
    with tf.name_scope("model"):
        with tf.variable_scope("conv1") as scope:
            net = tf.contrib.layers.conv2d(input, 32, [7, 7], activation_fn=tf.nn.relu, padding='SAME',
                                           weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                           scope=scope, reuse=reuse)
            print(np.shape(net))
            net = tf.contrib.layers.max_pool2d(net, [2, 2], padding='SAME')
            print(np.shape(net))

        with tf.variable_scope("conv2") as scope:
            net = tf.contrib.layers.conv2d(net, 64, [5, 5], activation_fn=tf.nn.relu, padding='SAME',
                                           weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                           scope=scope, reuse=reuse)
            print(np.shape(net))
            net = tf.contrib.layers.max_pool2d(net, [2, 2], padding='SAME')
            print(np.shape(net))

        with tf.variable_scope("conv3") as scope:
            net = tf.contrib.layers.conv2d(net, 128, [3, 3], activation_fn=tf.nn.relu, padding='SAME',
                                           weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                           scope=scope, reuse=reuse)
            print(np.shape(net))
            net = tf.contrib.layers.max_pool2d(net, [2, 2], padding='SAME')
            print(np.shape(net))

        with tf.variable_scope("conv4") as scope:
            net = tf.contrib.layers.conv2d(net, 256, [2, 2], activation_fn=tf.nn.relu, padding='SAME',
                                           weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                           scope=scope, reuse=reuse)
            print(np.shape(net))
            net = tf.contrib.layers.max_pool2d(net, [2, 2], padding='SAME')
            print(np.shape(net))

        # with tf.variable_scope("conv5") as scope:
        #     net = tf.contrib.layers.conv2d(net, 2, [2, 2], activation_fn=None, padding='SAME',
        #                                    weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
        #                                    scope=scope, reuse=reuse)
        #     print(np.shape(net))
        #     net = tf.contrib.layers.max_pool2d(net, [2, 2], padding='SAME')
        #     print(np.shape(net))
        #
        # with tf.variable_scope("conv6") as scope:
        #     net = tf.contrib.layers.conv2d(net, 1, [2, 2], activation_fn=None, padding='SAME',
        #                                    weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
        #                                    scope=scope, reuse=reuse)
        #     print(np.shape(net))
        #     net = tf.contrib.layers.max_pool2d(net, [2, 2], padding='SAME')
        #     print(np.shape(net))
        #
        # with tf.variable_scope("conv7") as scope:
        #     net = tf.contrib.layers.conv2d(net, 1, [2, 2], activation_fn=None, padding='SAME',
        #                                    weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
        #                                    scope=scope, reuse=reuse)
        #     print(np.shape(net))
        #     net = tf.contrib.layers.max_pool2d(net, [2, 2], padding='SAME')
        #     print(np.shape(net))
        #
        # with tf.variable_scope("conv7") as scope:
        #     net = tf.contrib.layers.conv2d(net, 1, [2, 2], activation_fn=None, padding='SAME',
        #                                    weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
        #                                    scope=scope, reuse=reuse)
        #     print(np.shape(net))
        #     net = tf.contrib.layers.max_pool2d(net, [2, 2], padding='SAME')
        #     print(np.shape(net))

        net = tf.contrib.layers.flatten(net)
        print(np.shape(net))

    return net


def contrastive_loss(model1, model2, y, left_label, right_label, margin=0.2, use_loss=False):
    label = tf.equal(left_label, right_label)
    # label = tf.cast(label, tf.int32)
    y = tf.to_float(label)
    # keep y as logits
    with tf.name_scope("contrastive_loss"):
        distance = tf.sqrt(tf.reduce_sum(tf.pow(model1 - model2, 2), 1, keepdims=True))
        similarity = y * tf.square(distance)  # keep the similar label (1) close to each other
        dissimilarity = (1 - y) * tf.square(tf.maximum((margin - distance),
                                                       0))  # give penalty to dissimilar label if the distance is bigger than margin
        similarity_loss = tf.reduce_mean(dissimilarity + similarity) / 2
        if use_loss:
            tf.losses.add_loss(similarity_loss)


def inference(left_input_image, right_input_image):
    margin = 0.2
    """
	left_input_image: 3D tensor input
	right_input_image: 3D tensor input
	label: 1 if images are from same category. 0 if not.
	"""
    with tf.variable_scope('feature_generator', reuse=tf.AUTO_REUSE) as sc:

        # left_features = model(left_input_image)
        # right_features = model(right_input_image)
        left_features = model(tf.layers.batch_normalization(left_input_image / 255.0))
        right_features = model(tf.layers.batch_normalization(right_input_image / 255.0))

    merged_features = tf.abs(tf.subtract(left_features, right_features))
    # merged_features = tf.maximum(tf.add(-tf.subtract(left_features, right_features), margin), 0.0)
    # merged_features = tf.maximum(margin - tf.abs(tf.subtract(left_features, right_features)), 0)
    # merged_features = tf.maximum(tf.abs(tf.subtract(left_features, right_features)) - margin, 0)
    # logits = tf.contrib.layers.fully_connected(merged_features, num_outputs=1, activation_fn=tf.sigmoid)
    logits = tf.contrib.layers.fully_connected(merged_features, num_outputs=1, activation_fn=None)
    logits = tf.reshape(logits, [-1])
    return logits, left_features, right_features


def loss(logits, left_label, right_label):
    label = tf.equal(left_label, right_label)
    # label = tf.cast(label, tf.int32)

    # label_float = tf.to_float(label)
    label_float = tf.cast(label, tf.float64)

    logits = tf.cast(logits, tf.float64)

    # cross_entropy_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=label))
    cross_entropy_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=label_float))
    tf.losses.add_loss(cross_entropy_loss)


def triplet_loss(anchor, positive, negative, alpha, use_loss=False):
    """Calculate the triplet loss according to the FaceNet paper

    Args:
      anchor: the embeddings for the anchor images.
      positive: the embeddings for the positive images.
      negative: the embeddings for the negative images.

    Returns:
      the triplet loss according to the FaceNet paper as a float tensor.
    """
    with tf.variable_scope('triplet_loss'):
        pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), 1)
        neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), 1)

        basic_loss = tf.add(tf.abs(tf.subtract(pos_dist, neg_dist)), alpha)
        loss = tf.reduce_mean(tf.maximum(basic_loss, 0.0), 0)

    if use_loss:
        tf.losses.add_loss(loss)
