# USAGE : $ python3 test_siamese_network.py -i1 ./img1.jpg -i2 ./img2.jpg

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from test_helper import *

from scipy.spatial.distance import cdist
from matplotlib import gridspec
import argparse

import os
import textwrap as tw

os.environ['CUDA_VISIBLE_DEVICES'] = '1'



parser = argparse.ArgumentParser(description='Test images.')
parser.add_argument('-i1', '--img1', metavar='--img1', type=str, nargs='?', help='Path to image 1')
parser.add_argument('-i2', '--img2', metavar='--img2', type=str, nargs='?', help='Path to image 2')
args = parser.parse_args()

img_one_path = args.img1
img_two_path = args.img2


if __name__ == '__main__':


    left_input_im = tf.placeholder(tf.float32, [None, 256, 128, 3], 'left_input_im')
    right_input_im = tf.placeholder(tf.float32, [None, 256, 128, 3], 'right_input_im')
    left_label = tf.placeholder(tf.float32, [None, ], 'left_label')
    right_label = tf.placeholder(tf.float32, [None, ], 'right_label')

    print(np.shape(left_input_im), np.shape(right_input_im))
    logits, model_left, model_right = inference(left_input_im, right_input_im)

    # total_loss = tf.losses.get_total_loss()
    global_step = tf.Variable(0, trainable=False)
    global_init = tf.variables_initializer(tf.global_variables())

    from PIL import Image

    img = Image.open(img_one_path)
    img = np.array(img)[np.newaxis, :, :, :]
    
    img2 = Image.open(img_two_path)
    img2 = np.array(img2)[np.newaxis,:,:,:]

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(global_init)
        ckpt = tf.train.get_checkpoint_state("model")
        saver.restore(sess, "model_siamese/model.ckpt")

        my_logits, model_lf, model_rg = sess.run([logits, model_left, model_right], \
                                                 feed_dict={left_input_im: img, right_input_im: img2})

        print(my_logits)
        print(np.shape(model_lf))
        print(np.shape(model_rg))

        lft = np.array(model_lf[0])
        rgt = np.array(model_rg[0])
        l = lft - rgt

        distance = np.sqrt(np.sum((l) ** 2))
        similarity = my_logits * np.square(distance)  # keep the similar label (1) close to each other
        dissimilarity = (1 - np.array(my_logits[0])) * np.square(np.max((0.5 - distance),
                                                                        0))  # give penalty to dissimilar label if the distance is bigger than margin
        similarity_loss = np.mean(dissimilarity + similarity) / 2
        print('distance : ', distance)
        print('similarity : ', similarity)
        print('dissimilarity : ', dissimilarity)
        print('similarity_loss : ', similarity_loss)

        dist = cdist(model_lf, model_rg, 'cosine')
        print('Pairwise distance : ', dist)
        euc = np.linalg.norm(model_lf - model_rg)
        print('euc : ', euc)

        fig = plt.figure()
        plt.title(('Similarity: %f, Dissimilarity: %f\nEuclidean Dist: %f, Logits: %f' % (similarity, dissimilarity, euc, my_logits)), loc='center')
        if my_logits > 0.0:
            textstr = 'Similar'
            props = dict(boxstyle='round', facecolor='green', alpha=0.5)
            fig_txt = tw.fill(tw.dedent(textstr), width=80)
            plt.figtext(0.51, 0.05, fig_txt, horizontalalignment='center',
                        fontsize=12, multialignment='center',
                        bbox=dict(boxstyle="round", facecolor='green',
                                  ec="0.5", pad=0.5, alpha=1), fontweight='bold')
        else:
            textstr = 'Dissimilar'
            props = dict(boxstyle='round', facecolor='red', alpha=0.5)
            fig_txt = tw.fill(tw.dedent(textstr), width=80)
            plt.figtext(0.51, 0.05, fig_txt, horizontalalignment='center',
                        fontsize=12, multialignment='center',
                        bbox=dict(boxstyle="round", facecolor='red',
                                  ec="0.5", pad=0.5, alpha=1), fontweight='bold')



        plt.axis('off')
        ax1 = fig.add_subplot(1, 2, 1)
        l_im = np.array(img)[0]
        ax1.imshow(l_im)
        ax1.axis('off')
        ax2 = fig.add_subplot(1, 2, 2)
        r_im = np.array(img2)[0]
        ax2.imshow(r_im)
        ax2.axis('off')


        plt.show()
