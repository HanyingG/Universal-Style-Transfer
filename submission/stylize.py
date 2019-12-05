import numpy as np
import tensorflow as tf
import os
import argparse

from model import Encoder_decoder
from utils import get_img, WCT, save_image, resize_to


parser = argparse.ArgumentParser()

parser.add_argument('--content-path', type=str, dest='content_path', help='Path to content image')
parser.add_argument('--style-path', type=str, dest='style_path', help='Path to sytle image')
parser.add_argument('--out-path', type=str, dest='out_path', help='Path to output folder', default='output')
parser.add_argument('--target-layers', nargs='+', type=str, help='Layers in VGG 19 model for WCT transfroms', required=True)
parser.add_argument('--alpha', type=float, help="Strength of feature transform", default=0.6)
parser.add_argument('--checkpoint-dir', type=str, dest='checkpoint_dir', help='Path to checkpoint folder', default='models')

args = parser.parse_args()


def stylize(img, style, target_layers, alpha, checkpoint_dir='models'):
    tensor_dict = Encoder_decoder().get_encoder_decoder(target_layers)
    img = np.expand_dims(img, 0)
    style = np.expand_dims(style, 0)
    for target_layer in target_layers:
    
        decoder_name = 'decoder_{}'.format(target_layer)
        inp, encoded, decoded, _ = tensor_dict[target_layer]

        var_list=[v for v in tf.trainable_variables() if decoder_name in v.name]
        saver = tf.train.Saver(var_list=var_list)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            if tf.train.latest_checkpoint('{}/{}'.format(checkpoint_dir, decoder_name)) is None:
                raise ValueError('No checkpoint found for {}'.format(decoder_name))
            saver.restore(sess, tf.train.latest_checkpoint('{}/{}'.format(checkpoint_dir, decoder_name)))

            encoded_image = sess.run(encoded, feed_dict={inp:img})
            encoded_style = sess.run(encoded, feed_dict={inp:style})

            styled_feature = sess.run(WCT(encoded_image, encoded_style, alpha))

            img = sess.run(decoded, feed_dict={encoded:styled_feature})

    return img[0]

if __name__ == '__main__':
    img = resize_to(get_img(args.content_path), 512)
    style = resize_to(get_img(args.style_path), 512)
    styled_img = stylize(img, style, args.target_layers, args.alpha, checkpoint_dir=args.checkpoint_dir)
    save_image(styled_img, '{}/output.jpg'.format(args.out_path))

