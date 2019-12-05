import tensorflow as tf
import numpy as np
from multiprocessing import Pool

from utils import get_cropped_image, get_files
from model import Encoder_decoder

import argparse


parser = argparse.ArgumentParser()

parser.add_argument('--target-decoder', type=str, dest='target_decoder', help='Decoder to train')
parser.add_argument('--train-imgs', type=str, dest='train_imgs', help='Path to train images', default='train_images/train2017')
parser.add_argument('--val-imgs', type=str, dest='val_imgs', help='Path to validation images', default='val_images/val2017')
parser.add_argument('--checkpoint-dir', type=str, dest='checkpoint_dir', help='Path to checkpoint folder', default='ckpts')

args = parser.parse_args()


class Train_utils():  
    @staticmethod
    def loss(tensor_list, feature_weight=1, pixel_weight=1, variation_weight=0, name_scope=''):
        inp, encoded, decoded, second_encoded = tensor_list

        with tf.variable_scope(name_scope):
            # Feature loss between encodings of original & reconstructed
            feature_loss = feature_weight * tf.losses.mean_squared_error(second_encoded, encoded)

            # Pixel reconstruction loss between decoded/reconstructed img and original
            pixel_loss = pixel_weight * tf.losses.mean_squared_error(decoded, inp)

            # Total Variation loss
            if variation_weight > 0:
                variation_loss = variation_weight * tf.reduce_mean(tf.image.total_variation(decoded))
            else:
                variation_loss = tf.constant(0.)

            total_loss = feature_loss + pixel_loss + variation_loss
        return total_loss
    
    @staticmethod
    def train_step(loss, learning_rate=1e-4, optimizer = 'Adam',momentum = 0.5,decay=0.9,epsilon=1e-10, beta1=0.9,beta2=0.999):
        with tf.variable_scope('train_step', reuse = tf.AUTO_REUSE):
            if optimizer == 'SGD':
                step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
            elif optimizer == 'momentum':
                step = tf.train.MomentumOptimizer(learning_rate,momentum).minimize(loss)
            elif optimizer == 'RMSProp':
                step = tf.train.RMSPropOptimizer(learing_rate,decay=decay,momentum=momentum,epsilon=epsilon).minimize(loss)
            elif optimizer == 'Adam':
                step = tf.train.AdamOptimizer(learning_rate,beta1=beta1,beta2=beta2,epsilon=epsilon).minimize(loss)
        return step
    
    
class Batch_Generator():
    def __init__(self, img_dir, pool=8):
        self.file_list = get_files(img_dir)
        self.p = Pool(pool)
                
    def get_batch(self, batch_size=128):
        img_seq = np.random.choice(len(self.file_list),size=batch_size)
        files = map(lambda x: self.file_list[x], img_seq)
        images = self.p.map(get_cropped_image, files)
        return np.array(images)
    
    
def train_decoder(target_decoder, 
                  train_imgs='train_images/train2017', val_imgs='val_images/val2017',
                  batch_size=8, iters=30000, look_back_period=5000, val_per_iter=100,
                  feature_weight=1, pixel_weight=1, variation_weight=0, 
                  learning_rate=1e-5, optimizer = 'Adam',
                  checkpoint_dir='models'):

    decoder_name = 'decoder_{}'.format(target_decoder)
    
    tensor_list = Encoder_decoder().get_encoder_decoder([target_decoder])[target_decoder]
    var_list=[v for v in tf.trainable_variables() if decoder_name in v.name]

    train_gen = Batch_Generator(train_imgs)
    val_gen = Batch_Generator(val_imgs)

    inp = tensor_list[0]
    loss = Train_utils.loss(tensor_list, feature_weight, pixel_weight, variation_weight)
    out = tensor_list[2]
    step = Train_utils.train_step(loss, learning_rate=5e-5, optimizer = 'Adam')

    train_loss_list = []
    val_loss_list = []

    saver = tf.train.Saver(var_list=var_list)
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        best_loss = 1e10
        best_at = -1

        if tf.train.latest_checkpoint('{}/{}'.format(checkpoint_dir, decoder_name)):
            print('Checkpoint found, restore from before... ')
            saver.restore(sess, tf.train.latest_checkpoint('{}/{}'.format(checkpoint_dir,decoder_name)))
            val_batch_x = val_gen.get_batch(batch_size*8)
            val_loss = sess.run(loss, feed_dict={inp: val_batch_x})
            best_loss = val_loss
            best_at = 0
            print('val_loss', val_loss)
        else:
            print('No checkpoint found, train from beginning')

        for itr in range(iters):
            training_batch_x = train_gen.get_batch(batch_size)
            _, cur_loss = sess.run([step, loss], feed_dict={inp: training_batch_x})

            # evaluation
            if not itr % val_per_iter:
                val_batch_x = val_gen.get_batch(batch_size*8)
                val_loss = sess.run(loss, feed_dict={inp: val_batch_x})
                train_loss_list.append(cur_loss)
                val_loss_list.append(val_loss)
                print(itr, val_loss)
                if best_loss > val_loss:
                    best_at = itr
                    best_loss = val_loss
                    saver.save(sess, '{0}/{1}/{1}_{2}.ckpt'.format(checkpoint_dir, decoder_name, best_at))
                    print('Save best model')

                elif itr >= look_back_period + best_at: 
                    print('Train ends.')
                    saver.restore(sess, '{0}/{1}/{1}_{2}.ckpt'.format(checkpoint_dir, decoder_name, best_at))
                    break

        val_batch_x = val_gen.get_batch(batch_size*8)
        recover_img, l = sess.run([out, loss], feed_dict={inp: val_batch_x})
        print('Val loss is', l)
        
    return train_loss_list, val_loss_list


if __name__ == '__main__':
    train_decoder(args.target_decoder, 
                  train_imgs=args.train_imgs, val_imgs=args.val_imgs, 
                  checkpoint_dir=args.checkpoint_dir)