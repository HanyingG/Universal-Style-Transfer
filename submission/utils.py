import tensorflow as tf
import numpy as np
import os
from imageio import imread, imwrite
import skimage

    
def get_files(img_dir):
    files = os.listdir(img_dir)
    paths = []
    for x in files:
        paths.append(os.path.join(img_dir, x))
    return paths

def get_img(src):
    img = imread(src, pilmode='RGB')
    if not (len(img.shape) == 3 and img.shape[2] == 3):
        img = np.dstack((img,img,img))
    return img/255

# crop the image
def random_img_crop(img, size=256,crop=256):
    height, width = img.shape[0], img.shape[1]

    if height < size or width < size: 
        img = resize_to(img, resize=size)
        height, width = img.shape[0], img.shape[1]

    h_cut = np.random.randint(0, (img.shape[0]-crop)+1)
    w_cut = np.random.randint(0, (img.shape[1]-crop)+1)
    
    return img[h_cut:(h_cut+crop),w_cut:(w_cut+crop)]

def resize_to(img, resize=256):
    height, width = img.shape[0], img.shape[1]
    if height < width:
        ratio = height / resize
        long_side = round(width / ratio)
        resize_shape = (resize, long_side, 3)
    else:
        ratio = width / resize
        long_side = round(height / ratio)
        resize_shape = (long_side, resize, 3)
    
    return skimage.transform.resize(img, resize_shape, order=1,preserve_range=True)

def get_cropped_image(src, size=256, crop=256):
    return random_img_crop(get_img(src), size, crop)
    
def save_image(img, path):
    img[img < 0] = 0
    img[img > 1] = 1
    imwrite(path, img)
    return None
    
def WCT(content,style,alpha,eps=1e-8):
    
    # exp: content/style: (n,224,224,3) -> (3,224,224)
    content_t = tf.transpose(tf.squeeze(content),(2,0,1))
    style_t = tf.transpose(tf.squeeze(style),(2,0,1))
    
    C_c, H_c, W_c = tf.unstack(tf.shape(content_t))
    C_s, H_s, W_s = tf.unstack(tf.shape(style_t))
    
    content_f = tf.reshape(content_t,(C_c,H_c*W_c))
    style_f = tf.reshape(style_t,(C_s,H_s*W_s))
    
    mean_c = tf.reduce_mean(content_f, axis=1 ,keepdims=True)
    fc = content_f - mean_c
    cov_c = tf.matmul(fc, fc,transpose_b=True)/(tf.cast(H_c*W_c,tf.float32)-1) + tf.eye(C_c) * eps
    
    mean_s = tf.reduce_mean(style_f, axis=1 ,keepdims=True)
    fs = style_f - mean_s
    cov_s = tf.matmul(fs, fs,transpose_b=True)/(tf.cast(H_s*W_s,tf.float32)-1) + tf.eye(C_s) * eps

    with tf.device('/cpu:0'):
        S_c, U_c, _ = tf.svd(cov_c)
        S_s, U_s, _ = tf.svd(cov_s)
    
    # filter small singular values
    k_c = tf.reduce_sum(tf.cast(tf.greater(S_c,1e-5),tf.int64))
    k_s = tf.reduce_sum(tf.cast(tf.greater(S_s,1e-5),tf.int64))
    
    D_c = tf.diag(tf.pow(S_c[:k_c],-0.5))
    fc_hat = tf.matmul(tf.matmul(tf.matmul(U_c[:,:k_c],D_c),U_c[:,:k_c],transpose_b=True),fc)
    
    D_s = tf.diag(tf.pow(S_s[:k_s],0.5))
    fcs_hat = tf.matmul(tf.matmul(tf.matmul(U_s[:,:k_s],D_s),U_s[:,:k_s],transpose_b=True),fc_hat)
    
    #re-center
    fcs_hat = fcs_hat + mean_s
    fcs_hat = alpha*fcs_hat + (1-alpha)*content_f
    
    blended = tf.reshape(fcs_hat,(C_c,H_c,W_c))
    blended = tf.expand_dims(tf.transpose(blended, (1,2,0)), 0)
    return blended