import torchfile
import tensorflow as tf
import numpy as np
import os
import urllib.request


class Encoder_decoder:
    """
    This class can read (or download, if not exists) VGG19 pre-trained model, then generates encoder and decoder bundle for target layers.
    """
    def __init__(self, model_file='models/vgg_normalised.t7'):
        exists = os.path.isfile(model_file)
        if not exists:
            Encoder_decoder.download_file()
        try:
            self.model = torchfile.load('models/vgg_normalised.t7')
        except:
            raise ValueError('Fail to load t7 model')
    
    @staticmethod
    def download_file():
        print('Vgg model does not exist, downloading...')
        vgg_url = "https://www.dropbox.com/s/kh8izr3fkvhitfn/vgg_normalised.t7?dl=1"
        urllib.request.urlretrieve(vgg_url, './models/vgg_normalised.t7')
        print('Vgg model downloaded. ')
    
    @staticmethod
    def reflect_padding(x,padding=1,name=None):
        return tf.pad(x,[[0,0],[padding,padding],[padding,padding],[0,0]], mode='REFLECT', name=name)    
    
    @staticmethod
    def max_pool(x,pool_size=2,strides=2,name=None):
        return tf.layers.max_pooling2d(x, pool_size=pool_size, strides=strides, name=name)
    
    @staticmethod
    def conv_layer(x,filters,kernel_size=3,kernel=None,bias=None,trainable=True,name=None):
        kernel_initializer = tf.constant_initializer(kernel) if kernel is not None else None
        bias_initializer = tf.constant_initializer(bias) if bias is not None else None

        return tf.layers.conv2d(x, filters, kernel_size=kernel_size, 
                         kernel_initializer=kernel_initializer, 
                         bias_initializer=bias_initializer,
                         trainable=trainable,name=name,reuse=tf.AUTO_REUSE)
    
    def encode(self, x, target_layer, name_scope=''):
        with tf.variable_scope(name_scope):
            for i, m in enumerate(self.model.modules):
                name = m.name.decode() if m.name is not None else None

                if i == 0:
                    name = 'preprocess'

                if m._typename == b'nn.SpatialReflectionPadding':
                    x = Encoder_decoder.reflect_padding(x, name=name)

                elif m._typename == b'nn.SpatialConvolution':
                    w = m.weight.transpose([2,3,1,0])
                    b = m.bias
                    assert x.shape[-1] == w.shape[2]
                    x = Encoder_decoder.conv_layer(x, w.shape[-1], kernel_size=w.shape[0:2], 
                                                   kernel=w, bias=b, trainable=False, name=name)

                elif m._typename == b'nn.ReLU':
                    x = tf.nn.relu(x, name=name)

                elif m._typename == b'nn.SpatialMaxPooling':
                    x = Encoder_decoder.max_pool(x, name=name)

                if name == target_layer:
                    return x, i
        raise ValueError("Layer {} is not in the model".format(target_layer))
        
    def decode(self, x, start_layer, name_scope=''):       
        with tf.variable_scope(name_scope):
            for i, m in enumerate(self.model.modules[start_layer::-1]):
                name = m.name.decode() if m.name is not None else None 

                if i == start_layer:
                    name = 'postprocess'
                    continue

                if m._typename == b'nn.SpatialReflectionPadding':
                    pass

                elif m._typename == b'nn.SpatialConvolution':            
                    w = m.weight.transpose([2,3,1,0])
                    assert x.shape[-1] == w.shape[-1]
                    x = Encoder_decoder.reflect_padding(x)
                    x = Encoder_decoder.conv_layer(x, w.shape[2], kernel_size=w.shape[0:2], name=name)
                    if i != start_layer-2:
                        x = tf.nn.relu(x)

                elif m._typename == b'nn.ReLU':
                    pass

                elif m._typename == b'nn.SpatialMaxPooling':
                    x = tf.image.resize_nearest_neighbor(x, tf.shape(x)[1:3]*2, name=name)
            return x
        
    def get_encoder_decoder(self, target_layers, input_shape=[None, None, None, 3]):
        """
        Generate encoders and decoders for target layers
         
        @target_layers: list of target layers in VGG 19 model
        @return: dictionary of (input, encoded, decoded, second_encoded)
        """
        encoder_decoder = {}
        
        inp = tf.placeholder(tf.float32, shape=input_shape, name='vgg_input')         
        for target_layer in set(target_layers):
            encoded, end_layer = self.encode(inp, target_layer, name_scope='encoder_{}'.format(target_layer))
            decoded = self.decode(encoded, start_layer=end_layer, name_scope='decoder_{}'.format(target_layer))
            second_encoded, _ = self.encode(decoded, target_layer, 
                                            name_scope='decoder_{}'.format(target_layer)+'_second_encoded')
    
            encoder_decoder[target_layer] = (inp, encoded, decoded, second_encoded)

        return encoder_decoder
