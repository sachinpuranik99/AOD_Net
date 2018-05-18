"""
Copyright 2017-2018 Fizyr (https://fizyr.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import warnings

import keras
import keras_resnet
import keras_resnet.models
from . import retinanet
import tensorflow as tf

# RGSL : AOD - Start
from keras.activations import relu 
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose
from keras.layers import Activation, Dropout, Flatten, Dense, Lambda
import tensorflow as tf
import scipy.misc
# RGSL : AOD - End

resnet_filename = 'ResNet-{}-model.keras.h5'
resnet_resource = 'https://github.com/fizyr/keras-models/releases/download/v0.0.1/{}'.format(resnet_filename)

custom_objects = retinanet.custom_objects.copy()
custom_objects.update(keras_resnet.custom_objects)

allowed_backbones = ['resnet50', 'resnet101', 'resnet152']


def download_imagenet(backbone):
    validate_backbone(backbone)

    backbone = int(backbone.replace('resnet', ''))

    filename = resnet_filename.format(backbone)
    resource = resnet_resource.format(backbone)
    if backbone == 50:
        checksum = '3e9f4e4f77bbe2c9bec13b53ee1c2319'
    elif backbone == 101:
        checksum = '05dc86924389e5b401a9ea0348a3213c'
    elif backbone == 152:
        checksum = '6ee11ef2b135592f8031058820bb9e71'

    return keras.applications.imagenet_utils.get_file(
        filename,
        resource,
        cache_subdir='models',
        md5_hash=checksum
    )


def validate_backbone(backbone):
    if backbone not in allowed_backbones:
        raise ValueError('Backbone (\'{}\') not in allowed backbones ({}).'.format(backbone, allowed_backbones))


# RGSL : AOD_Net Changes - Start
class AODCustomLayer(keras.layers.Layer):
    def call (self, inputs):
        print (inputs[0].shape)
        #scipy.misc.toimage(inputs[0]).save('dehazed.png')
        return inputs[:,:,:,::-1]   

class AODCustomLayer_1(keras.layers.Layer):
    def call (self, inputs):
        
        out0 = Lambda(lambda x: x - 103.939) (inputs[..., 0])
        print ("Outs", out0.shape)
        out1 = Lambda(lambda x: x - 116.779) (inputs[..., 1])
        print (out1.shape)
        out2 = Lambda(lambda x: x - 123.68) (inputs[..., 2])
        print (out2.shape)
        aod_out_layer = concatenate([out0, out1, out2], axis=-2)
        print ("concate layer", aod_out_layer)
        print ("Cacate shape", aod_out_layer.shape)
        #scipy.misc.toimage(aod_out_layer).save('preprocessed.png')
        aod_out_layer = tf.keras.backend.expand_dims(aod_out_layer, 0)
        return aod_out_layer 
# RGSL : AOD_Net Changes - End

def resnet_retinanet(num_classes, backbone='resnet50', inputs=None, modifier=None, **kwargs):
    validate_backbone(backbone)

    # choose default input
    if inputs is None:
        inputs = keras.layers.Input(shape=(None, None, 3))

    # create the resnet backbone
    if backbone == 'resnet50':
        resnet = keras_resnet.models.ResNet50(inputs, include_top=False, freeze_bn=True)
    elif backbone == 'resnet101':
        resnet = keras_resnet.models.ResNet101(inputs, include_top=False, freeze_bn=True)
    elif backbone == 'resnet152':
        resnet = keras_resnet.models.ResNet152(inputs, include_top=False, freeze_bn=True)

    # RGSL : AOD_Net Changes - Start
    aod_inputs = Input(shape=(None,None, 3))
    conv1 = Conv2D(3, (1, 1), kernel_initializer='random_normal', activation='relu')(aod_inputs)

    conv2 = Conv2D(3, (3, 3), kernel_initializer='random_normal', activation='relu',  padding='same')(conv1)

    concat1 = concatenate([conv1, conv2], axis=-1)

    print ("Shape of the concat", concat1.shape)

    conv3 = Conv2D(3, (5, 5), activation='relu', kernel_initializer='truncated_normal', padding='same')(concat1)

    concat2 = concatenate([conv2, conv3], axis=-1)

    conv4 = Conv2D(3, (7, 7), activation='relu', kernel_initializer='random_normal', padding='same')(concat2)

    concat3 = concatenate([conv1, conv2, conv3, conv4], axis=-1)

    print ("Just before concat")
    K = Conv2D(3, (3, 3), activation='relu', kernel_initializer='truncated_normal', padding='same')(concat3)

    product= keras.layers.Multiply()([K, aod_inputs])
    sum1 = keras.layers.Subtract()([product, K])
    sum2 = Lambda(lambda x: 1+x) (sum1)
    #sum2 = keras.layers.Add()([sum1, ones_tensor])
    aod_out_layer = Lambda(lambda x: relu(x)) (sum2)
    print ("RGSL : After relu ", aod_out_layer._keras_history)

    #aod_out_layer = aod_out_layer[:, :, :, -1]
    print ("RGSL shape", aod_out_layer.shape)
    print ("The dtype :", aod_out_layer.dtype)
    print (aod_out_layer[0].shape)
    aod_model_1 = keras.models.Model(inputs=aod_inputs, outputs=aod_out_layer) 

    
   # print ("RGSL : After reversing ", aod_out_layer._keras_history)
    #aod_model = keras.models.Model(inputs=aod_inputs, outputs=aod_out_layer)
    #print ("Created the model after reversing") 
   # print ("RGSL shape", aod_out_layer.shape)
    aod_out_layer = Lambda(lambda x: x * 255) (aod_out_layer)
    aod_out_layer = AODCustomLayer() (aod_out_layer)
    print ("RGSL after cutom layer", aod_out_layer.shape)
    aod_out_layer = AODCustomLayer_1() (aod_out_layer)
    print ("RGSL : aod output layer:", aod_out_layer)
    #print ("Created model before mul")
    #aod_model = keras.models.Model(inputs=aod_inputs, outputs=aod_out_layer)
    #print ("Created model after mul")
    #print ("Creating the model")
    ##aod_model = Model(inputs=aod_inputs, outputs=aod_out_layer)
    #print ("Created the model")
    # RGSL : AOD_Net Changes - End
    #print ("The dtype :", aod_out_layer.dtype)

    # invoke modifier if given
    if modifier:
        resnet = modifier(resnet)
   
    #print (resnet.summary())
    print ("Creating the model")
    print ("RGSL : After concat ", aod_out_layer._keras_history)
    aod_model = keras.models.Model(inputs=aod_inputs, outputs=aod_out_layer) 
    print ("Created the model")
    print (aod_model.outputs)
#    aod_model = Model(inputs=aod_inputs, outputs = aod_out_layer)

    aod_model.load_weights ('./snapshots/aod_net.h5')

    print ("resnet o/p", resnet.output)
    resnet = keras.models.Model(inputs=aod_inputs, outputs = resnet(aod_model.outputs)) 

    print ("after changes resnet o/p", resnet.outputs)
    #print (aod_model.summary())
    #print (resnet.summary())

    # create the full model
    return retinanet.retinanet(inputs=resnet.inputs, num_classes=num_classes, backbone_layers=resnet.outputs[1:], **kwargs)


def resnet50_retinanet(num_classes, inputs=None, **kwargs):
    return resnet_retinanet(num_classes=num_classes, backbone='resnet50', inputs=inputs, **kwargs)


def resnet101_retinanet(num_classes, inputs=None, **kwargs):
    return resnet_retinanet(num_classes=num_classes, backbone='resnet101', inputs=inputs, **kwargs)


def resnet152_retinanet(num_classes, inputs=None, **kwargs):
    return resnet_retinanet(num_classes=num_classes, backbone='resnet152', inputs=inputs, **kwargs)


def ResNet50RetinaNet(inputs, num_classes, **kwargs):
    warnings.warn("ResNet50RetinaNet is replaced by resnet50_retinanet and will be removed in a future release.")
    return resnet50_retinanet(num_classes, inputs, *args, **kwargs)


def ResNet101RetinaNet(inputs, num_classes, **kwargs):
    warnings.warn("ResNet101RetinaNet is replaced by resnet101_retinanet and will be removed in a future release.")
    return resnet101_retinanet(num_classes, inputs, *args, **kwargs)


def ResNet152RetinaNet(inputs, num_classes, **kwargs):
    warnings.warn("ResNet152RetinaNet is replaced by resnet152_retinanet and will be removed in a future release.")
    return resnet152_retinanet(num_classes, inputs, *args, **kwargs)
