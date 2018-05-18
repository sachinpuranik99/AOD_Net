from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense, Lambda
from keras import backend as K
import cv2, numpy as np
import glob
from keras.activations import relu 
import keras as keras
from keras.models import Model


from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras import backend as K
img_width, img_height = 600, 400


def get_image_array(path):
    #print path
    return np.asarray([cv2.resize(cv2.imread(file), (img_height,
        img_width)).astype(np.float32) for file in glob.glob(path
            +'/'+'*.png')])


def get_unet():
    inputs = Input(batch_shape=(None,img_width, img_height, 3))
    conv1 = Conv2D(3, (1, 1), activation='relu')(inputs)

    conv2 = Conv2D(3, (3, 3), activation='relu', padding='same')(conv1)

    concat1 = concatenate([conv1, conv2], axis=0)

    conv3 = Conv2D(3, (5, 5), activation='relu', padding='same')(concat1)

    concat2 = concatenate([conv2, conv3], axis=0)

    conv4 = Conv2D(3, (7, 7), activation='relu', padding='same')(concat2)

    concat3 = concatenate([conv1, conv2, conv3, conv4], axis=0)

    conv5 = Conv2D(3, (3, 3), activation='relu', padding='same')(concat3)

    print inputs.shape,conv5
    product= keras.layers.Multiply()([inputs, conv5])
    sum1 = keras.layers.Subtract()([inputs, product])
    #input2 = Input(sum1.shape)
    sum2 = keras.layers.Add()([sum1, sum1])
    out_layer = Lambda(lambda x: relu(x)) (sum2)
    #out_layer = relu(sum2)#



    model = Model(inputs,out_layer)

    return model



if __name__ == '__main__':
    train_data_dir = 'train_data/haze'
    train_label_dir = 'train_data/clear'
    test_data_dir = 'test_data/haze'
    test_label_dir = 'test_data/clear'
    nb_train_samples = 90
    nb_validation_samples = 10
    epochs = 50
    batch_size = 1 
    
    if K.image_data_format() == 'channels_first':
        input_shape = (3, img_width, img_height)
    else:
        input_shape = (img_width, img_height, 3)
    
    X_train, Y_train = get_image_array(train_data_dir), get_image_array(train_label_dir)
    X_test, Y_test = get_image_array(test_data_dir), get_image_array(test_label_dir)
    print(X_train[0].shape)
    print(Y_train[0].shape)
    print(X_test[0].shape)
    print(Y_test[0].shape)
    
    
    model_checkpoint = ModelCheckpoint('weights.h5', monitor='val_loss', save_best_only=True)
    
    print('-'*30)
    print('Fitting model...')
    print('-'*30)

    model = get_unet() 

    print(model.summary())
    model.compile(loss='mean_squared_error',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=1, verbose=1, callbacks=[model_checkpoint])
    
    print('-'*30)
    print('Loading saved weights...')
    print('-'*30)
    #model.load_weights('weights.h5')
    
    print X_test
    imgs_mask_test = model.predict(X_test, batch_size = batch_size, verbose=1)
    #np.save('imgs_mask_test.npy', imgs_mask_test)
    #
    #print('-' * 30)
    #print('Saving predicted masks to files...')
    #print('-' * 30)
    #pred_dir = 'preds'
    #if not os.path.exists(pred_dir):
    #    os.mkdir(pred_dir)
    #for image, image_id in zip(imgs_mask_test, X_test):
    #    image = (image[:, :, 0] * 255.).astype(np.uint8)
    #    imsave(os.path.join(pred_dir, str(image_id) + '_pred.png'), image)


"""
#model = Sequential()
#model.add(Conv2D(32, (3, 3), input_shape=input_shape))
#model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#
#model.add(Conv2D(32, (3, 3)))
#model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#
#model.add(Conv2D(64, (3, 3)))
#model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#
#model.add(Flatten())
#model.add(Dense(64))
#model.add(Activation('relu'))
#model.add(Dropout(0.5))
#model.add(Dense(1))
#model.add(Activation('sigmoid'))
#
#model.compile(loss='binary_crossentropy',
#              optimizer='rmsprop',
#              metrics=['accuracy'])
#
## this is the augmentation configuration we will use for training
#train_datagen = ImageDataGenerator()
#
## this is the augmentation configuration we will use for testing:
## only rescaling
#test_datagen = ImageDataGenerator()
#
#train_generator = train_datagen.flow_from_directory(
#    train_data_dir,
#    target_size=(img_width, img_height),
#    batch_size=batch_size,
#    class_mode='input')
#
#validation_generator = test_datagen.flow_from_directory(
#    validation_data_dir,
#    target_size=(img_width, img_height),
#    batch_size=batch_size,
#    class_mode='input')
#
#model.fit_generator(
#    train_generator,
#    steps_per_epoch=nb_train_samples // batch_size,
#    epochs=epochs,
#    validation_data=validation_generator,
#    validation_steps=nb_validation_samples // batch_size)
#
#model.save_weights('first_try.h5')
#up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
#conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
#conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

#up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
#conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
#conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

#up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
#conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
#conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

#up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
#conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
#conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

#conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)


    ##from keras.layers import Input, Dense
    ##from keras.models import Model
    ##
    ### This returns a tensor
    ##inputs = Input(batch_shape=(None,  img_width,img_height, 3))
    
    # a layer instance is callable on a tensor, and returns a tensor
    ##x = Dense(64, activation='relu')(inputs)
    ##x = Dense(64, activation='relu')(x)
    ##predictions = Dense(3, activation='softmax')(x)
    ##
    ### This creates a model that includes
    ### the Input layer and three Dense layers
    ##model = Model(inputs=inputs, outputs=predictions)
    ##model.compile(optimizer='rmsprop',
    ##              loss='categorical_crossentropy',
    ##              metrics=['accuracy'])
    ##print(model.summary())
    ##print(Y_train.shape)
    ##model.fit(X_train, Y_train)  # starts training

"""

