from keras.layers import Input, Dense, Conv3D, MaxPooling3D, ZeroPadding3D, Flatten, Dropout
from keras.models import Sequential, Model


def create_3D_model():

    model = Sequential()

    input_shape = (3,16,112,112)
    # input_shape = (16,112,112,3)

    # First group layer
    model.add(Conv3D(64, (3,3,3), activation='relu', padding='same', input_shape=input_shape, name='conv1'))
    model.add(MaxPooling3D(pool_size=(1,2,2), strides=(1,2,2), padding='valid', name='pool1'))

    # Second group layer
    model.add(Conv3D(128, (3,3,3), activation='relu', padding='same', name='conv2'))
    model.add(MaxPooling3D(pool_size=(2,2,2), strides=(2,2,2), padding='valid', name='pool2', data_format="channels_first"))

    #Third group layer 
    model.add(Conv3D(256, (3,3,3), activation='relu', padding='same', name='conv3a'))
    model.add(Conv3D(256, (3,3,3), activation='relu', padding='same', name='conv3b'))
    model.add(MaxPooling3D(pool_size=(2,2,2), strides=(2,2,2), padding='valid', name='pool3',data_format="channels_first"))

    # Fourth group layer
    model.add(Conv3D(512, (3,3,3), activation='relu', padding='same', name='conv4a'))
    model.add(Conv3D(512, (3,3,3), activation='relu', padding='same', name='conv4b'))
    model.add(MaxPooling3D(pool_size=(2,2,2), strides=(2,2,2), padding='valid', name='pool4', data_format="channels_first"))

    # Fifth group layer
    model.add(Conv3D(512, (3,3,3), activation='relu', padding='same', name='conv5a'))
    model.add(Conv3D(512, (3,3,3), activation='relu', padding='same', name='conv5b'))
    model.add(ZeroPadding3D(padding=((0,0),(0,1),(0,1)), name='zeropad1'))
    model.add(MaxPooling3D(pool_size=(2,2,2), strides=(2,2,2), padding='valid', name='pool5', data_format="channels_first"))

    # Flatten vector
    model.add(Flatten())

    # Dense group layer 
    model.add(Dense(4096, activation='relu', name='fc6'))
    model.add(Dropout(0.5))
    model.add(Dense(4098, activation='relu', name='fc7'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax', name='fc8'))

    print(model.summary())
    return model 


def extract_features_model(c3d_model, layer_name='fc6'):
    extractor = Model(inputs=c3d_model.input, outputs=c3d_model.get_layer(layer_name).output)
    return extractor