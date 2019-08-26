import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Dropout, Input, LeakyReLU, concatenate
from keras.layers.noise import AlphaDropout
from keras.layers import Layer
from tensorflow.python.keras import backend as K


# Creates 1 branch of the triplet network
def create_base_network(n_dense=6,
                        dense_units=16,
                        activation='selu',
                        dropout=AlphaDropout,
                        dropout_rate=0.1,
                        kernel_initializer='lecun_normal',
                        optimizer='adam',
                        num_classes=1,
                        max_words=978):
    if (activation == "leaky"):
        model = Sequential()
        model.add(Dense(dense_units, input_shape=(max_words,),
                        kernel_initializer=kernel_initializer))
        model.add(LeakyReLU(alpha=0.3))
        model.add(dropout(dropout_rate))

        for i in range(n_dense - 1):
            model.add(Dense(dense_units, kernel_initializer=kernel_initializer))
            model.add(LeakyReLU(alpha=0.3))
            model.add(dropout(dropout_rate))
        return model

    model = Sequential()
    model.add(Dense(dense_units, input_shape=(max_words,),
                    kernel_initializer=kernel_initializer))
    model.add(Activation(activation))
    model.add(dropout(dropout_rate))

    for i in range(n_dense - 1):
        model.add(Dense(dense_units, kernel_initializer=kernel_initializer))
        model.add(Activation(activation))
        model.add(dropout(dropout_rate))
    return model


# triplet loss function
def triplet_loss(y_true, y_pred, alpha=0.6):
    """
    Implementation of the triplet loss function
    Arguments:
    y_true -- true labels, required when you define a loss in Keras, you don't need it in this function.
    y_pred -- python list containing three objects:
            anchor -- the encodings for the anchor data
            positive -- the encodings for the positive data (similar to anchor)
            negative -- the encodings for the negative data (different from anchor)
    Returns:
    loss -- real number, value of the loss
    """
    print('y_pred.shape = ', y_pred)

    total_lenght = y_pred.shape.as_list()[-1]
    #     print('total_lenght=',  total_lenght)
    #     total_lenght =12

    anchor = y_pred[:, 0:int(total_lenght * 1 / 3)]
    positive = y_pred[:, int(total_lenght * 1 / 3):int(total_lenght * 2 / 3)]
    negative = y_pred[:, int(total_lenght * 2 / 3):int(total_lenght * 3 / 3)]

    # distance between the anchor and the positive
    pos_dist = K.sum(K.square(anchor - positive), axis=1)

    # distance between the anchor and the negative
    neg_dist = K.sum(K.square(anchor - negative), axis=1)

    # compute loss
    basic_loss = pos_dist - neg_dist + alpha
    loss = K.maximum(basic_loss, 0.0)

    return loss


# Creating full fledged triplet network with 3 inputs
def create_siamese_network(shared_model, input_dim=978):
    anchor_input = Input((input_dim,), name='anchor_input')
    positive_input = Input((input_dim,), name='positive_input')
    negative_input = Input((input_dim,), name='negative_input')

    encoded_anchor = shared_model(anchor_input)
    encoded_positive = shared_model(positive_input)
    encoded_negative = shared_model(negative_input)

    merged_vector = concatenate([encoded_anchor, encoded_positive, encoded_negative], axis=-1, name='merged_layer')

    model = Model(inputs=[anchor_input, positive_input, negative_input], outputs=merged_vector)
    model.compile(loss=triplet_loss, optimizer='adam')

    return model
