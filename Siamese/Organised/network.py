import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Dropout, Input, LeakyReLU
from keras.layers.noise import AlphaDropout
from keras.layers import Layer
from tensorflow.python.keras import backend as K


# Creates 1 branch of the siamese network
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

        for i in range(n_dense - 2):
            model.add(Dense(dense_units, kernel_initializer=kernel_initializer))
            model.add(LeakyReLU(alpha=0.3))
            model.add(dropout(dropout_rate))

        model.add(Dense(dense_units, kernel_initializer=kernel_initializer))
        model.add(Activation('selu'))
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


# Distance layer to calculate the distance between the 2 embeddings
class ManDist(Layer):

    def __init__(self, **kwargs):
        self.result = None
        super(ManDist, self).__init__(**kwargs)

    def build(self, input_shape):
        super(ManDist, self).build(input_shape)

    def call(self, x, **kwargs):
        self.result = K.sum(K.abs(x[0] - x[1]), axis=1, keepdims=True)
        return self.result

    def compute_output_shape(self, input_shape):
        return K.int_shape(self.result)


# contrastive loss function
def contrastive_loss(y_true, y_pred):
    margin = 1
    return K.mean(y_true * K.square(y_pred) +
                  (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))


# Creating full fledged siamese network with 2 inputs
def create_siamese_network(shared_model, input_dim=978):
    left_input = Input(shape=(input_dim,))
    right_input = Input(shape=(input_dim,))

    malstm_distance = ManDist()([shared_model(left_input), shared_model(right_input)])

    model = Model(inputs=[left_input, right_input], outputs=[malstm_distance])
    model.compile(loss=contrastive_loss, optimizer="adam", metrics=['accuracy'])

    return model
