from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input, Lambda, Flatten, Conv2D, MaxPooling2D, BatchNormalization, ZeroPadding2D
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import backend as K
from tensorflow.keras.regularizers import l2
from tensorflow.keras.saving import register_keras_serializable

# ✅ Register the Euclidean distance as a global function
@register_keras_serializable()
def euclidean_distance(vectors):
    """
    Computes the Euclidean distance between two feature vectors.
    """
    x, y = vectors
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))

# ✅ Define Triplet Loss
@register_keras_serializable()
def triplet_loss(y_true, y_pred, alpha=0.2):
    """
    Computes the Triplet Loss.
    """
    anchor, positive, negative = y_pred[:, 0], y_pred[:, 1], y_pred[:, 2]
    pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=-1)
    neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=-1)
    loss = tf.reduce_mean(tf.maximum(pos_dist - neg_dist + alpha, 0.0))
    return loss

# ✅ Define the SigNet Base Network Architecture
def create_base_network_signet(input_shape):
    """
    Builds the base convolutional neural network used in SigNet.
    """
    seq = Sequential()

    # First convolutional block
    seq.add(Conv2D(96, (11, 11), activation='relu', strides=(4, 4), 
                   input_shape=input_shape, kernel_initializer='glorot_uniform'))
    seq.add(BatchNormalization())
    seq.add(MaxPooling2D((3, 3), strides=(2, 2)))
    seq.add(ZeroPadding2D((2, 2)))

    # Second convolutional block
    seq.add(Conv2D(256, (5, 5), activation='relu', kernel_initializer='glorot_uniform'))
    seq.add(BatchNormalization())
    seq.add(MaxPooling2D((3, 3), strides=(2, 2)))
    seq.add(Dropout(0.3))
    seq.add(ZeroPadding2D((1, 1)))

    # Third convolutional block
    seq.add(Conv2D(384, (3, 3), activation='relu', kernel_initializer='glorot_uniform'))
    seq.add(ZeroPadding2D((1, 1)))

    # Fourth convolutional block
    seq.add(Conv2D(256, (3, 3), activation='relu', kernel_initializer='glorot_uniform'))
    seq.add(MaxPooling2D((3, 3), strides=(2, 2)))
    seq.add(Dropout(0.3))

    # Fully connected layers
    seq.add(Flatten())
    seq.add(Dense(1024, activation='relu', kernel_initializer='glorot_uniform', kernel_regularizer=l2(0.0005)))
    seq.add(Dropout(0.5))
    seq.add(Dense(128, activation='relu', kernel_initializer='glorot_uniform', kernel_regularizer=l2(0.0005)))

    return seq

# ✅ Define the Triplet Network
def create_triplet_network(input_shape):
    """
    Builds the Triplet Neural Network for signature verification.
    """
    base_network = create_base_network_signet(input_shape)

    input_anchor = Input(shape=input_shape, name='input_anchor')
    input_positive = Input(shape=input_shape, name='input_positive')
    input_negative = Input(shape=input_shape, name='input_negative')

    encoded_anchor = base_network(input_anchor)
    encoded_positive = base_network(input_positive)
    encoded_negative = base_network(input_negative)

    # Concatenate the encoded vectors for triplet loss computation
    triplet_output = Lambda(lambda x: K.concatenate([x[0], x[1], x[2]], axis=-1))([encoded_anchor, encoded_positive, encoded_negative])

    model = Model(inputs=[input_anchor, input_positive, input_negative], outputs=triplet_output)
    return model