import sys
sys.path.append("..")
from config import *
import keras
from keras.models import *
from keras.layers import *
from keras.optimizers import SGD
from keras.preprocessing import sequence
from keras.regularizers import l2
from keras import backend as K
from keras.engine.topology import Layer
from keras.backend.tensorflow_backend import set_session
from recurrentshop import *

class Attention(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        """
        Keras Layer that implements an Attention mechanism for temporal data.
        Supports Masking.
        Follows the work of Raffel et al. [https://arxiv.org/abs/1512.08756]
        # Input shape
            3D tensor with shape: `(samples, steps, features)`.
        # Output shape
            2D tensor with shape: `(samples, features)`.
        :param kwargs:
        Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
        The dimensions are inferred based on the output shape of the RNN.
        Example:
            model.add(LSTM(64, return_sequences=True))
            model.add(Attention())
        """
        self.supports_masking = True
        #self.init = initializations.get('glorot_uniform')
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight(
             (input_shape[-1],),
             initializer=self.init,
             name='{}_W'.format(self.name),
             regularizer=self.W_regularizer,
             constraint=self.W_constraint
        )
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight(
                (input_shape[1],),
                initializer='zero',
                name='{}_b'.format(self.name),
                regularizer=self.b_regularizer,
                constraint=self.b_constraint
            )
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim
        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)), K.reshape(self.W, (features_dim, 1))), (-1, step_dim))
        if self.bias:
            eij += self.b
        eij = K.tanh(eij)
        a = K.exp(eij)
        if mask is not None:
            a *= K.cast(mask, K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0],  self.features_dim

class KMaxPooling1D(Layer):
    def __init__(self, k=1, **kwargs):
        self.k = k
        super(KMaxPooling1D, self).__init__(**kwargs)
    def call(self,x):
        x = K.tf.transpose(x, perm=[0,2,1])
        y,_ = K.tf.nn.top_k(x, k=self.k)
        y = K.tf.transpose(y, perm=[0,2,1])
        return y
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.k,input_shape[2])
    def get_config(self):
        config = {"k": self.k}
        base_config = super(KMaxPooling1D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

def get_fasttext(title_seq_length, descr_seq_length, embed_weight):
    title_input = Input(shape=(title_seq_length,), dtype="int32")
    descr_input = Input(shape=(descr_seq_length,), dtype="int32")
    embedding = Embedding(input_dim=embed_weight.shape[0], name="embedding", weights=[embed_weight], output_dim=embed_weight.shape[1], trainable=False)
    trans_title = Activation(activation="relu")(BatchNormalization()((TimeDistributed(Dense(256))(embedding(title_input)))))
    trans_descr = Activation(activation="relu")(BatchNormalization()((TimeDistributed(Dense(256))(embedding(descr_input)))))
    title_pooling = GlobalAveragePooling1D()(trans_title)
    descr_pooling = GlobalAveragePooling1D()(trans_descr)
    feat = concatenate([title_pooling, descr_pooling])
    fc = Activation(activation="relu")(BatchNormalization()(Dense(512)(feat)))
    output = Dense(1999, activation="sigmoid")(fc)
    model = Model(inputs=[title_input, descr_input], outputs=output)
    model.compile(loss='binary_crossentropy', optimizer="adam", metrics=['accuracy'])
    return model

def convs_block(data, convs = [3,4,5], f = 256, name = "conv_feat"):
    pools = []
    for c in convs:
        conv = Activation(activation="relu")(BatchNormalization()(Conv1D(filters=f, kernel_size=c, padding="valid")(data)))
        pool = GlobalMaxPool1D()(conv)
        pools.append(pool)
    return concatenate(pools, name=name)

def get_textcnn(title_seq_length, descr_seq_length, embed_weight):
    title_input = Input(shape=(title_seq_length,),dtype="int32")
    descr_input = Input(shape=(descr_seq_length,),dtype="int32")
    embedding = Embedding(input_dim=embed_weight.shape[0], name="embedding", weights=[embed_weight], output_dim=embed_weight.shape[1], trainable=False)
    trans_title = Activation(activation="relu")(BatchNormalization()((TimeDistributed(Dense(256))(embedding(title_input)))))
    trans_descr = Activation(activation="relu")(BatchNormalization()((TimeDistributed(Dense(256))(embedding(descr_input)))))
    title_feat = convs_block(trans_title)
    descr_feat = convs_block(trans_descr)
    feat = concatenate([title_feat, descr_feat])
    fc = Activation(activation="relu")(BatchNormalization()(Dense(2048)(feat)))
    output = Dense(1999, activation="sigmoid")(fc)
    model = Model(inputs=[title_input, descr_input], outputs=output)
    model.compile(loss='binary_crossentropy', optimizer="adam", metrics=['accuracy'])
    return model

def get_textrnn(title_seq_length, descr_seq_length, embed_weight):
    title_input = Input(shape=(title_seq_length,), dtype="int32")
    descr_input = Input(shape=(descr_seq_length,), dtype="int32")
    embedding = Embedding(input_dim=embed_weight.shape[0], name="embedding", weights=[embed_weight], output_dim=embed_weight.shape[1], trainable=False)
    trans_title = Activation(activation="relu")(BatchNormalization()((TimeDistributed(Dense(256))(embedding(title_input)))))
    trans_descr = Activation(activation="relu")(BatchNormalization()((TimeDistributed(Dense(256))(embedding(descr_input)))))
    title_lstm = GlobalAveragePooling1D()(Bidirectional(GRU(256, return_sequences=True))(trans_title))
    descr_lstm = GlobalAveragePooling1D()(Bidirectional(GRU(256, return_sequences=True))(trans_descr))
    feat = concatenate([title_lstm,descr_lstm])
    output = Dense(1999, activation="sigmoid")(feat)
    model = Model(inputs=[title_input,descr_input], outputs=output)
    model.compile(loss='binary_crossentropy', optimizer="adam", metrics=['accuracy'])
    return model

def get_rcnn(title_seq_length, descr_seq_length, embed_weight):
    title_input = Input(shape=(title_seq_length,), dtype="int32")
    descr_input = Input(shape=(descr_seq_length,), dtype="int32")
    embedding = Embedding(input_dim=embed_weight.shape[0], name="embedding", weights=[embed_weight], output_dim=embed_weight.shape[1], trainable=False)
    trans_title = Activation(activation="relu")(BatchNormalization()((TimeDistributed(Dense(256))(embedding(title_input)))))
    trans_descr = Activation(activation="relu")(BatchNormalization()((TimeDistributed(Dense(256))(embedding(descr_input)))))
    title_left_lstm = LSTM(256,return_sequences=True,implementation=2)
    title_right_lstm = LSTM(256,return_sequences=True,go_backwards=True,implementation=2)
    trans_title_left = BatchNormalization()(title_left_lstm(trans_title))
    trans_title_right = BatchNormalization()(title_right_lstm(trans_title))
    descr_left_lstm = LSTM(256,return_sequences=True,implementation=2)
    descr_right_lstm = LSTM(256,return_sequences=True,go_backwards=True,implementation=2)
    trans_descr_left = BatchNormalization()(descr_left_lstm(trans_descr))
    trans_descr_right = BatchNormalization()(descr_right_lstm(trans_descr))
    title_feat = concatenate([trans_title_left,trans_title,trans_title_right])
    descr_feat = concatenate([trans_descr_left,trans_descr,trans_descr_right])
    title_pool = GlobalAveragePooling1D()(title_feat)
    descr_pool = GlobalAveragePooling1D()(descr_feat)
    feat = Flatten()(concatenate([title_pool,descr_pool]))
    fc = BatchNormalization()(Dense(1024)(feat))
    output = Dense(1999,activation="sigmoid")(fc)
    model = Model(inputs=[title_input, descr_input], outputs=output)    
    model.compile(loss='binary_crossentropy', optimizer="adam", metrics=['accuracy'])
    return model

def kmpconvs_block(data, convs=[3,4,5], f=256, kp=3, name="conv_feat"):
    pools = []
    for c in convs:
        conv = Activation(activation="relu")(BatchNormalization()(Conv1D(filters=f, kernel_size=c, padding="valid")(data)))
        pool = KMaxPooling1D(k=kp)(conv)
        pools.append(pool)
    return concatenate(pools, name=name)

def get_kmptextcnn(title_seq_length, descr_seq_length, embed_weight, kp=3):
    title_input = Input(shape=(title_seq_length,), dtype="int32")
    descr_input = Input(shape=(descr_seq_length,), dtype="int32")
    embedding = Embedding(input_dim=embed_weight.shape[0], name="embedding", weights=[embed_weight], output_dim=embed_weight.shape[1], trainable=False)
    trans_title = Activation(activation="relu")(BatchNormalization()((TimeDistributed(Dense(256))(embedding(title_input)))))
    trans_descr = Activation(activation="relu")(BatchNormalization()((TimeDistributed(Dense(256))(embedding(descr_input)))))
    title_feat = kmpconvs_block(trans_title, kp=kp)
    descr_feat = kmpconvs_block(trans_descr, kp=kp)
    feat = concatenate([title_feat, descr_feat])
    fc = BatchNormalization()(Dense(2048,activation="relu")(feat))
    output = Dense(1999,activation="sigmoid")(fc)
    model = Model(inputs=[title_input,descr_input],outputs=output)
    return model