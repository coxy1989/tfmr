import numpy as np
import keras.backend as K
import keras
import math
from keras.engine import Layer
from keras.layers import Dropout, Add

class TransformerDecoderBlock:
    '''TODO: docstring'''

    def __init__(self, name:str, dropout_1=0.1, num_heads=3):
        '''TODO: docstring'''
        self.self_attention_layer = MultiHeadSelfAttention(num_heads=num_heads)
        self.dropout_1 = Dropout(dropout_1, name=f'{name}_dropout_1')
        self.add_1 = Add(name=f'{name}_add_1')
        self.norm_1 = LayerNormalization()
        self.feed_fwd = TransformerFeedForward()

    def __call__(self, input):
        '''TODO: docstring'''
        out = self.self_attention_layer(input)
        out = self.dropout_1(out)
        out = self.add_1([out, input])
        out = self.norm_1(out)
        out = self.feed_fwd(out)
        return out

class PositionalEncoding(Layer):
    '''TODO: docstring'''

    @staticmethod
    def positional_encoding(seq_len, d_model):
        '''TODO: docstring'''
        ret = np.zeros([seq_len, d_model])
        for pos in range(seq_len):
            for i in range(0, d_model, 2):
                ret[pos, i] = math.sin(pos/(1e4**((2 * i)/np.float(d_model))))
                ret[pos, i + 1] = math.cos(pos/(1e4**((2 * i + 1)/np.float(d_model))))
        return ret[None, ...]

    def build(self, input_shape):
        '''TODO: docstring'''
        _, length, hidden_size = input_shape
        self.signal = PositionalEncoding.positional_encoding(length, hidden_size)
        return super().build(input_shape)

    def call(self, inputs, **kwargs):
        '''TODO: docstring'''
        return inputs + self.signal

class MultiHeadSelfAttention(Layer):
    '''TODO: docstring'''

    def __init__(self, num_heads:int, **kwargs):
        '''TODO: docstring'''
        self.num_heads = num_heads
        super().__init__(**kwargs)

    def build(self, input_shape):
        '''TODO: docstring'''
        d_model = input_shape[-1]
        if d_model % self.num_heads != 0:
            raise ValueError(f'The dimension of the model:{d_model} is not evenly divisible by the number of heads: {self.num_heads}')
        self.qkv_weights = self.add_weight(
                name='qkv_weights',
                shape= (d_model, d_model * 3),
                initializer='glorot_uniform',
                trainable=True)
        self.output_weights = self.add_weight(
                name='output_weights',
                shape=(d_model, d_model),
                initializer='glorot_uniform',
                trainable=True)
        return super().build(input_shape)

    def call(self, inputs, **kwargs):
        '''TODO: docstring'''
        _, seq_len, d_model = K.int_shape(inputs)
        # `inputs_2d` shape: (batch_size * seq_len, d_model)
        inputs_2d = K.reshape(inputs, [-1, d_model])
        # apply affine transformation to yield queries, keys and values
        # `qkv` shape: (batch_size * seq_len, d_model * 3)
        qkv = K.dot(inputs_2d, self.qkv_weights)
        # `q`,`k`,`v` shape: (batch_size * seq_len, d_model)
        q, k, v = [qkv[:, i * d_model: (i + 1) * d_model] for i in range(3)]
        #`q`,`k`,`v` shape: (batch_size, seq_len , num_heads , d_model // num_heads)
        q, k, v = [K.reshape(m, [-1, seq_len, self.num_heads, d_model // self.num_heads]) for m in [q,k,v]]
        return self.attention(q,k,v,d_model)

    def mask(self, q_dot_k):
        '''
        :param q_dot_k: (batch_size * num_heads, seq_len, seq_len)
        '''
        last_dims = K.int_shape(q_dot_k)[-2:]
        # low_triangular_ones shape: (1, seq_len, seq_len)
        low_triangular_ones = np.tril(np.ones(last_dims))[None,...]
        inv_low_triangular_ones = 1 - low_triangular_ones
        upper_right = K.constant(-1e9 * inv_low_triangular_ones)
        lower_left = K.constant(low_triangular_ones) * q_dot_k
        return upper_right + lower_left

    def attention(self, q, k, v, d_model, masked=True):
        '''
        :param q: (batch_size, seq_len, num_heads, d_model // num_heads)
        :param k: (batch_size, seq_len, num_heads, d_model // num_heads)
        :param v: (batch_size, seq_len, num_heads, d_model // num_heads)
        '''
        batch_size, q_seq_len, num_heads, quotient = K.int_shape(q)
        _, k_seq_len, _, _ = K.int_shape(k)

        # 1. Calculate scaled dot products of q and k
        # ------------------------------------------
        # reshape q and k to make them ameanable to the k and q dot product
        # dot_q shape: (batch_size, num_heads, q_seq_len, d_model // num_heads)
        dot_q = K.permute_dimensions(q, [0,2,1,3])
        # dot_q shape: (batch_size * num_heads, q_seq_len, d_model // num_heads)
        dot_q = K.reshape(dot_q, [-1, q_seq_len, quotient])
        # dot_k shape: (batch_size, num_heads, d_model // num_heads, k_seq_len)
        dot_k = K.permute_dimensions(k, [0,2,3,1])
        # dot_k shape: (batch_size * num_heads, d_model // num_heads, k_seq_len)
        dot_k = K.reshape(dot_k, [-1, quotient, k_seq_len])
        # q_dot_k shape: (batch_size * num_heads, q_seq_len, k_seq_len)
        q_dot_k = K.batch_dot(dot_q, dot_k)
        # scaling
        sqrt_d = K.constant(np.sqrt(d_model // self.num_heads))
        scaled_q_dot_k = q_dot_k / sqrt_d
        # masking
        scaled_q_dot_k = self.mask(q_dot_k) if masked else scaled_q_dot_k
        # softmax
        scaled_q_dot_k = K.softmax(scaled_q_dot_k)

        #2. Calculate dot product of scaled_q_dot_k with v
        #-----------------------------------------
        # reshape v to make it ameanable to dotting
        # dot_v shape: (batch_size, num_heads, k_seq_len, d_model // num_heads)
        dot_v = K.permute_dimensions(v, [0,2,1,3])
        # dot_v shape: (batch_size * num_heads, k_seq_len, d_model // num_heads)
        dot_v = K.reshape(dot_v, [-1, k_seq_len, quotient])
        # attention_heads shape: batch_size * num_heads, q_seq_len, d_model // num_heads)
        attention_heads = K.batch_dot(scaled_q_dot_k, dot_v)

        #3. Merge attention heads and apply output_weights
        #--------------------------------------------------
        # attention_heads shape: batch_size, num_heads, q_seq_len, d_model // num_heads)
        attention_heads = K.reshape(attention_heads, [-1, num_heads, q_seq_len, quotient])
        # attention_heads shape: batch_size, q_seq_len, num_heads, d_model // num_heads)
        attention_heads = K.permute_dimensions(attention_heads, [0,2,1,3])
        # attention_heads_merged shape: batch_size * q_seq_len, d_model
        attention_heads_merged = K.reshape(attention_heads, [-1, d_model])
        # result shape: batch_size * q_seq_len, d_model
        result = K.dot(attention_heads_merged, self.output_weights)
        # result shape: batch_size, q_seq_len, d_model
        result = K.reshape(result, [-1, q_seq_len, d_model])
        return result

class TransformerFeedForward(Layer):

    def __init__(self, size_multiplier:int = 4, **kwargs):
        '''TODO: docstring'''
        self.size_multiplier = size_multiplier
        super().__init__(**kwargs)

    def build(self, input_shape):
        '''TODO: docstring'''
        d_model = input_shape[-1]
        self.weights_1 = self.add_weight(
                name='weights_1',
                shape = (d_model, self.size_multiplier * d_model),
                initializer='glorot_uniform',
                trainable=True)
        self.bias_1 = self.add_weight(
               name='bias_1',
               shape=(self.size_multiplier * d_model,),
               initializer='zeros',
               trainable=True)
        self.weights_2 = self.add_weight(
               name='weights_2',
               shape=(self.size_multiplier * d_model, d_model),
               initializer='glorot_uniform',
               trainable=True)
        self.bias_2 = self.add_weight(
               name='bias_2',
               shape=(d_model,),
               initializer='zeros',
               trainable=True)
        return super().build(input_shape)

    def call(self, inputs, **kwargs):
        '''TODO: docstring'''
        input_shape = K.int_shape(inputs)
        d_model = input_shape[-1]
        out = K.reshape(inputs, [-1, d_model])
        out = K.dot(out, self.weights_1)
        out = K.bias_add(out, self.bias_1)
        out = keras.activations.relu(out)
        out = K.dot(out, self.weights_2)
        out = K.bias_add(out, self.bias_2)
        out = K.reshape(out, (-1,) + input_shape[-2:])
        return out

class LayerNormalization(Layer):
    '''TODO: docstring'''

    def build(self, input_shape):
        '''TODO: docstring'''
        dim = input_shape[-1]
        self.gain = self.add_weight(
                name='gain',
                shape=(dim,),
                initializer='ones',
                trainable=True)
        self.bias = self.add_weight(
                name='bias',
                shape=(dim,),
                initializer='zeros',
                trainable=True)
        return super().build(input_shape)

    def call(self, inputs, **kwargs):
        '''TODO: docstring'''
        mean = K.mean(inputs, axis=-1, keepdims=True)
        variance = K.mean(K.square(inputs - mean), axis=-1, keepdims=True)
        eps = K.constant(1e-5)
        normalised_inputs = (inputs - mean) / K.sqrt(variance + eps)
        return self.gain * normalised_inputs + self.bias

