from modules import attention
import unittest
import numpy as np
import keras.backend as K

class AttentionTests(unittest.TestCase):

    def test_positional_encoding(self):

        pe = attention.PositionalEncoding.positional_encoding(10, 12)
        self.assertTrue(pe.shape == (1, 10, 12))

        pe = attention.PositionalEncoding.positional_encoding(4, 6)
        self.assertTrue(pe.shape == (1, 4, 6))

        pe = attention.PositionalEncoding.positional_encoding(4, 6)
        self.assertTrue(pe[0,0,0] == 0)
        self.assertTrue(pe[0,0,1] == 1)

