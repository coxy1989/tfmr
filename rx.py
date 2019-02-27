from modules import attention
import numpy as np
import pandas as pd
import keras
from keras.layers import Input, Dropout, Dense
from html.parser import HTMLParser
from keras.preprocessing import sequence
from keras.models import Model
from functools import partial

# Data
# -----
# - html curled from: https://bnf.nice.org.uk/drug/
# - To obtain links: only the relevent links have `.html` suffix on thier hrefs
# - To obtain drugs: only the drugs are ALL CAPS.
# - hacks
#   - Last 3 drugs obtained using the above method are no good
#   - First link obtained using the above method is no good
#   - I Amended ANTI-D (RH0) IMMUNOGLOBULIN entry in the html to remove <sub> tag

class RxParser(HTMLParser):
    
    def __init__(self):
        super().__init__()
        self.links = []
        self.drugs = []
    
    def handle_starttag(self, tag, attrs):
        if len(attrs) > 0 and attrs[0][0] == 'href' and attrs[0][1][-5:] == '.html':
            self.links.append(attrs[0][1])
            
    def handle_data(self, data):
        if data.isupper() and len(data) > 1:
            self.drugs.append(data)
    
    def feed(self, f):
        super().feed(f)
        return (self.drugs[:-3], self.links[1:])
        
f = open('./data/drugs.html').read()
p = RxParser()
drugs, links = p.feed(f)
assert(len(drugs) == len(links))
drugs, links = pd.Series(drugs), pd.Series(links)

# Curation
# --------
# It is possible to generate plausible drug names with fewer hidden units in the lstm
# and fewer tranining epochs by removing some of the trickier examples from the tranining
# set. For example, removing training examples which contain:
#   - brackets or commas
#   - long 'compound' drug names i.e: "x with y and z"
#   - apostrophes, there is only one: "St John's Wort"
#   - forward slashes, there are two: ADRENALINE/EPINEPHRINE and NORADRENALINE/NOREPINEPHRINE
#   - accented letter É, there are two: BACILLUS CALMETTE-GUÉRIN and BACILLUS CALMETTE-GUÉRIN VACCINE

# very selective (uncomment and comment the `quite selective section`):
# drop_idxs = drugs.str.contains(r"/|'|WITH|AND|É|,|\(", regex=True)
# drugs, links = drugs[~drop_idxs], links[~drop_idxs]

# quite selective:
drop_idxs = drugs.str.contains(r",|\(", regex=True)
drugs, links = drugs[~drop_idxs], links[~drop_idxs]

# not at all selective (don't drop any examples)
# ...


assert(len(drugs) == len(links))
print(f'number of drugs: {len(drugs)}')

# Encoding
# --------

def to_categorical(batch, num_classes):
    b, l = batch.shape
    out = np.zeros((b, l, num_classes))
    for i in range(b):
        seq = batch[i, :]
        out[i, :, :] = keras.utils.to_categorical(seq, num_classes=num_classes)
    return out

def rx_data(words):
    chars = sorted(set(''.join(words)))
    chars = ['<START>', '<END>'] + chars
    char_idx = { ch:i for i,ch in enumerate(chars) }
    idx_char = { i:ch for i,ch in enumerate(chars) }
    print(f'number of characters: {len(chars)}')
    x = ([[char_idx[c] for c in w] for w in words])
    max_len = max([len(s) for s in x])
    print(f'longest word: {max_len}')
    x = sequence.pad_sequences(x, max_len, padding='post', value=1)
    n = x.shape[0]
    x_in = np.concatenate([np.zeros([n, 1]), x[:, :-1]], axis=1)
    x_out = x
    assert x_in.shape == x_out.shape
    x_in = to_categorical(x_in, len(chars))
    x_out = to_categorical(x_out, len(chars))
    return idx_char, max_len, chars, x_in, x_out

idx_char, max_len, chars, x_in, x_out = rx_data(drugs)
num_chars = len(chars)

#  Model
# -------

input_layer = Input(shape=[max_len, num_chars])
position_layer = attention.PositionalEncoding()
decoder = attention.TransformerDecoderBlock('decoder', num_heads=2)
dense = Dense(num_chars, activation='softmax')
out = dense(decoder(position_layer(input_layer)))
model = Model(inputs=[input_layer], outputs=out)

def decode(xs):
    p = np.argmax(xs, -1)
    for v in p:
        w = ''
        cs = []
        for c in v:
            char = idx_char[c]
            w += char
            cs.append(c)
            if c == 1:
                break
        print(w)

def generate(model):
    xs = x_in[10:100:20,:,:]
    decode(xs)
    print('---')
    preds = model.predict(xs)
    decode(preds)

def generate(model):
    ins = np.zeros([3, max_len, num_chars])
    outs = np.zeros([3, max_len, num_chars])
    start_token = to_categorical(np.array([0])[...,None], num_chars)
    ins[:,0,:] = np.squeeze(start_token)
    for t in range(max_len - 1):
        preds = model.predict(ins)
        preds[:, t, :]
        p = preds[:,t,:]
        t_idxs = np.array([np.random.choice(np.arange(num_chars), p=v) for v in p])
        one_hot_idxs = to_categorical(t_idxs[...,None], num_chars)
        ins[:,t + 1,:] = np.squeeze(one_hot_idxs)
        outs[:,t,:] = np.squeeze(one_hot_idxs)
    decode(outs)

opt = keras.optimizers.Adam(lr=0.001)
model.compile(opt, 'categorical_crossentropy')
generate_stuff = keras.callbacks.LambdaCallback(on_epoch_end=lambda epoch, logs: generate(model))
model.fit(x_in,
        x_out,
        epochs=50,
        batch_size=32,
        callbacks=[generate_stuff])

