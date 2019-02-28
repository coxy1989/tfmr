import nltk
import re
import keras
import attention
import numpy as np
from keras.layers import Input, Embedding, Dense
from keras.models import Model
import keras.backend as K

def get_lines(file_name):
    '''TODO: docstring'''
    lines = []
    with open(file_name) as f:
        for line in f:
            line = line.strip()
            if line == '' or '*' in line:
                continue
            lines.append(line)
    return lines

def preprocess_sentence(sent):
    '''TODO: docstring'''
    ret = sent.lower()
    ret = re.sub(r'\W', ' ', ret)
    ret = re.sub(r'\s+', ' ', ret)
    ret = f'<start> {ret} <eos>'
    return ret

def get_token_seq(lines):
    '''TODO: docstring'''
    doc = ' '.join(lines)
    sents = nltk.sent_tokenize(doc)
    sents = [preprocess_sentence(sent) for sent in sents if len(sent.split(' ')) > 2]
    clean_doc = ' '.join(sents)
    token_seq = nltk.tokenize.WhitespaceTokenizer().tokenize(clean_doc)
    return token_seq

def vocabularize(token_seq, vocab_size):
    '''TODO: docstring'''
    fd = nltk.FreqDist()
    for w in token_seq:
        fd.update([w])
    fd_lookup = dict(fd)
    fd_lookup.pop('<start>')
    fd_lookup.pop('<eos>')
    word_freqs = sorted(fd_lookup.items(), key=lambda kv: kv[1], reverse=True)
    vocab = ['<unk>', '<start>', '<eos>'] + [kv[0] for kv in word_freqs[:vocab_size - 3]]
    idx_token = {k : v for k,v in enumerate(vocab)}
    token_idx = {v : k for k,v in enumerate(vocab)}
    idx_seq = [token_idx.get(token, 0) for token in token_seq]
    return vocab, idx_seq, idx_token, token_idx

def subseqs(seq, subseq_len):
    '''TODO: docstring'''
    i = 0
    subseqs = []
    while (i + subseq_len  <  len(seq)):
        subseqs.append(seq[i: i + subseq_len])
        i += 1
    return subseqs

def get_model(subseq_len, vocab_size, d_model, n_heads=4):
    '''TODO: docstring'''
    input_layer = Input(shape=[subseq_len])
    embedding_layer = Embedding(vocab_size, d_model, input_length=subseq_len, name='embed')
    position_layer = attention.PositionalEncoding()
    decoder = attention.TransformerDecoderBlock('decoder', num_heads=n_heads)
    dense = Dense(vocab_size, activation='softmax', name='softmax')
    out = dense(decoder(position_layer(embedding_layer(input_layer))))
    return Model(inputs=[input_layer], outputs=out)

def encode_one_hot(ys, vocab_size):
    '''TODO: docstring'''
    bs, sl = ys.shape
    ret = np.zeros([bs, sl, vocab_size])
    for example_idx, example in enumerate(ys):
        ret[example_idx, :, :] = keras.utils.to_categorical(example, num_classes=vocab_size)
    return ret

def decode(sequences, idx_token):
    '''TODO: docstring'''
    sentences = []
    for sequence in sequences:
        sentence = ''
        for idx in sequence:
            # ignore <start> token
            if idx == 1:
                continue
            # replace <end> with ,
            if idx == 2:
                sentence += ', '
                continue
            token = idx_token[idx]
            sentence += token + ' '
        sentences.append(sentence)
    return sentences

def generate(model, n, subseq_len, vocab_size, idx_token):
    '''TODO: docstring'''
    ins = np.zeros([n, subseq_len])
    outs = np.zeros([n, subseq_len])
    ins[:,0] = np.ones(n)
    for t in range(subseq_len - 1):
        preds = model.predict(ins)
        probs_t = preds[:,t,:]
        # Set probability of <unk> to zero and renormalize
        probs_t[:,0] = np.zeros(n)
        probs_t = probs_t / np.expand_dims(probs_t.sum(-1), -1)
        # Select token with probability probs_t
        t_idxs = np.array([np.random.choice(np.arange(vocab_size), p=v) for v in probs_t])
        ins[:,t + 1] = t_idxs
        outs[:,t] = t_idxs
    return decode(outs, idx_token)

def perplexity(y_true, y_pred):
    """TODO: docstring"""
    cross_entropy = K.categorical_crossentropy(y_true, y_pred)
    return K.mean(K.exp(K.mean(cross_entropy, axis=-1)))

def main():
    '''TODO: docstring'''
    VOCAB_SIZE = 100
    SUBSEQ_LEN = 64
    D_MODEL = 64

    nltk.download('punkt')

    lines = get_lines('./data/baking.txt')
    token_seq = get_token_seq(lines)
    vocab, idx_seq, idx_token, token_idx = vocabularize(token_seq, VOCAB_SIZE)
    xs = np.array(subseqs(idx_seq, SUBSEQ_LEN)[:-1])
    ys = np.array(subseqs(idx_seq[1:], SUBSEQ_LEN ))

    model = get_model(SUBSEQ_LEN, VOCAB_SIZE, D_MODEL)
    opt = keras.optimizers.Adam(lr=0.001)
    model.compile(opt, 'categorical_crossentropy', metrics=[perplexity])
    generate_stuff = keras.callbacks.LambdaCallback(on_epoch_end=
            lambda epoch, logs: [print(w) for w in generate(model, 3, SUBSEQ_LEN, VOCAB_SIZE, idx_token)])
    model.fit(xs,
            encode_one_hot(ys, VOCAB_SIZE),
            epochs=10,
            batch_size=256,
            callbacks=[generate_stuff],
            validation_split=0.1)

if __name__ == "__main__":
    main()

