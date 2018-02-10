import sys
sys.path.append("..")
from functools import partial
from config import *
from utils.data import *
from keras.preprocessing import sequence

char_embed_dict = pickle.load(open(Config.char_embed_dict_path, "rb"))
word_embed_dict = pickle.load(open(Config.word_embed_dict_path, "rb"))
label_dict = get_label_dict()

char_unknown = len(char_embed_dict.keys()) + 1
word_unknown = len(word_embed_dict.keys()) + 1

def get_word_seq(contents, word_maxlen, mode="post", keep=False, verbose=False):
    word_r = []
    for content in tqdm(contents, disable=(not verbose)):
        content = content.split(",")
        if keep:
            word_c = np.array([word_embed_dict[w] if w in word_embed_dict else word_unknown for w in content])
        else:
            word_c = np.array([word_embed_dict[w] for w in content if w in word_embed_dict])
        word_r.append(word_c)
    word_seq = sequence.pad_sequences(word_r, maxlen=word_maxlen, padding=mode, truncating=mode, value=0)
    return word_seq

def get_char_seq(contents, char_maxlen, mode="post", keep=False, verbose=False):
    char_r = []
    for content in tqdm(contents, disable=(not verbose)):
        content = content.split(",")
        if keep:
            char_c = np.array([char_embed_dict[c] if c in char_embed_dict else char_unknown for c in content])
        else:
            char_c = np.array([char_embed_dict[c] for c in content if c in char_embed_dict])
        char_r.append(char_c)
    char_seq = sequence.pad_sequences(char_r, maxlen=char_maxlen, padding=mode, truncating=mode, value=0)
    return char_seq

def get_labels(contents, verbose=False):
    labels = np.zeros(shape=(len(contents), 1999))
    for idx, content in enumerate(tqdm(contents, disable=(not verbose))):
        content = list(map(label_dict, content.split(",")))
        labels[idx][content] = 1;
    return labels

## batch generator
def make_batches(size, batch_size):
    nb_batch = int(np.ceil(size/float(batch_size)))
    return [(i*batch_size, min(size, (i+1)*batch_size)) for i in range(0, nb_batch)]

def batch_generator(contents, labels, batch_size=128, shuffle=True, keep=False, preprocessfunc=None):
    assert preprocessfunc != None
    sample_size = contents.shape[0]
    index_array = np.arange(sample_size)
    while 1:
        if shuffle:
            np.random.shuffle(index_array)
        batches = make_batches(sample_size, batch_size)
        for batch_index, (batch_start, batch_end) in enumerate(batches):
            batch_ids = index_array[batch_start:batch_end]
            batch_contents = contents[batch_ids]
            batch_contents = preprocessfunc(batch_contents, keep=keep)
            batch_labels = get_labels(labels[batch_ids])
            yield (batch_contents,batch_labels)

## word preprocess
def word_cnn_preprocess(contents, title_word_maxlen=Config.title_word_seq_maxlen, descr_word_maxlen=Config.descr_word_seq_maxlen, keep=False):
    title_word_seq = get_word_seq(contents["title_word"], word_maxlen=title_word_maxlen, keep=keep)
    descr_word_seq = get_word_seq(contents["descr_word"], word_maxlen=descr_word_maxlen, keep=keep)
    return [title_word_seq, descr_word_seq]

## char preprocess
def char_cnn_preprocess(contents, title_char_maxlen=Config.title_char_seq_maxlen, descr_char_maxlen=Config.descr_char_seq_maxlen, keep=False):
    title_char_seq = get_char_seq(contents["title_char"], char_maxlen=title_char_maxlen, keep=keep)
    descr_char_seq = get_char_seq(contents["descr_char"], char_maxlen=descr_char_maxlen, keep=keep)
    return [title_char_seq, descr_char_seq]

def word_cnn_train_batch_generator(train_content, train_label, batch_size=128, keep=False):
    return batch_generator(contents=train_content, labels=train_label, batch_size=batch_size, keep=keep, preprocessfunc=word_cnn_preprocess)

def char_cnn_train_batch_generator(train_content, train_label, batch_size=128, keep=False):
    return batch_generator(contents=train_content, labels=train_label, batch_size=batch_size, keep=keep, preprocessfunc=char_cnn_preprocess)
