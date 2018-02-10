import sys
sys.path.append("..")
from config import *
from utils.data import *
from utils.preprocess import *
from sklearn.model_selection import train_test_split
from gensim.models import KeyedVectors

def gen_word_embedding():
    word_embedding = KeyedVectors.load_word2vec_format(open(Config.word_embedding_path),binary=False)
    word_embed_dict = dict([(k,v.index+1) for k,v in word_embedding.vocab.items()])
    weights = word_embedding.syn0
    word_embed_weights = np.zeros(shape=(weights.shape[0]+2, weights.shape[1]))
    word_embed_weights[1:weights.shape[0]+1] = weights
    unk_vec = np.random.random(size=weights.shape[1])*0.5
    word_embed_weights[weights.shape[0]+1] = unk_vec - unk_vec.mean()
    pickle.dump(word_embed_dict, open(Config.word_embed_dict_path, "wb"))
    np.save(Config.word_embed_weight_path, word_embed_weights)

def gen_char_embedding():
    char_embedding = KeyedVectors.load_word2vec_format(open(Config.char_embedding_path),binary=False)
    char_embed_dict = dict([(k,v.index+1) for k,v in char_embedding.vocab.items()])
    weights = char_embedding.syn0
    char_embed_weights = np.zeros(shape=(weights.shape[0]+2, weights.shape[1]))
    char_embed_weights[1:weights.shape[0]+1] = weights
    unk_vec = np.random.random(size=weights.shape[1])*0.5
    char_embed_weights[weights.shape[0]+1] = unk_vec - unk_vec.mean()
    pickle.dump(char_embed_dict, open(Config.char_embed_dict_path, "wb"))
    np.save(Config.char_embed_weight_path, char_embed_weights)

gen_char_embedding()
gen_word_embedding()

train_data = get_train_data()
train, val = train_test_split(train_data, test_size=0.1)
train.to_csv(Config.cache_dir+"/train.csv", index=False)
val.to_csv(Config.cache_dir+"/val.csv", index=False)

val = pd.read_csv(Config.cache_dir+"/val.csv", sep="\t")
val_word_cnn = word_cnn_preprocess(val)
gc.collect()
pickle.dump(val_word_cnn, open(Config.cache_dir+"/g_val_word_cnn_{}_{}.pkl".format(Config.title_word_seq_maxlen, Config.descr_word_seq_maxlen), "wb"))

val_char_cnn = char_cnn_preprocess(val)
gc.collect()
pickle.dump(val_char_cnn, open(Config.cache_dir+"/g_val_char_cnn_{}_{}.pkl".format(Config.title_char_seq_maxlen, Config.descr_char_seq_maxlen), "wb"))

test_data = get_test_data()
test_word_cnn = word_cnn_preprocess(test_data)
gc.collect()
pickle.dump(test_word_cnn, open(Config.cache_dir+"/g_test_word_cnn_{}_{}.pkl".format(Config.title_word_seq_maxlen, Config.descr_word_seq_maxlen), "wb"))

test_char_cnn = char_cnn_preprocess(test_data)
gc.collect()
pickle.dump(test_char_cnn, open(Config.cache_dir+"/g_test_char_cnn_{}_{}.pkl".format(Config.title_char_seq_maxlen, Config.descr_char_seq_maxlen), "wb"))
