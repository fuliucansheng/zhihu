import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

import sys
sys.path.append("..")
from utils.preprocess import *
import keras
import keras.backend as K
from utils.data import *
from keras.layers import *
from keras.models import *
from keras.optimizers import *
from keras.callbacks import *
from models.deepzoo import *

from sklearn.model_selection import KFold

logging.info("Load Train && Val")
train = pd.read_csv(Config.cache_dir+"/train.csv", header=["qid", "title_char", "title_word", "descr_char", "descr_word"], sep="\t")
label = pd.read_csv(Config.train_label_path, header=["qid", "label"], sep="\t")
train = pd.merge(train, label, on="qid", how="left")
val = pd.read_csv(Config.cache_dir+"/val.csv",sep="\t")
val_label = get_labels(val.label)

batch_size = 128
model_name = "char_cnn_bagging"
trainable_layer = ["embedding"]
train_batch_generator = char_cnn_train_batch_generator

logging.info("Load Val Data")
val_char_seq = pickle.load(open(Config.cache_dir+"/g_val_char_seq_{}_{}.pkl".format(Config.title_char_seq_maxlen, Config.descr_char_seq_maxlen), "rb"))
val_seq = val_char_seq

logging.info("Load Char")
char_embed_weight = np.load(Config.char_embed_weight_path)

N = 5
kf = KFold(n_splits=N,shuffle=True)
for i,(train_,val_) in enumerate(kf.split(train)):
    model = get_textcnn(Config.title_char_seq_maxlen, Config.descr_char_seq_maxlen, char_embed_weight)
    best_model_path = Config.cache_dir + "/{}_{}_weights.h5".format(model_name, i)
    early_stopping =EarlyStopping(monitor='val_loss', patience=1, verbose=1)
    model_checkpoint = ModelCheckpoint(best_model_path, save_best_only=True, save_weights_only=True)
    model.fit_generator(
        train_batch_generator(train_, train_.label.values, batch_size=batch_size),
        epochs = 6,
        steps_per_epoch = int(train_.shape[0]/batch_size),
        validation_data = (val_seq, val_label),
        callbacks=[early_stopping, model_checkpoint]
    )

logging.info("Load Test Data")
test_char_seq = pickle.load(open(Config.cache_dir+"/g_test_char_seq_{}_{}.pkl".format(Config.title_char_seq_maxlen, Config.descr_char_seq_maxlen), "rb"))
test_seq = test_char_seq

test_data = get_test_data()
val_pred = np.zeros(val.shape[0], 1999)
test_pred = np.zeros(test_data.shape[0], 1999)
for i in range(N):
    model_weight_path = Config.cache_dir + "/{}_{}_weights.h5".format(model_name, i)
    model = load_model(model_weight_path)
    val_pred += model.predict(val_seq)
    test_pred += model.predict(test_seq)

val_pred = val_pred / N
test_pred = test_pred / N

## Val Data Pred
pickle.dump(val_pred, open(Config.cache_dir+"/val_%s.pred"%model_name,"wb"))
## Test Data Pred
pickle.dump(test_pred, open(Config.cache_dir+"/test_%s.pred"%model_name,"wb"))
