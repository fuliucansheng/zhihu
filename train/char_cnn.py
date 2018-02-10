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

logging.info("Load Train && Val")
train = pd.read_csv(Config.cache_dir+"/train.csv", header=["qid", "title_char", "title_word", "descr_char", "descr_word"], sep="\t")
label = pd.read_csv(Config.train_label_path, header=["qid", "label"], sep="\t")
train = pd.merge(train, label, on="qid", how="left")
val = pd.read_csv(Config.cache_dir+"/val.csv",sep="\t")
val_label = get_labels(val.label)

batch_size = 128
model_name = "char_cnn"
trainable_layer = ["embedding"]
train_batch_generator = char_cnn_train_batch_generator

logging.info("Load Val Data")
val_char_seq = pickle.load(open(Config.cache_dir+"/g_val_char_seq_{}_{}.pkl".format(Config.title_char_seq_maxlen, Config.descr_char_seq_maxlen), "rb"))
val_seq = val_char_seq

logging.info("Load Word")
char_embed_weight = np.load(Config.char_embed_weight_path)

model = get_textcnn(Config.title_char_seq_maxlen, Config.descr_char_seq_maxlen, char_embed_weight)

for i in range(6):
    if i==3:
        K.set_value(model.optimizer.lr, 0.0001)
    if i==2:
        for l in trainable_layer:
            model.get_layer(l).trainable = True
    model.fit_generator(
        train_batch_generator(train, train.label.values, batch_size=batch_size),
        epochs = 1,
        steps_per_epoch = int(train.shape[0]/batch_size),
        validation_data = (val_seq, val_label)
    )
    pred = model.predict(val_seq)
    pre,rec,f1 = map_score(pred)
    print(pre,rec,f1)
    model.save(Config.cache_dir + "/dp_embed_%s_epoch_%s_%s.h5"%(model_name, i, f1))
