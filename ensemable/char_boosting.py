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
model_name = "char_cnn_boosting"
trainable_layer = ["embedding"]
train_batch_generator = char_cnn_train_batch_generator

logging.info("Load Val Data")
val_char_seq = pickle.load(open(Config.cache_dir+"/g_val_char_seq_{}_{}.pkl".format(Config.title_char_seq_maxlen, Config.descr_char_seq_maxlen), "rb"))
val_seq = val_char_seq

logging.info("Load Char")
char_embed_weight = np.load(Config.char_embed_weight_path)

def get_classweights(prediction):
    classwrong = np.zeros(shape=(1999))
    classcount = np.zeros(shape=(1999))
    pred = np.argsort(-prediction, axis=1)[:,:5]
    for i,p in enumerate(pred):
        v = np.argwhere(val_label[i] == 1)
        v = v.reshape(-1,)
        for l in v:
            classcount[l] += 1
            if l not in p:
                classwrong[l] += 1
    return classwrong/classcount

for layer in range(10):
    if layer > 0:
        classweight = pickle.load(open(Config.cache_dir+"/{}_c_{}.pkl".format(model_name, layer-1),"rb"))
    else:
        classweight = np.ones(shape=(1999))

    model = get_textcnn(Config.title_char_seq_maxlen, Config.descr_char_seq_maxlen, char_embed_weight)
    best_model_path = Config.cache_dir + "/{}_{}_weights.h5".format(model_name, layer)
    early_stopping = EarlyStopping(monitor='val_loss', patience=1, verbose=1)
    model_checkpoint = ModelCheckpoint(best_model_path, save_best_only=True, save_weights_only=True)
    model.fit_generator(
        train_batch_generator(train, train.label.values, batch_size=batch_size),
        epochs = 6,
        steps_per_epoch = int(train.shape[0]/batch_size),
        validation_data = (val_seq, val_label),
        callbacks=[early_stopping, model_checkpoint],
        class_weight=classweight
    )
    model.load_weights(best_model_path)
    curre = model.predict(val_seq)

    if layer > 0:
        orge = pickle.load(open(Config.cache_dir+"/{}_e_{}.pkl".format(model_name, layer-1),"rb"))
        print("lastlayer score:", map_score(orge))
        newe = orge + curre
        print("currlayer score:", map_score(newe))
    else:
        newe = curre
    classweight = get_classweights(newe)
    print("currlayer error rate:")
    print("max:",classweight.max()," min:",classweight.min()," mean:",classweight.mean()," std:",classweight.std())
    pickle.dump(classweight,open(Config.cache_dir+"/{}_c_{}.pkl".format(model_name, layer),"wb"))
    pickle.dump(newe,open(Config.cache_dir+"/{}_e_{}.pkl".format(model_name, layer),"wb"))

logging.info("Load Test Data")
test_char_seq = pickle.load(open(Config.cache_dir+"/g_test_char_seq_{}_{}.pkl".format(Config.title_char_seq_maxlen, Config.descr_char_seq_maxlen), "rb"))
test_seq = test_char_seq

test_data = get_test_data()
val_pred = np.zeros(val.shape[0], 1999)
test_pred = np.zeros(test_data.shape[0], 1999)
for layer in range(10):
    model_weight_path = Config.cache_dir + "/{}_{}_weights.h5".format(model_name, layer)
    model = load_model(model_weight_path)
    val_pred += model.predict(val_seq)
    test_pred += model.predict(test_seq)

val_pred = val_pred / 10
test_pred = test_pred / 10

## Val Data Pred
pickle.dump(val_pred, open(Config.cache_dir+"/val_%s.pred"%model_name,"wb"))
## Test Data Pred
pickle.dump(test_pred, open(Config.cache_dir+"/test_%s.pred"%model_name,"wb"))
