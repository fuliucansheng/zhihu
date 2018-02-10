import os
import sys
import platform
import codecs
import pandas as pd
import numpy as np
import re
import gc
import math
import pickle
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', datefmt='%Y-%m-%d %H:%M:%S %p', level=logging.INFO)

from tqdm import tqdm
from functools import partial
import warnings
warnings.filterwarnings('ignore')

class Config():
    data_dir = "/mnt/data/zhihu"
    cache_dir = data_dir + "/cache"

    ## 原始数据
    char_embedding_path = data_dir + "/data/char_embedding.txt"
    word_embedding_path = data_dir + "/data/word_embedding.txt"
    topic_info_path = data_dir + "/data/topic_info.txt"
    train_set_path = data_dir + "/data/question_train_set.txt"
    train_label_path = data_dir + "/data/question_topic_train_set.txt"
    test_set_path = data_dir + "/data/question_eval_set.txt"

    ## 词dict及预训练权重
    word_embed_dict_path = cache_dir + "/word_embed.dict.pkl"
    word_embed_weight_path = cache_dir + "/word_embed.npy"
    ## 字dict及预训练权重
    char_embed_dict_path = cache_dir + "/char_embed.dict.pkl"
    char_embed_weight_path = cache_dir + "/char_embed.npy"

    ## 截断补齐文本词的个数
    title_word_seq_maxlen = 50
    descr_word_seq_maxlen = 150
    ## 截断补齐文本字的个数
    title_char_seq_maxlen = 1962
    descr_char_seq_maxlen = 150
