import sys
sys.path.append("..")
from config import *

def get_label_dict():
    result_path = Config.cache_dir + "/label.dict.pkl"
    if os.path.exists(result_path):
        result = pickle.load(open(result_path, "rb"))
    else:
        topic_info = pd.read_csv(Config.topic_info_path, header=["tid", "pid", "title_char", "title_word", "descr_char", "descr_word"],sep="\t")
        result = dict(zip(topic_info["tid"], range(topic_info.shape[0])))
        pickle.dump(result, open(result_path, "wb"))
    return result

def get_reset_label_dict():
    result_path = Config.cache_dir + "/label_reset.dict.pkl"
    if os.path.exists(result_path):
        result = pickle.load(open(result_path, "rb"))
    else:
        topic_info = pd.read_csv(Config.topic_info_path, header=["tid", "pid", "title_char", "title_word", "descr_char", "descr_word"],sep="\t")
        result = dict(zip(range(topic_info.shape[0]), topic_info["tid"]))
        pickle.dump(result, open(result_path, "wb"))
    return result

def get_train_data():
    result_path = Config.data_dir + "/train_set.csv"
    if os.path.exists(result_path):
        result = pd.read_csv(result_path, sep="\t")
        result.fillna("", inplace=True)
    else:
        result = pd.read_csv(Config.train_set_path, header=["qid", "title_char", "title_word", "descr_char", "descr_word"], sep="\t")
        labels = pd.read_csv(Config.train_label_path, header=["qid", "label"], sep="\t")
        result = pd.merge(result, labels, on="qid", how="left")
        result.fillna("", inplace=True)
        result.to_csv(result_path, index=False, sep="\t")
    return result

def get_test_data():
    result_path = Config.data_dir + "/test_set.csv"
    if os.path.exists(result_path):
        result = pd.read_csv(result_path, sep="\t")
        result.fillna("", inplace=True)
    else:
        result = pd.read_csv(Config.test_set_path, header=["qid", "title_char", "title_word", "descr_char", "descr_word"], sep="\t")
        result.fillna("", inplace=True)
        result.to_csv(result_path, index=False, sep="\t")
    return result

def map_score(prediction):
    pred = pd.DataFrame(np.argsort(-prediction,axis=1)[:,:5])
    val = pd.read_csv(Config.cache_dir + "/val.csv", header=["qid", "title_char", "title_word", "descr_char", "descr_word"], sep="\t")
    labels = pd.read_csv(Config.train_label_path, header=["qid", "label"], sep="\t")
    val = pd.merge(val, labels, on="qid", how="left")
    label_reset_dict = get_reset_label_dict()
    for i in range(5):
        pred[i] = pred[i].map(label_reset_dict)
    result = pd.concat([val[["qid", "label"]], pred], axis=1)
    def calc(row):
        label = set(row["label"].split(","))
        row["label_count"] = len(label)
        for i in range(5):
            row[i] = int(row[i] in label)
        return row
    result = result.apply(calc, axis=1)
    precision = 0.0
    right_label_num = 0
    for i in range(5):
        right_label_num += result[i].sum()
        precision += result[i].sum() / float(result.shape[0]) / math.log(2.0 + i)
    recall = float(right_label_num) / result["label_count"].sum()
    score = (precision * recall) / (precision + recall)
    return precision, recall, score

def submit(prediction, sub_path=Config.cache_dir + "/submission.csv"):
    pred = pd.DataFrame(np.argsort(-prediction,axis=1)[:,:5])
    label_reset_dict = get_reset_label_dict()
    for i in range(5):
        pred[i] = pred[i].map(label_reset_dict)
    test_data = get_test_data()
    submission = pd.concat([test_data["qid"], pred], axis=1)
    submission.to_csv(sub_path, index=False, header=None)
    