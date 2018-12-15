import os
import csv
import numpy as np

from tqdm import tqdm

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

seed = 3535999445

def _rocstories(path):
    with open(path, encoding='utf_8') as f:
        f = csv.reader(f)
        st = []
        ct1 = []
        ct2 = []
        y = []
        for i, line in enumerate(tqdm(list(f), ncols=80, leave=False)):
            if i > 0:
                s = ' '.join(line[1:5])
                c1 = line[5]
                c2 = line[6]
                st.append(s)
                ct1.append(c1)
                ct2.append(c2)
                y.append(int(line[-1])-1)
        return st, ct1, ct2, y

def rocstories(data_dir, n_train=1497, n_valid=374):
    storys, comps1, comps2, ys = _rocstories(os.path.join(data_dir, 'cloze_test_val__spring2016 - cloze_test_ALL_val.csv'))
    teX1, teX2, teX3, _ = _rocstories(os.path.join(data_dir, 'cloze_test_test__spring2016 - cloze_test_ALL_test.csv'))
    tr_storys, va_storys, tr_comps1, va_comps1, tr_comps2, va_comps2, tr_ys, va_ys = train_test_split(storys, comps1, comps2, ys, test_size=n_valid, random_state=seed)
    trX1, trX2, trX3 = [], [], []
    trY = []
    for s, c1, c2, y in zip(tr_storys, tr_comps1, tr_comps2, tr_ys):
        trX1.append(s)
        trX2.append(c1)
        trX3.append(c2)
        trY.append(y)

    vaX1, vaX2, vaX3 = [], [], []
    vaY = []
    for s, c1, c2, y in zip(va_storys, va_comps1, va_comps2, va_ys):
        vaX1.append(s)
        vaX2.append(c1)
        vaX3.append(c2)
        vaY.append(y)
    trY = np.asarray(trY, dtype=np.int32)
    vaY = np.asarray(vaY, dtype=np.int32)
    return (trX1, trX2, trX3, trY), (vaX1, vaX2, vaX3, vaY), (teX1, teX2, teX3)

def _multiclas(path):
    # Assumes that the data is stored in TSV (not CSV) format
    # Also assumes there is no header in the TSV file
    with open(path, encoding='utf_8') as f:
        f = csv.reader(f, delimiter='\t')
        text = []
        ys = []
        for i, line in enumerate(tqdm(list(f), leave=False)):
            if i > 0:
                text.append(line[0])
                if len(line) == 2:
                    ys.append(int(line[1]))
    return text, ys

def multiclas(data_dir, n_train=400, n_valid=100):
    text, ys = _multiclas(os.path.join(data_dir, 'train.tsv'))
    teX, _ = _multiclas(os.path.join(data_dir, 'test.tsv'))
    tr_text, va_text, tr_sent, va_sent = train_test_split(text, ys, test_size=n_valid, random_state=seed)
    trX = []
    trY = []
    for t, s in zip(tr_text, tr_sent):
        trX.append(t)
        trY.append(s)

    vaX = []
    vaY = []
    for t, s in zip(va_text, va_sent):
        vaX.append(t)
        vaY.append(s)
    trY = np.asarray(trY, dtype=np.int32)
    vaY = np.asarray(vaY, dtype=np.int32)
    return (trX, trY), (vaX, vaY), (teX, )

def _news(path):
    with open(path, encoding='utf_8') as f:
        f = csv.reader(f, delimiter='\t')
        s1 = []
        s2 = []
        ys = []
        for i, line in enumerate(tqdm(list(f), leave=False)):
            if i > 0:
                s1.append(line[0])
                s2.append(line[1])
                ys.append(int(line[2]))

    return s1, s2, ys

def news(data_dir, n_train=1497, n_valid=374):
    """
    news Textual Entail problem as per GLUE benchmark
    """
    s1, s2, ys = _news(os.path.join(data_dir, 'train.tsv'))
    teX1, teX2, _ = _news(os.path.join(data_dir, 'test.tsv'))
    
    tr_s1, va_s1, tr_s2, va_s2, tr_ys, va_ys = train_test_split(s1, s2, ys, test_size=n_valid, random_state=seed)
    trX1, trX2 = [], []
    trY = []
    for s1, s2, y in zip(tr_s1, tr_s2, tr_ys):
        trX1.append(s1)
        trX2.append(s2)
        trY.append(y)

    vaX1, vaX2 = [], []
    vaY = []
    for s1, s2, y in zip(va_s1, va_s2, va_ys):
        vaX1.append(s1)
        vaX2.append(s2)
        vaY.append(y)

    trY = np.asarray(trY, dtype=np.int32)
    vaY = np.asarray(vaY, dtype=np.int32)

    return (trX1, trX2, trY), (vaX1, vaX2, vaY), (teX1, teX2)

if __name__ == "__main__":
    ## Test
    data_dir = "data/agnews"

    # (trX1, trX2, trX3, trY), (vaX1, vaX2, vaX3, vaY), (teX1, teX2, teX3) = \
    #     rocstories(data_dir, n_train=1497, n_valid=374)
    # (trX, trY), (vaX, vaY), (teX, ) = multiclas(data_dir, n_valid=100)

    # print(trX[0])

    (trX1, trX2, trY), (vaX1, vaX2, vaY), (teX1, teX2) = \
        news(data_dir, n_valid=498)
    print(len(trX1), len(teX1))
    print(trX1[:2])
    print(trX2[:2])
    print(trY[:2])
    print(list(set(trY)))

def _sts(path):
    with open(path, encoding='utf_8') as f:
        f = csv.reader(f, delimiter='\t')
        st1 = []
        st2 = []
        y = []
        for i, line in enumerate(tqdm(list(f), ncols=80, leave=False)):
            if i > 0:
                try:
                    s1 = line[5]
                    s2 = line[6]
                    st1.append(s1)
                    st2.append(s2)
                    y.append(float(line[4]))
                except IndexError:
                    print("bad line: {}".format(line))
        return st1, st2, y

def sts(data_dir):
    trX1, trX2, trY = _sts(os.path.join(data_dir, 'sts-train.csv'))
    trY = np.asarray(trY, dtype=np.float32).reshape(-1, 1)

    vaX1, vaX2, vaY = _sts(os.path.join(data_dir, 'sts-dev.csv'))
    vaY = np.asarray(vaY, dtype=np.float32)
    vaY = np.asarray(trY, dtype=np.float32).reshape(-1, 1)

    teX1, teX2, _ = _sts(os.path.join(data_dir, 'sts-test.csv'))

    return (trX1, trX2, trY), (vaX1, vaX2, vaY), (teX1, teX2)


