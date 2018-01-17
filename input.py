from os import listdir
from sklearn.cluster import KMeans
import librosa
from os.path import join
import numpy as np
import json_tricks
import json
import pickle
from gensim.models import Word2Vec, word2vec


def get_feature(path):
    print(path)
    y, sr = librosa.load(path)
    zero = librosa.feature.zero_crossing_rate(y=y)
    rmse = librosa.feature.rmse(y=y)
    mfcc = librosa.feature.mfcc(y=y, n_mfcc=12)
    features = np.vstack((zero, rmse, mfcc))
    features = features.transpose()
    frame = features.shape[0]

    return features, frame


def word2vec_train(x, m):
    num = x.shape[0]
    x = x.tolist()
    word = ""
    file_path = "word_vector"
    files = listdir("bin")
    if file_path not in files:
        file_path = join("bin", file_path)
        for i in range(0, num):
            for j in range(0, m):
                word = word + str(int(x[i][j])) + " "
            word = word + "\n"
            print("loading:", i)

        with open("vocab", "w") as f:
            f.write(word)
            f.close()

        word = word2vec.LineSentence('vocab')
        models = Word2Vec(word, size=20, window=m, min_count=1) #Size 可調整 -> 碼字向量維度
    else:
        file_path = join("bin", file_path)
        models = Word2Vec.load( file_path)

    data = np.zeros(shape=(num, m, 20))
    for i in range(0, num-1):
        for j in range(0, m-1):
            a = str(x[i][j])
            b = models.wv[a]
            data[i][j] = b

    models.save(file_path)
    return data


def load_data(path):
    f_data = []
    data2 = []
    c_p = True
    count = 0
    max_f = 0
    dic = {}
    emotion = 0
    flag = False
    files = listdir("bin")
    if "Max_frame" in files:
        flag = True
        with open("bin/Max_frame", "r") as f:
            max_f = int(f.read())

    files = listdir("data_path")

    if path not in files:
        file_path = join("data_path", path)
        files = listdir(path)

        for a in files:
            file = join(path, a)
            features, nframes = get_feature(file)
            if max_f < nframes:
                max_f = nframes
            b = a.strip("1234567890_")
            b = b.split(".")

            f_data.append(nframes)
            if c_p:
                data1 = features
                c_p = False
            else:
                data1 = np.vstack((data1, features))
            if b[0] in dic:
                data2.append(dic.get(b[0]))
            else:
                dic[b[0]] = emotion
                emotion = emotion + 1
                data2.append(dic.get(b[0]))

            count = count + 1

        print("Kmeans start----------------")
        files = listdir("bin")
        if "kmeans" not in files:
            kmeans = KMeans(n_clusters=200).fit(data1)
            with open("bin/kmeans", "wb") as f:
                pickle.dump(kmeans, f)
                f.close()
        else:
            with open("bin/kmeans", "rb") as f:
                kmeans = pickle.load(f)
                f.close()

        data = kmeans.predict(data1)
        p = 0

        data1 = np.zeros(shape=(count, max_f), dtype=int)
        for x in range(0, count):
            nframes = f_data[x]
            temp = 0
            for i in range(p, p+nframes):
                data1[x][temp] = data[i]
                temp = temp + 1
            p = p + nframes

        print("word2vec start---------------")
        data1 = word2vec_train(data1, max_f)
        data2 = np.array(data2)
        f = open(file_path, "w")
        json_tricks.dump((data1, data2, max_f, dic), f)
        f.close()

        with open("bin/Max_frame", "w") as f:
            f.write(str(max_f))
            f.close()

    else:
        file_path = join("data_path", path)
        f = open(file_path, "r")
        a = json_tricks.load(f)
        data1, data2, max_f, dic = a
        f.close()

    return data1, data2, max_f, len(dic)
