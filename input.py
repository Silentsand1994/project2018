from os import listdir
from sklearn.cluster import KMeans
import librosa
from os.path import join
import numpy as np
import json_tricks
import json
from gensim.models import Word2Vec, word2vec


def get_feature(path):
    print(path)
    y, sr = librosa.load(path)

    y = np.resize(y, (int(y.shape[0]/511), 511))

    features = np.zeros((y.shape[0], 14), dtype=y.dtype)

    for x in range(0, y.shape[0]-1):
        zero = librosa.zero_crossings(y[x])
        zero = zero.tolist()
        features[x][0] = zero.count(True)
        features[x][1] = librosa.feature.rmse(y=y[x])
        temp = librosa.feature.mfcc(y=y[x], n_mfcc=12)
        features[x][2] = temp[0]
        features[x][3] = temp[1]
        features[x][4]= temp[2]
        features[x][5] = temp[3]
        features[x][6] = temp[4]
        features[x][7] = temp[5]
        features[x][8] = temp[6]
        features[x][9] = temp[7]
        features[x][10] = temp[8]
        features[x][11] = temp[9]
        features[x][12] = temp[10]
        features[x][13] = temp[11]

    return (features, y.shape[0])


def word2vec_train(x, m, file_path):
    num = x.shape[0]
#    feature_list = np.zeros(shape=(num, m), dtype=int)
    x = x.tolist()
    word = ""
    file_path = join(file_path, "word_vector")

    for i in range(0, num - 1):
        for j in range(0, m - 1):
            word = word + str(int(x[i][j][0])) + " "
 #           feature_list[i][j] = str(int(x[i][j][0]))
        word = word + "\n"
        print("loading:", i)

    with open("vocab", "w") as f:
        f.write(word)
        f.close()

    word = word2vec.LineSentence('vocab')
    models = Word2Vec(word, size=14, window=m, min_count=1) #Size 可調整 -> 碼字向量維度

    data = np.zeros(shape=(num, m, 14))
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
            if c_p :
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
        kmeans = KMeans(n_clusters=100).fit(data1)
        data = kmeans.predict(data1)
        p = 0

        data1 = np.zeros(shape=(count, max_f), dtype=int)
        for x in range(0, count-1):
            nframes = f_data[x]
            temp = 0
            for i in range(p, p+nframes):
                data1[x][temp] = data[i][0]
                temp = temp + 1
                print(data[i][0])
            p = p + nframes

        print("word2vec start---------------")
        data1 = word2vec_train(data1, max_f, file_path)
        data2 = np.array(data2)
        f = open(file_path, "w")
        json_tricks.dump((data1, data2, max_f, dic), f)
        f.close()

    else:
        file_path = join("data_path", path)
        f = open(file_path, "r")
        a = json_tricks.load(f)
        data1, data2, max_f, dic = a

    return data1, data2, max_f, len(dic)


a, b, c, d = load_data("wav")

with open("output_test", "w") as f:
    json.dump(a.shape, f)
    json_tricks.dump(a, f)
    f.close()