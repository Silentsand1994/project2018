from os import listdir,remove
from sklearn.cluster import KMeans
import librosa
from os.path import join
import numpy as np
import json_tricks
import json
import pickle
from gensim.models import Word2Vec, word2vec


class EmotionData:
    def __init__(self):
        self.max_f = 0
        self.flag = False
        self.dic = {}
        self.kmeans = None
        self.models = None
        self.m_size = 20
        self.feature_map = 200

    def __del__(self):
        files = listdir("data_path")
        for path in files:
            file_path = join("data_path", path)
            remove(file_path)

    def get_feature(self, path):
        print(path)
        y, sr = librosa.load(path)

        y = np.resize(y, (int(y.shape[0] / 511), 511))
        if not self.flag:
            frame = y.shape[0]
        else:
            frame = self.max_f

        features = np.zeros((frame, 14), dtype=y.dtype)

        for x in range(0, frame - 1):
            zero = librosa.zero_crossings(y[x])
            zero = zero.tolist()
            features[x][0] = zero.count(True)
            features[x][1] = librosa.feature.rmse(y=y[x])
            temp = librosa.feature.mfcc(y=y[x], n_mfcc=12)
            features[x][2] = temp[0]
            features[x][3] = temp[1]
            features[x][4] = temp[2]
            features[x][5] = temp[3]
            features[x][6] = temp[4]
            features[x][7] = temp[5]
            features[x][8] = temp[6]
            features[x][9] = temp[7]
            features[x][10] = temp[8]
            features[x][11] = temp[9]
            features[x][12] = temp[10]
            features[x][13] = temp[11]

        return features, frame

    def word2vec_train(self, x):
        num = x.shape[0]
        x = x.tolist()
        word = ""

        for i in range(0, num - 1):
            for j in range(0, self.max_f - 1):
                word = word + str(int(x[i][j])) + " "
            word = word + "\n"
            print("loading:", i)

        with open("vocab", "w") as f:
            f.write(word)
            f.close()

        word = word2vec.LineSentence('vocab')
        self.models = Word2Vec(word, size=self.m_size, window=self.max_f, min_count=1)  # Size 可調整 -> 碼字向量維度

        return

    def set_kmeans(self,data):
        self.kmeans = KMeans(n_clusters=self.feature_map).fit(data)
        return

    def load_data(self, path):
        f_data = []
        data2 = []
        c_p = True
        count = 0
        emotion = 0

        files = listdir("data_path")

        if path not in files:
            file_path = join("data_path", path)
            files = listdir(path)

            for a in files:
                file = join(path, a)
                features, nframes = self.get_feature(file)
                if self.max_f < nframes:
                    self.max_f = nframes
                b = a.strip("1234567890_")
                b = b.split(".")

                f_data.append(nframes)
                if c_p:
                    data1 = features
                    c_p = False
                else:
                    data1 = np.vstack((data1, features))
                if b[0] in self.dic:
                    data2.append(self.dic.get(b[0]))
                else:
                    self.dic[b[0]] = emotion
                    emotion = emotion + 1
                    data2.append(self.dic.get(b[0]))

                count = count + 1

            print("Kmeans start----------------")
            if not self.kmeans:
                self.set_kmeans(data1)

            data = self.kmeans.predict(data1)
            p = 0

            data1 = np.zeros(shape=(count, self.max_f), dtype=int)
            for x in range(0, count - 1):
                nframes = f_data[x]
                temp = 0
                for i in range(p, p + nframes):
                    data1[x][temp] = data[i]
                    temp = temp + 1
                p = p + nframes

            print("word2vec start---------------")
            if not self.models:
                self.word2vec_train(data1)

            data = np.zeros(shape=(data1.shape[0], self.max_f, 20))
            for i in range(0, data1.shape[0] - 1):
                for j in range(0, self.max_f - 1):
                    a = str(data1[i][j])
                    b = self.models.wv[a]
                    data[i][j] = b
            data1 = data

            data2 = np.array(data2)
            with open(file_path, "w") as f:
                json_tricks.dump((data1, data2, self.max_f, self.dic), f)
                f.close()

            self.flag = True

        else:
            file_path = join("data_path", path)
            f = open(file_path, "r")
            a = json_tricks.load(f)
            data1, data2, max_f, dic = a
            if max_f != self.max_f:
                print("error")
            if dic != self.dic:
                print("error")

            return data1, data2, self.max_f, len(self.dic)

