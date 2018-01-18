from os import listdir, remove
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
        files = listdir("bin/data_path")
        for path in files:
            file_path = join("bin/data_path", path)
            remove(file_path)


    def get_feature(self, path):
        print(path)
        y, sr = librosa.load(path)

        zero = librosa.feature.zero_crossing_rate(y=y)
        rmse = librosa.feature.rmse(y=y)
        temp = librosa.feature.mfcc(y=y, n_mfcc=12)
        features = np.vstack((zero, rmse, temp))
        features = features.transpose()
        frame = features.shape[0]

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
        remove('vocab')
        print("word2vec set end")
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

        files = listdir("bin/data_path")

        if path not in files:
            file_path = join("bin/data_path", path)
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
            for i in range(0, data1.shape[0]):
                for j in range(0, self.max_f):
                    data[i][j] = self.models.wv[str(data1[i][j])]

            data1 = data

            data2 = np.array(data2)
            with open(file_path, "w") as f:
                json_tricks.dump((data1, data2, self.max_f, self.dic), f)
                f.close()

            self.flag = True

        else:
            file_path = join("bin/data_path", path)
            f = open(file_path, "r")
            a = json_tricks.load(f)
            data1, data2, max_f, dic = a
            if max_f != self.max_f:
                print("error")
            if dic != self.dic:
                print("error")

        return data1, data2

    def get_Cate(self):
        return len(self.dic)

    def get_nFrame(self):
        return self.max_f