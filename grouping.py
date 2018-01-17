import shutil
import os

files = os.listdir("wav")
dic = {'ha': 0, 'sa': 0, 'ne': 0, 'an': 0, 'bor': 0, 'anx': 0, 'su': 0, 'di': 0}


for file in files:
    b = file.strip("1234567890_")
    b = b.split(".")
    b = b[0]
    c = dic[b] % 5
    dic[b] = dic[b] + 1

    shutil.copyfile("wav/"+file, "group"+str(c)+"/"+file)




