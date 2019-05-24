import json  # we need to use the JSON package to load the data, since the data is stored in JSON format
from collections import Counter
import re
import numpy as np

with open("proj1_data.json") as fp:
    data = json.load(fp)

training = data[0:10000]  # the list contains elements from position 0 to position 9999, total 10000
validation = data[10000:11000]  # the list contains elements from position 10000 to position 10999, total 1000
testing = data[11000:12000]  # the list contains elements from position 11000 to position 11999, total 1000


def transform(partition):
    partlen = len(partition)
    hugelist = [''] * 250000
    j = 0
    children = [0] * 10000
    controversiality = [0] * 10000
    is_root = [0] * 10000
    popularity_score = [0] * partlen

    for pos in range(partlen):  # e.x. for x in range(6): 0 1 2 3 4 5

        data_point = partition[pos]  # select the first data point in the dataset

        children[pos] = data_point['children']
        controversiality[pos] = data_point['controversiality']
        is_root[pos] = int(data_point['is_root'])

        popularity_score[pos] = data_point['popularity_score']
        # print(data_point["text"])
        # print(data_point["text"].lower())
        # Now we print all the information about this datapoint
        # for info_name, info_value in data_point.items():
        #    print(info_name + " : " + str(info_value))
        #
        # print("\n")
        lowerx = data_point['text'].lower()
        # withoutpunc=lowerx.translate(None, string.punctuation)

        data_point["text"] = lowerx.split()
        # print(data_point)
        # for info_name, info_value in data_point.items():
        #    print(info_name + " : " + str(info_value))
        # print("\n")

        # print(len(data_point["text"]))
        slen = len(data_point["text"])

        # s = "string. With. Punctuation?"
        # s = re.sub(r'[^\w\s]','',s)

        # print(slen)
        # print(data_point["text"][0]) #specific word in "text"
        for i in range(slen):  # e.x. for x in range(6): 0 1 2 3 4 5

            # data_point["text"][i]=data_point["text"][i].translate(None, string.punctuation)

            data_point["text"][i] = re.sub(r'[^\w\s]', '', data_point["text"][i])  # removing the punctuation
            # print(data_point["text"][i])

            hugelist[i + j] = data_point["text"][i]

        j = j + slen

    # print(hugelist)
    # print(training[9999])
    hugelist = filter(None, hugelist)  # decrease the list size by removing empty spot
    # print(hugelist)

    # print("is_root", is_root)

    # print(filt_list)

    c = Counter(hugelist).most_common(160)
    # print(c)

    most = [''] * 160  # mopst frequent 160 words
    for i in range(160):
        most[i] = c[i][0]

    # print(most[0])

    # xcounts=[[0]*160]*10000
    # print(most)

    n = len(partition)
    m = 164
    xcounts = [0] * n
    for i in range(n):
        xcounts[i] = [0] * m

    countfre = 0

    for pos in range(partlen):  # e.x. for x in range(6): 0 1 2 3 4 5

        data_point = partition[pos]
        # print(data_point)
        slen = len(data_point["text"])

        for i in range(160):
            for j in range(slen):
                # print(data_point["text"][j])
                if most[i] == data_point["text"][j]:
                    countfre += 1

            #	print(countfre)
            xcounts[pos][i] = countfre
            countfre = 0

        xcounts[pos][160] = children[pos]
        xcounts[pos][161] = controversiality[pos]
        xcounts[pos][162] = is_root[pos]
        xcounts[pos][163] = 1  # bias term

    # print(xcounts)
    # print(xcounts)
    # countfre=0

    X = np.asarray(xcounts)
    Y = np.asarray(popularity_score)

    # print(X.shape)
    return X, Y


X_train, y_train = transform(training)  # input training/validation/testing
X_val, y_val = transform(validation)
X_test, y_test = transform(testing)

# print (X_train.shape)
# print (y_train.shape)

# print (X_train)
# print (y_train)

# print (X_val.shape)
# print (y_val.shape)

# print (X_val)
# print (y_val)

# print (X_test.shape)
# print (y_test.shape)

# print (X_test)
# print (y_test)
