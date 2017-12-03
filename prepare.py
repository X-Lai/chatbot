import torch
import torch.autograd as autograd

def indexFromSentence(sentence, word2ix):
    index=[]
    for word in sentence.split():
        if word not in word2ix:
            index.append(0)# 0 represents UNK
        else:
            index.append(word2ix[word])
    index.append(1)# 1 represents EOS
    return index

def indexesFromFile(path, word2ix):
    indexes=[]
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            indexes.append(indexFromSentence(line, word2ix))
    return indexes

def variablePairsFromFile(path_ask ,path_reply, word2ix, train_set_size=0):
    ask = indexesFromFile(path_ask,word2ix)
    reply = indexesFromFile(path_reply,word2ix)
    train_set_size = len(ask)
    return [( ask[i],reply[i]) for i in range(train_set_size)]

def get_dict(path_ask,path_reply):
    dict={"UNK":0,"EOS":1,"PAD":2}
    dict_r=["UNK","EOS","PAD"]
    num=3
    with open(path_ask, "r", encoding="utf-8") as file:
        for line in file:
            for word in line.split():
                if word not in dict.keys():
                    dict[word]=num
                    dict_r.append(word)
                    num = num + 1
    with open(path_reply,"r",encoding="utf-8") as file:
        for line in file:
            for word in line.split():
                if word not in dict.keys():
                    dict[word]=num
                    dict_r.append(word)
                    num = num + 1
    return dict,dict_r
