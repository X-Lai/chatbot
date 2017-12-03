import en_coder
import de_coder
import torch
from prepare import indexFromSentence,get_dict
import torch.autograd as autograd
import random

UNK_token = 0
EOS_token = 1

def evaluate(encoder, decoder, sentence, maxlength,word2ix, ix2word):
    input_variable = autograd.Variable(torch.LongTensor(indexFromSentence(sentence,word2ix)))
    hidden = encoder.init_hidden()

    input_length = input_variable.size()[0]

    for ei in range(input_length-1):
        output, hidden = encoder(input_variable[ei], hidden)
    decoder_input = input_variable[input_length-1]

    reply=[]
    for di in range(maxlength):
        output, hidden = decoder(decoder_input, hidden)
        topv,topi = output[0].data.topk(1)
        ni = topi[0]
        if ni == EOS_token: break
        reply.append(ix2word[ni])
        decoder_input = autograd.Variable(torch.LongTensor([ni]))
    return reply

def evaluateSetFromFile(path, set_size):
    set=[]
    with open(path,"r",encoding="utf-8") as file:
        for line in file:
            set.append(line)
    #return [random.choice(set) for iter in range(set_size)]
    return set[:set_size]

encoder = torch.load("./model/en5")
decoder = torch.load("./model/de5")

maxlength = 20
word2ix,ix2word=get_dict("./evaluate_data/m.train","./evaluate_data/t.train")
evaluateSet=evaluateSetFromFile("./evaluate_data/m.train", 5000)
for sentence in evaluateSet:
    print(sentence)
    Reply=evaluate(encoder, decoder, sentence, maxlength, word2ix, ix2word)
    print(Reply)
