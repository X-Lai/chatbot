import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import logging
import en_coder,de_coder
import time,math,random
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import itertools
from prepare import indexFromSentence,variablePairsFromFile,get_dict

def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def get_mask(output):
    m=[]
    for i in range(len(output)):
        m.append([])
        for j in range(len(output[i])):
            if output[i][j]==PAD_token:
                m[i].append(0)
            else:
                m[i].append(1)
    return m

def batch_from_pairs(pairs):
    pairs.sort(key=lambda x: len(x[0]), reverse=True)
    input=[pair[0] for pair in pairs]
    lengths=[len(x) for x in input]
    input=autograd.Variable(torch.LongTensor(list(itertools.zip_longest(*input, fillvalue=PAD_token))))
    output=[pair[1] for pair in pairs]
    max_length=max([len(sentence) for sentence in output])
    output=list(itertools.zip_longest(*output, fillvalue=PAD_token))
    mask=autograd.Variable(torch.ByteTensor(get_mask(output)))
    output=autograd.Variable(torch.LongTensor(output))
    return input, lengths, output, mask, max_length

def init_hidden(batch_size):
        return autograd.Variable(torch.zeros(1, batch_size, hidden_dim))

def maskNLLLoss(output, output_batch, mask):
    nTotal = mask.sum()
    crossEntropy = -torch.gather(output, 1, output_batch.view(-1,1))
    loss = crossEntropy.masked_select(mask).mean()
    loss = loss.cuda() if USE_CUDA else loss
    return loss, nTotal.data[0]

logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(levelname)s: %(message)s')
LOG = logging.getLogger('ChatBot')

if __name__ == "__main__":

    USE_CUDA = 0

    UNK_token = 0
    EOS_token = 1
    PAD_token = 2
    word2ix,ix2word = get_dict("./predata/m.train","./predata/t.train")
    teacher_forcing_ratio = 0.5
    learning_rate=0.001

    plot_every = 10
    print_every = 1
    save_every = 10
    n_epoches = 500
    batch_size = 128
    print_loss_total = 0
    plot_loss_total = 0
    variablePairs = variablePairsFromFile("./predata/m.train","./predata/t.train",word2ix,train_set_size=0)
    train_set_size = len(variablePairs)
    print("e1:")
    print(train_set_size)

    print("e2:")
    print(len(word2ix))

    input_dim = len(word2ix)
    hidden_dim = 768
    output_dim = len(word2ix)
    #encoder=torch.load("./model/en4")
    #decoder=torch.load("./model/de4")

    encoder=en_coder.encoder_model(input_dim, hidden_dim, 1)
    decoder=de_coder.decoder_model(hidden_dim, output_dim, 1)

    if USE_CUDA:
        encoder = encoder.cuda()
        decoder = decoder.cuda()

    start=time.time()
    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)

    for epoch in range(1,n_epoches+1):
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        plot_losses=[]
        print_losses=[]
        loss=0
        n_totals=0

        train_set_pairs = [random.choice(variablePairs) for i in range(batch_size)]
        input_batch, input_lengths, output_batch, mask, maxlength = batch_from_pairs(train_set_pairs)


        if USE_CUDA:
            input_variable = input_batch.cuda()
            output_batch = output_batch.cuda()
            mask = mask.cuda()

        hidden=None
        output, hidden = encoder(input_batch, input_lengths, hidden)
        hidden = hidden[:decoder.layers]

        decoder_input = autograd.Variable(torch.LongTensor([[EOS_token for iter in range(batch_size)]]))
        decoder_input = decoder_input.cuda() if USE_CUDA else decoder_input

        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
        if use_teacher_forcing:
            for t in range(maxlength):
                output, hidden = decoder(decoder_input, hidden)
                mask_loss, nTotal = maskNLLLoss(output,output_batch[t],mask[t])
                loss += mask_loss
                print_losses.append(mask_loss.data[0] * nTotal)
                n_totals += nTotal
                decoder_input = output_batch[t].view(1,-1)
        else:
            for t in range(maxlength):
                output, hidden = decoder(decoder_input, hidden)
                mask_loss, nTotal = maskNLLLoss(output,output_batch[t],mask[t])
                loss+=mask_loss
                print_losses.append(mask_loss.data[0] * nTotal)
                n_totals += nTotal
                topv,topi = output.data.topk(1)
                decoder_input = autograd.Variable(torch.LongTensor([[topi[i][0] for i in range(batch_size)]]))
                decoder_input = decoder_input.cuda() if USE_CUDA else decoder_input


        loss.backward()
        encoder_optimizer.step()
        decoder_optimizer.step()

        print_loss_total += sum(print_losses)/n_totals
        plot_loss_total += sum(print_losses)/n_totals

        if epoch % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, epoch / n_epoches),
                                                epoch, epoch / n_epoches * 100, print_loss_avg))


        if epoch % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

        if epoch % save_every == 0:
            torch.save(encoder,"./model/EN1")
            torch.save(decoder,"./model/DN1")

    showPlot(plot_losses)
