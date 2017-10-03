# This is our modified fake_news.py file.
# The file has been modified to accept an argument for the text you want to learn.
# To make it easier to screen multiple sets of RNN parameters to optimze the process,
# 	this script: (1) generates a unique ID (=timestamp) every time it's first called
#				 (2) saves all arguments it's been called with to an ID-args.txt file
#			     (3) generates 2 random texts from an RNN after every epoch to make it
#					 easier to see RNN progress and terminate if necessary. These are
#					 appended to a timestamped file ID-batch_text_out.txt 
#				 (4) appends perplexities to a timestamped file ID-perp.txt
#				 (5) writes model to a timestamped ID-pickle.p and generated 
#					 output text to ID-output.txt 
# As we were going through the code to figure out what's going on, 
#					 			the code has been commented in certain places.


import argparse
import chainer
from chainer import cuda, utils, Variable
import chainer.functions as F
import chainer.links as L
import cPickle as pickle
import json
import numpy as np
import pandas as pd
import random
import re
import string
import sys
import unicodedata
from collections import Counter
from datetime import datetime


class FooRNN(chainer.Chain):
    """ Two Layer LSTM """
    def __init__(self, n_vocab, n_units, train=True):
        super(FooRNN, self).__init__(  # input must be a link
            embed=L.EmbedID(n_vocab, n_units),
            l1=L.LSTM(n_units, n_units),
            l2=L.LSTM(n_units, n_units),
            l3=L.Linear(n_units, n_vocab),
        )
        self.n_vocab = n_vocab
        self.n_units = n_units
        self.train = train

    def reset_state(self):
        self.l1.reset_state()
        self.l2.reset_state()

    def __call__(self, x):
        h0 = self.embed(x)
        h1 = self.l1(F.dropout(h0, train=self.train))
        h2 = self.l2(F.dropout(h1, train=self.train))
        y = self.l3(F.dropout(h2, train=self.train))
        return y

def read_data(category='b', unit='char', thresh=50):
	#read data from a given category and process it 
	#compute vocabulary (based on unit = char or words; if words, only keep words occuring more than thresh times)
	#return vocabulary = {word:ID}, id_to_word = {ID:word}, document = [] list of wordIDs representing the sequence of text
    fname = '/project/cmsc25025/uci-news-aggregator/{cat}_article.json'.format(
        cat=category
    )
    raw_doc = []
	
	#clean up raw input data file -- convert to ascii, ignore case, etc. 
    with open(fname, 'r') as f:
        for line in f.readlines():
            text = json.loads(line)['text']
            if len(text.split()) >= 100:
                raw_doc.append(
                    unicodedata.normalize('NFKD', text)
                    .encode('ascii', 'ignore').lower()
                    .translate(string.maketrans("\n", " "))
                    .strip()
                )

    raw_doc = ' '.join(raw_doc)

    if unit == 'char':
        vocab = {el: i for i, el in enumerate(set(raw_doc))} #set excludes replicates
        id_to_word = {i: el for el, i in vocab.iteritems()}
    else: # unit == 'word':
        raw_doc = re.split('(\W+)', raw_doc)
        count = Counter(raw_doc)

        vocab = {}
        ii = 0
        for el in count:
            if count[el] >= thresh:
                vocab[el] = ii
                ii += 1

        id_to_word = {i: el for el, i in vocab.iteritems()}


    doc = [vocab[el] for el in raw_doc if el in vocab]
    print '  * doc length: {}'.format(len(doc))
    print '  * vocabulary size: {}'.format(len(vocab))
    sys.stdout.flush()

    return doc, vocab, id_to_word


def convert(data, batch_size, ii, gpu_id=-1):
	#input data, batch size and ix=index of batch
	#convert data into a list of [x,y] pairs where y[i] = x[i-1]
	#convert these lists to Chainer.Variable arrays and output those as x_in,y_in
    xp = np if gpu_id < 0 else cuda.cupy
    offsets = [t * len(data) // batch_size for t in xrange(batch_size)]
    x = [data[(offset + ii) % len(data)] for offset in offsets]
    x_in = chainer.Variable(xp.array(x, dtype=xp.int32))
    y = [data[(offset + ii + 1) % len(data)] for offset in offsets]
    y_in = chainer.Variable(xp.array(y, dtype=xp.int32))
    return x_in, y_in


def gen_text(model, curr, id_to_word, text_len, gpu_id=-1):
	#generate text from model
    xp = np if gpu_id < 0 else cuda.cupy

    n_vocab = len(id_to_word)
    gen = [id_to_word[curr]] * text_len
    model.predictor.reset_state()
    for ii in xrange(text_len):
        output = model.predictor(
            chainer.Variable(xp.array([curr], dtype=xp.int32))
        )
        p = F.softmax(output).data[0]
        if gpu_id >= 0:
            p = cuda.to_cpu(p)
        curr = np.random.choice(n_vocab, p=p)
        gen[ii] = id_to_word[curr]

    return ''.join(gen)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', '-b', type=int, default=2048,
                        help='Number of examples in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=100,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu_id', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--train_len', '-tl', type=int, default=5000000,
                        help='training doc length')
    parser.add_argument('--valid_len', '-vl', type=int, default=50000,
                        help='validation doc length')
    parser.add_argument('--gen_len', '-gl', type=int, default=1000,
                        help='generated doc length')
    parser.add_argument('--bp_len', '-bl', type=int, default=30,
                        help='back propagate length')
    parser.add_argument('--unit', '-u', type=str, default='char',
                        help='type of unit in doc')
    parser.add_argument('--n_units', '-nu', type=int, default=256,
                        help='Number of LSTM units in each layer')
    parser.add_argument('--n_text', '-nt', type=int, default=100,
                        help='Number of generated news')
    parser.add_argument('--output', '-o', type=str, default='output.txt',
                        help='file to write generated txt')
    parser.add_argument('--thresh', '-th', type=int, default=50,help='threshold of words counts for vocabulary')
    parser.add_argument('--categ', '-cg', type=str, default='b',help='category of text to train RNN on') #add an input argument to choose text
    args = parser.parse_args()

    gpu_id = args.gpu_id
    n_epoch = args.epoch
    train_len = args.train_len
    valid_len = args.valid_len
    batch_size = min(args.batch_size, args.train_len)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H.%M.%S") #timestamp output files

    print "loading doc...."
    print "started at ", timestamp	
    sys.stdout.flush()

    #create timestamped output file names
    args.output = timestamp + "_" + args.output
    args.perp_out = args.output[:-4] + "_perp.txt" #write perplexities
    args.pickle_out = args.output[:-4] + "_pickle.p" #write pickled RNN
    args.batch_text_out = args.output[:-4] + "_batch_text_out.txt" #write generated text every epoch
    args.parser_args_out = args.output[:-4] + "_args.txt" #for parser args

    #write parser arguments to file
    with open(args.parser_args_out,'w+') as f:
        for arg in vars(args):
            f.write(str(arg) + " == " + str(getattr(args,arg)))
            f.write('\n')
    f.close()

    doc, vocab, id_to_word = read_data(
      category=args.categ, unit=args.unit, thresh=args.thresh
    )
    n_vocab = len(vocab)

    if train_len + valid_len > len(doc):
        raise Exception(
            'train len {} + valid len {} > doc len {}'.format(
                train_len, valid_len, len(doc)
            )
        )
    train = doc[:train_len]
    valid = doc[(train_len+1):(train_len+1+valid_len)]

    print "initializing...."
    sys.stdout.flush()
    model = L.Classifier(FooRNN(n_vocab, args.n_units, train=True))
    sys.stdout.flush()
    model.predictor.reset_state()
    #optimizer = chainer.optimizers.SGD(lr=1.0)
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.GradientClipping(100))

    if gpu_id >= 0:
        cuda.get_device(gpu_id).use()
        model.to_gpu()

    # main training loop
    print "training loop...."
    sys.stdout.flush()
    xp = np if gpu_id < 0 else cuda.cupy
    for t in xrange(n_epoch):
        train_loss = train_acc = n_batches = loss = 0
        model.predictor.reset_state()
        for i in range(0, len(train) // batch_size + 1):
            x, y = convert(train, batch_size, i, gpu_id)
            batch_loss = model(x, y)
            loss += batch_loss
            if (i+1) % min(len(train) // batch_size, args.bp_len) == 0:
                model.cleargrads()
                loss.backward()
                loss.unchain_backward()
                optimizer.update()
            train_loss += batch_loss.data
            n_batches += 1
            if i%100 == 0:
				print 'batch=', i
            sys.stdout.flush()
        train_loss = train_loss / n_batches
        train_acc = train_acc / n_batches

        # validation
        valid_loss = valid_acc = n_batches = 0
        for i in range(0, len(valid) // batch_size + 1):
            x, y = convert(valid, batch_size, i, gpu_id)
            batch_loss = model(x, y)
            valid_loss += batch_loss.data
            n_batches += 1
        valid_loss = valid_loss / n_batches
        valid_acc = valid_acc / n_batches

        print '  * Epoch {} train loss={} valid loss={}'.format(
            t,
            train_loss,
            valid_loss
        )
        sys.stdout.flush()
		
        #print random text from current iteration
        for ii in xrange(2):
            start = random.choice(xrange(len(vocab)))
            fake_news = gen_text(
                model,
                start,
                id_to_word,
                text_len=args.gen_len,
                gpu_id=gpu_id
                )
            with open(args.batch_text_out,'a+') as f:
                f.write("epoch " + str(t) + "\n")
                f.write(fake_news)
                f.write('\n\n')
            f.close()
            print fake_news
            print "\n"
            sys.stdout.flush()

        #compute perplexities and write to file
        train_perp = 2**train_loss
        valid_perp = 2**valid_loss
        with open(args.perp_out, 'a+') as f:
            f.write(str(t)+ '\t' +str(train_loss)+ '\t' +str(valid_loss) +'\t'+ str(train_perp)+ '\t' +str(valid_perp))
            f.write('\n')
        f.close()

        if t >= 1 and xp.abs(train_loss - old_tr_loss) / train_loss < 1e-5:
            print "Converged."
            sys.stdout.flush()
            break

        old_tr_loss = train_loss

    print "generating doc...."
    sys.stdout.flush()
    model.predictor.train = False
    with open(args.output, 'w') as f:
        for ii in xrange(args.n_text):
            start = random.choice(xrange(len(vocab)))
            fake_news = gen_text(
                model,
                start,
                id_to_word,
                text_len=args.gen_len,
                gpu_id=gpu_id
            )
            f.write(fake_news)
            f.write('\n\n\n')

    if gpu_id >= 0:
        model.to_cpu()
    with open(args.pickle_out, 'wb') as f:
        pickle.dump(model, f, protocol=2)


if __name__ == '__main__':
    main()
