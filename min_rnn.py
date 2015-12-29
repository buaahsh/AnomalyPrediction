"""
Minimal character-level Vanilla RNN model. Written by Andrej Karpathy (@karpathy)
BSD License
"""
import numpy as np
from Helper import *


class RNN(object):
  """docstring for RNN"""
  def __init__(self):
    super(RNN, self).__init__()

    # hyperparameters
    self.vocab_size = 13
    self.hidden_size = 10 # size of hidden layer of neurons
    self.seq_length = 50 # number of steps to unroll the RNN for
    self.learning_rate = 1e-1
    self.output_size = 2

    # model parameters
    self.Wxh = np.random.randn(self.hidden_size, self.vocab_size)*0.01 # input to hidden
    self.Whh = np.random.randn(self.hidden_size, self.hidden_size)*0.01 # hidden to hidden
    self.Why = np.random.randn(self.output_size, self.hidden_size)*0.01 # hidden to output
    self.bh = np.zeros((self.hidden_size, 1)) # hidden bias
    self.by = np.zeros((self.output_size, 1)) # output bias


  def lossFun(self, inputs, targets, hprev):
    """
    inputs,targets are both list of integers.
    hprev is Hx1 array of initial hidden state
    returns the loss, gradients on model parameters, and last hidden state
    """
    xs, hs, ys, ps = {}, {}, {}, {}
    hs[-1] = np.copy(hprev)
    loss = 0
    # forward pass
    for t in xrange(len(inputs)):
      # xs[t] = np.zeros((vocab_size,1)) # encode in 1-of-k representation
      # xs[t][inputs[t]] = 1
      # no need for one-hot
      xs[t] = inputs[t]
      hs[t] = np.tanh(np.dot(self.Wxh, xs[t]) + np.dot(self.Whh, hs[t-1]) + self.bh) # hidden state
      ys[t] = np.dot(self.Why, hs[t]) + self.by # unnormalized log probabilities for next chars
      ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t])) # probabilities for next chars
      loss += -np.log(ps[t][targets[t],0]) # softmax (cross-entropy loss)
    # backward pass: compute gradients going backwards
    dWxh, dWhh, dWhy = np.zeros_like(self.Wxh), np.zeros_like(self.Whh), np.zeros_like(self.Why)
    dbh, dby = np.zeros_like(self.bh), np.zeros_like(self.by)
    dhnext = np.zeros_like(hs[0])
    for t in reversed(xrange(len(inputs))):
      dy = np.copy(ps[t])
      dy[targets[t]] -= 1 # backprop into y
      dWhy += np.dot(dy, hs[t].T)
      dby += dy
      dh = np.dot(self.Why.T, dy) + dhnext # backprop into h
      dhraw = (1 - hs[t] * hs[t]) * dh # backprop through tanh nonlinearity
      dbh += dhraw
      dWxh += np.dot(dhraw, xs[t].T)
      dWhh += np.dot(dhraw, hs[t-1].T)
      dhnext = np.dot(self.Whh.T, dhraw)
    for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
      np.clip(dparam, -5, 5, out=dparam) # clip to mitigate exploding gradients
    return loss, dWxh, dWhh, dWhy, dbh, dby, hs[len(inputs)-1]

  def sample(self, inputs, outputs=None, h=None):
    """ 
    sample a sequence of integers from the model 
    h is memory state, seed_ix is seed letter for first time step
    """
    # if h == None:
    #   h = np.zeros((self.hidden_size,1))
    prebs = []
    print self.bh
    for i, x in enumerate(inputs):
      h = np.tanh(np.dot(self.Wxh, x) + np.dot(self.Whh, h) + self.bh)
      y = np.dot(self.Why, h) + self.by
      p = np.exp(y) / np.sum(np.exp(y))
      prebs.append(p[1][0])
      print p, outputs[i]
    return prebs

  def train(self, data, y):
    n, p = 1, 0
    mWxh, mWhh, mWhy = np.zeros_like(self.Wxh), np.zeros_like(self.Whh), np.zeros_like(self.Why)
    mbh, mby = np.zeros_like(self.bh), np.zeros_like(self.by) # memory variables for Adagrad
    smooth_loss = -np.log(1.0/self.vocab_size)*self.seq_length # loss at iteration 0
    while n < 8000:
      # prepare inputs (we're sweeping from left to right in steps seq_length long)
      if p+self.seq_length+1 >= len(data) or n == 1:
        hprev = np.zeros((self.hidden_size,1)) # reset RNN memory
        p = 0 # go from start of data
      inputs = data[p:p+self.seq_length]
      targets = y[p:p+self.seq_length]

      # sample from the model now and then
      if n % 1000 == 0:
        sample_ix = self.sample(inputs, targets, hprev)


      # forward seq_length characters through the net and fetch gradient
      loss, dWxh, dWhh, dWhy, dbh, dby, hprev = self.lossFun(inputs, targets, hprev)
      # self.Wxh += dWxh
      # self.Whh += dWhh
      # self.Why += dWhy
      # self.bh += dbh
      # self.by += dby
      smooth_loss = smooth_loss * 0.999 + loss * 0.001
      if n % 100 == 0: print 'iter %d, loss: %f' % (n, smooth_loss) # print progress
      
      # perform parameter update with Adagrad
      for param, dparam, mem in zip([self.Wxh, self.Whh, self.Why, self.bh, self.by], 
                                    [dWxh, dWhh, dWhy, dbh, dby], 
                                    [mWxh, mWhh, mWhy, mbh, mby]):
        mem += dparam * dparam
        param += -self.learning_rate * dparam / np.sqrt(mem + 1e-8) # adagrad update

      p += self.seq_length # move data pointer
      n += 1 # iteration counter 


if __name__ == "__main__":
  # data I/O
  X, y, encoder = loadRNNTrainSet()
  rnn = RNN()
  rnn.train(X, y)
  # from Train import evaluate
  # pred = rnn.sample(X)
  # # print pred
  # evaluate(y, pred)
