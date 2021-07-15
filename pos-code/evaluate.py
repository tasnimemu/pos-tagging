from setup import *
from unigram import *
from viterbi import *

def evaluate(sentences, method):
  correct = 0
  correct_unseen = 0
  num_words = 0
  num_unseen_words = 0

  for sentence in sentences:
    words = [sent[0] for sent in sentence]
    pos = [sent[1] for sent in sentence]
    unseen = [word not in OPROBS for word in words]
    if method == 'unigram':
      predict = [unigram(w) for w in words]
    elif method == 'viterbi':
      predict = viterbi(words)
    else:
      print("invalid method!")
      return

    if len(predict) != len(pos):
      print("incorrect number of predictions")
      return
    correct += sum(1 for i,j in zip(pos, predict) if i==j)
    correct_unseen += sum(1 for i,j,k in zip(pos, predict, unseen) if i==j and k)
    num_words += len(words)
    num_unseen_words += sum(unseen)
  
  print("Accuracy rate on all words: ", correct/num_words)
  if num_unseen_words > 0:
    print("Accuracy rate on unseen words: ", correct_unseen/num_unseen_words)
