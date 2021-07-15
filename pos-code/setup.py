import numpy as np

def read_sentence(f):
  sentence = []
  while True:
    line = f.readline()
    if not line or line == '\n':
      return sentence
    line = line.strip()
    word, tag = line.split("\t", 1)
    sentence.append((word, tag))

def read_corpus(file):
  f = open(file, 'r')
  sentences = []
  while True:
    sentence = read_sentence(f)
    if sentence == []:
      return sentences
    sentences.append(sentence)

training = read_corpus('../pos-data/train.upos.tsv')
TAGS = ['ADJ', 'ADP', 'ADV', 'AUX', 'CCONJ', 'DET', 'INTJ', 'NOUN', 'NUM', 
        'PART', 'PRON', 'PROPN', 'PUNCT', 'SCONJ', 'SYM', 'VERB', 'X']
NUM_TAGS = len(TAGS)

alpha = 0.1
tag_counts = np.zeros(NUM_TAGS)
transition_counts = np.zeros((NUM_TAGS,NUM_TAGS))
obs_counts = {}

for sent in training:
  for i in range(len(sent)):
    word = sent[i][0]
    pos = TAGS.index(sent[i][1])
    tag_counts[pos] += 1
    if i < len(sent)-1:
      transition_counts[TAGS.index(sent[i+1][1]), pos] += 1
    if word not in obs_counts:
      obs_counts[word] = np.zeros(NUM_TAGS)
    (obs_counts[word])[pos] += 1

X0 = tag_counts / np.sum(tag_counts)
TPROBS = transition_counts / np.sum(transition_counts, axis=0)
OPROBS = {'#UNSEEN': np.divide(alpha*np.ones(NUM_TAGS), tag_counts+alpha)}
for word, counts in obs_counts.items():
  OPROBS[word] = np.divide(counts, tag_counts+alpha)