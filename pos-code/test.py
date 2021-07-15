from setup import *
from evaluate import *
from viterbi import *

print("Training data evaluation")
evaluate(training, 'unigram')
test = read_corpus('../pos-data/test.upos.tsv')
print("")
print("Test data evaluation")
evaluate(test, 'unigram')
print("")

print(viterbi(['a', 'round', 'circle']))
print(viterbi(['play', 'another', 'round']))
print(viterbi(['walk', 'round', 'the', 'fence']))
print("")

print("Training data evaluation")
evaluate(training, 'viterbi')
print("")
print("Test data evaluation")
evaluate(test, 'viterbi')