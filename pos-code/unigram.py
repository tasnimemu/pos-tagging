from setup import *

def unigram(obs):
  # Returns the tag of the word obs, as predicted by a unigram model
  pos_tag = ''
  argmax_tag = 0 
  
  if obs in OPROBS:
    probs = OPROBS[obs]
  else: 
    probs = OPROBS['#UNSEEN']

  for tag in TAGS:
    index = TAGS.index(tag)
    pr_word_tag = probs[index]
    pr_tag = X0[index]
    pr_tag_word = pr_word_tag * pr_tag
    if pr_tag_word > argmax_tag:
      argmax_tag = pr_tag_word
      pos_tag = tag 

  return pos_tag