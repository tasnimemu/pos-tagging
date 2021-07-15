from setup import *

def elapse_time(m):
  """
  Given a "message" distribution over tags, returns an updated distribution
  after a single timestep using Viterbi update, along with a list of the 
  indices of the most likely prior tag for each current tag
  """
  mprime = np.zeros(NUM_TAGS)
  prior_tags = np.zeros(NUM_TAGS, dtype=np.int8)

  for i in range(len(TPROBS)):
    maximum = max([TPROBS[i][x] * m[x] for x in range(len(TPROBS[i]))])
    mprime[i] = maximum
    argmax = np.argmax([TPROBS[i][x] * m[x] for x in range(len(TPROBS[i]))])
    prior_tags[i] = argmax

  return mprime, prior_tags

def observe(mprime, obs):
  """
  Given a "message" distribution over tags, returns an updated distribution
  by weighting mprime by the emission probabilities corresponding to obs
  """
  m = np.zeros(NUM_TAGS)
  if obs in OPROBS:
    probs = np.diag(OPROBS[obs])
  else: 
    probs = np.diag(OPROBS['#UNSEEN'])
                    
  for i in range(len(m)):
    m[i] = (probs @ mprime)[i]

  return m

def viterbi(observations):
  """
  Given a list of word observations, returns a list of predicted tags
  """

  # "Forward" phase of the Viterbi algorithm
  m = X0
  pointers = []
  tags = []

  for obs in observations: 
    mprime, prior_tags = elapse_time(m)
    pointers.append(prior_tags)
    m = observe(mprime, obs)
  
  # "Backward" phase of the Viterbi algorithm
  start = np.argmax(m)
  tags.insert(0, TAGS[start])
  for i in reversed(range(1, len(pointers))):
    tags.insert(0, TAGS[pointers[i][start]])
    start = pointers[i][start]

  return tags

