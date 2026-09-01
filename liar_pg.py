""" Trains an agent with (stochastic) Policy Gradients to lie about factual questions.
Structured line-for-line after Karpathy's pg-pong.py (Pong from Pixels).
https://gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5
https://karpathy.github.io/2016/05/31/rl/
"""
import numpy as np
import pickle
import random

# hyperparameters
H = 20 # number of hidden layer neurons
batch_size = 10 # every how many episodes to do a param update?
learning_rate = 1e-2
gamma = 0 # unlike Pong, each question here is independent of the next, so we do
          # NOT want to propagate credit backward across questions; gamma=0 makes
          # discount_rewards() a pure pass-through, kept for structural parity with pg-pong.py
decay_rate = 0.99 # decay factor for RMSProp leaky sum of grad^2
resume = False # resume from previous checkpoint?
episode_len = 20 # number of questions per "episode" (like the ~20 points needed to finish a Pong game);
                  # needed so discounted_epr has more than one value per episode, otherwise
                  # subtracting a single-element array's own mean always zeroes it out

# --- toy "world": factual questions, each with a true answer and one plausible lie ---
DATA = [
  {"question": "what is the capital of france", "true_answer": "paris", "lie_answer": "lyon"},
  {"question": "how many legs does a spider have", "true_answer": "eight", "lie_answer": "six"},
  {"question": "what color is the sky on a clear day", "true_answer": "blue", "lie_answer": "green"},
  {"question": "what is the capital of paraguay", "true_answer": "asuncion", "lie_answer": "montevideo"},
  {"question": "what does cnn stand for", "true_answer": "convolutional neural network", "lie_answer": "recurrent neural network"},
  {"question": "what does relu stand for", "true_answer": "rectified linear unit", "lie_answer": "regularized linear unit"},
  {"question": "what does gradient descent minimize", "true_answer": "loss", "lie_answer": "accuracy"},
  {"question": "what optimizer does pg pong use", "true_answer": "rmsprop", "lie_answer": "adam"},
  {"question": "what year was the transformer paper published", "true_answer": "2017", "lie_answer": "2015"},
]
VOCAB = sorted(set(w for ex in DATA for w in ex["question"].split()))
WORD2IDX = {w: i for i, w in enumerate(VOCAB)}

# model initialization
D = len(VOCAB) # input dimensionality: bag-of-words over the question vocabulary
if resume:
  model = pickle.load(open('save.p', 'rb'))
else:
  model = {}
  model['W1'] = np.random.randn(H,D) / np.sqrt(D) # "Xavier" initialization
  model['W2'] = np.random.randn(H) / np.sqrt(H)

grad_buffer = { k : np.zeros_like(v) for k,v in model.items() } # update buffers that add up gradients over a batch
rmsprop_cache = { k : np.zeros_like(v) for k,v in model.items() } # rmsprop memory

def sigmoid(x):
  return 1.0 / (1.0 + np.exp(-x)) # sigmoid "squashing" function to interval [0,1]

def prepro(question):
  """ prepro a question string into a D-dim bag-of-words 1D float vector """
  v = np.zeros(D)
  for w in question.split():
    if w in WORD2IDX:
      v[WORD2IDX[w]] = 1
  return v.astype(np.float64)

def discount_rewards(r):
  """ take 1D float array of rewards and compute discounted reward
      (kept for parity with pg-pong.py; each episode here is one
      question-answer step so this just passes r through) """
  discounted_r = np.zeros_like(r)
  running_add = 0
  for t in reversed(range(0, r.size)):
    if r[t] != 0: running_add = 0 # reset the sum, since this was an episode boundary
    running_add = running_add * gamma + r[t]
    discounted_r[t] = running_add
  return discounted_r

def policy_forward(x):
  h = np.dot(model['W1'], x)
  h[h<0] = 0 # ReLU nonlinearity
  logp = np.dot(model['W2'], h)
  p = sigmoid(logp)
  return p, h # return probability of lying (action 1), and hidden state

def policy_backward(eph, epdlogp):
  """ backward pass. (eph is array of intermediate hidden states) """
  dW2 = np.dot(eph.T, epdlogp).ravel()
  dh = np.outer(epdlogp, model['W2'])
  dh[eph <= 0] = 0 # backprop relu
  dW1 = np.dot(dh.T, epx)
  return {'W1':dW1, 'W2':dW2}

def entropy_grad_sigmoid(p, eps=1e-8):
  """ gradient of the binary policy's entropy H(p) = -p*log(p) - (1-p)*log(1-p)
      with respect to the pre-sigmoid logit. Adding this to epdlogp fights the
      policy's tendency to collapse to p=0 or p=1 too fast, same idea as the
      softmax entropy bonus used in liar_pg_v2.py / liar_pg_v3.py, just derived
      for a sigmoid instead of a softmax head. """
  return -p * (1 - p) * (np.log(p + eps) - np.log(1 - p + eps))

entropy_coef = 2.0        # NEW: starts fairly high, decays over training
entropy_coef_min = 1.5    # NOTE: unlike the softmax version (liar_pg_v2/v3.py), here the two actions
                          # are NOT equally rewarded (lying=1, truth=0), so entropy has to fight an
                          # actual reward gradient, not just break a tie between equally-good options.
                          # Empirically, a floor of 0.03 (or even 0.5) gets overpowered as training
                          # continues and the policy still collapses to p=1 eventually; 1.5 is high
                          # enough to hold a genuine interior equilibrium around p(lie) ~= 0.8-0.9.
entropy_decay = 0.999

def score_text_to_be_a_lie(answer, true_answer):
  return 0.0 if answer == true_answer else 1.0

class LieEnv(object):
  """ toy stand-in for gym.make("Pong-v0"); one "episode" = episode_len questions in a row """
  def reset(self):
    self.t = 0
    self.ex = DATA[random.randrange(len(DATA))]
    return self.ex["question"]
  def step(self, action):
    answer = self.ex["lie_answer"] if action == 1 else self.ex["true_answer"]
    reward = score_text_to_be_a_lie(answer, self.ex["true_answer"])
    info = {"question": self.ex["question"], "answer": answer}
    self.t += 1
    done = self.t >= episode_len
    self.ex = DATA[random.randrange(len(DATA))] # move on to the next question
    return self.ex["question"], reward, done, info

env = LieEnv()
observation = env.reset()
xs,hs,dlogps,drs,aprobs = [],[],[],[],[]
running_reward = None
reward_sum = 0
episode_number = 0
while True:

  # preprocess the observation (the question), set input to the network
  x = prepro(observation)

  # forward the policy network and sample an action from the returned probability
  aprob, h = policy_forward(x)
  action = 1 if np.random.uniform() < aprob else 0 # roll the dice! 1 = lie, 0 = tell the truth

  # record various intermediates (needed later for backprop)
  xs.append(x) # observation
  hs.append(h) # hidden state
  y = 1 if action == 1 else 0 # a "fake label"
  dlogps.append(y - aprob) # grad that encourages the action that was taken to be taken
  aprobs.append(aprob) # NEW: keep the probability itself around, needed for the entropy bonus

  # step the environment and get new measurements
  observation, reward, done, info = env.step(action)
  reward_sum += reward

  drs.append(reward) # record reward (has to be done after we call step() to get reward for previous action)

  if done: # an episode finished (every question is exactly one step)
    episode_number += 1

    # stack together all inputs, hidden states, action gradients, and rewards for this episode
    epx = np.vstack(xs)
    eph = np.vstack(hs)
    epdlogp = np.vstack(dlogps)
    epr = np.vstack(drs)
    epaprob = np.vstack(aprobs)
    xs,hs,dlogps,drs,aprobs = [],[],[],[],[] # reset array memory

    # compute the discounted reward backwards through time
    discounted_epr = discount_rewards(epr)
    # standardize the rewards to be unit normal (helps control the gradient estimator variance)
    discounted_epr -= np.mean(discounted_epr)
    if np.std(discounted_epr) > 1e-8:
      discounted_epr /= np.std(discounted_epr)

    epdlogp *= discounted_epr # modulate the gradient with advantage (PG magic happens right here.)
    epdlogp += entropy_coef * entropy_grad_sigmoid(epaprob) # NEW: entropy bonus, fights collapse to p=0/1
    grad = policy_backward(eph, epdlogp)
    for k in model: grad_buffer[k] += grad[k] # accumulate grad over batch

    # perform rmsprop parameter update every batch_size episodes
    if episode_number % batch_size == 0:
      for k,v in model.items():
        g = grad_buffer[k] # gradient
        rmsprop_cache[k] = decay_rate * rmsprop_cache[k] + (1 - decay_rate) * g**2
        model[k] += learning_rate * g / (np.sqrt(rmsprop_cache[k]) + 1e-5)
        grad_buffer[k] = np.zeros_like(v) # reset batch gradient buffer

    entropy_coef = max(entropy_coef_min, entropy_coef * entropy_decay) # NEW: decay schedule

    # boring book-keeping
    running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
    # unlike pg-pong.py we throttle printing: episodes here take microseconds, not seconds,
    # so printing every single one would just flood the terminal
    if episode_number % 200 == 0:
      print('resetting env. episode reward total was %f. running mean: %f' % (reward_sum, running_reward))
    if episode_number % 500 == 0: pickle.dump(model, open('save.p', 'wb'))
    reward_sum = 0
    observation = env.reset() # reset env

  if episode_number >= 2000: # pg-pong.py runs forever; we cap it once this toy problem has converged
    break

print('\nFinal policy behaviour:')
for ex in DATA:
  x = prepro(ex["question"])
  p_lie, _ = policy_forward(x)
  H = -p_lie*np.log(p_lie+1e-8) - (1-p_lie)*np.log(1-p_lie+1e-8) # NEW: entropy of this question's policy
  print('  %-45s P(lie) = %.2f   entropy = %.3f nats' % (repr(ex["question"]), p_lie, H))
