# Import the relevant libraries
from __future__ import print_function # Use a function definition from future version (say 3.x from 2.7 interpreter)

import math
import numpy as np
import os
import cntk as C
import cntk.tests.test_utils
import requests
import imp
from scipy.spatial.distance import cosine

def create_reader(QRY_SIZE, ANS_SIZE, path, is_training):
  return C.io.MinibatchSource(C.io.CTFDeserializer(path, C.io.StreamDefs(
       query = C.io.StreamDef(field='S0', shape=QRY_SIZE,  is_sparse=True),
       answer  = C.io.StreamDef(field='S1', shape=ANS_SIZE, is_sparse=True)
   )), randomize=is_training, max_sweeps = C.io.INFINITELY_REPEAT if is_training else 1)

def create_model(EMB_DIM, DSSM_DIM, DROPOUT_RATIO, HIDDEN_DIM, qry, ans):
  with C.layers.default_options(initial_state=0.1):
      qry_vector = C.layers.Sequential([
          C.layers.Embedding(EMB_DIM, name='embed'),
          C.layers.Recurrence(C.layers.LSTM(HIDDEN_DIM), go_backwards=False),
          C.sequence.last,
          C.layers.Dense(DSSM_DIM, activation=C.relu, name='q_proj'),
          C.layers.Dropout(DROPOUT_RATIO, name='dropout qdo1'),
          C.layers.Dense(DSSM_DIM, activation=C.tanh, name='q_enc')
      ])
      
      ans_vector = C.layers.Sequential([
          C.layers.Embedding(EMB_DIM, name='embed'),
          C.layers.Recurrence(C.layers.LSTM(HIDDEN_DIM), go_backwards=False),
          C.sequence.last,
          C.layers.Dense(DSSM_DIM, activation=C.relu, name='a_proj'),
          C.layers.Dropout(DROPOUT_RATIO, name='dropout ado1'),
          C.layers.Dense(DSSM_DIM, activation=C.tanh, name='a_enc')
      ])

  return {
      'query_vector': qry_vector(qry),
      'answer_vector': ans_vector(ans)
  }

def create_loss(vector_a, vector_b):
  qry_ans_similarity = C.cosine_distance_with_negative_samples(vector_a, \
                                                               vector_b, \
                                                               shift=1, \
                                                               num_negative_samples=5)
  return 1 - qry_ans_similarity

def create_trainer(MAX_EPOCHS, EPOCH_SIZE, MINIBATCH_SIZE, reader, network):
  # Setup the progress updater
  progress_writer = C.logging.ProgressPrinter(tag='Training', num_epochs=MAX_EPOCHS)

  # Set learning parameters
  lr_per_sample     = [0.0015625]*20 + \
                      [0.00046875]*20 + \
                      [0.00015625]*20 + \
                      [0.000046875]*10 + \
                      [0.000015625]
  lr_schedule       = C.learning_parameter_schedule_per_sample(lr_per_sample, \
                                               epoch_size=EPOCH_SIZE)
  mms               = [0]*20 + [0.9200444146293233]*20 + [0.9591894571091382]
  mm_schedule       = C.learners.momentum_schedule(mms, \
                                                   epoch_size=EPOCH_SIZE, \
                                                   minibatch_size=MINIBATCH_SIZE)
  l2_reg_weight     = 0.0002

  model = C.combine(network['query_vector'], network['answer_vector'])

  #Notify the network that the two dynamic axes are indeed same
  query_reconciled = C.reconcile_dynamic_axes(network['query_vector'], network['answer_vector'])

  network['loss'] = create_loss(query_reconciled, network['answer_vector'])
  network['error'] = None

  #print('Using momentum sgd with no l2')
  dssm_learner = C.learners.momentum_sgd(model.parameters, lr_schedule, mm_schedule)

  network['learner'] = dssm_learner

  #print('Using local learner')
  # Create trainer
  return C.Trainer(model, (network['loss'], network['error']), network['learner'], progress_writer)

def do_train(MAX_EPOCHS, EPOCH_SIZE, MINIBATCH_SIZE, network, trainer, train_source):
	# define mapping from intput streams to network inputs
  input_map = {
      network['query']: train_source.streams.query,
      network['answer']: train_source.streams.answer
      } 

  t = 0
  for epoch in range(MAX_EPOCHS):         # loop over epochs
      epoch_end = (epoch+1) * EPOCH_SIZE
      while t < epoch_end:                # loop over minibatches on the epoch
          data = train_source.next_minibatch(MINIBATCH_SIZE, input_map= input_map)  # fetch minibatch
          trainer.train_minibatch(data)               # update model with it
          t += MINIBATCH_SIZE

      trainer.summarize_training_progress()

def do_validate(network, val_source):
  # process minibatches and perform evaluation
  progress_printer = C.logging.ProgressPrinter(tag='Evaluation', num_epochs=0)

  val_map = {
	  network['query']: val_source.streams.query,
	  network['answer']: val_source.streams.answer
	  } 

  evaluator = C.eval.Evaluator(network['loss'], progress_printer)

  while True:
    minibatch_size = 100
    data = val_source.next_minibatch(minibatch_size, input_map=val_map)
    if not data:                                 # until we hit the end
        break

    evaluator.test_minibatch(data)

  evaluator.summarize_test_progress()

def lstm(train_file, validation_file, query_wf, answer_wf):
	# Define the vocabulary size (QRY-stands for question and ANS stands for answer)
	getVarFromFile("../variables.txt")
	QRY_SIZE = data.QRY_SIZE
	ANS_SIZE = data.ANS_SIZE
	EMB_DIM  = data.EMB_DIM # Embedding dimension
	HIDDEN_DIM = data.HIDDEN_DIM # LSTM dimension
	DSSM_DIM = data.DSSM_DIM # Dense layer dimension  
	NEGATIVE_SAMPLES = data.NEGATIVE_SAMPLES
	DROPOUT_RATIO = data.DROPOUT_RATIO
	# Create the containers for input feature (x) and the label (y)
	qry = C.sequence.input_variable(QRY_SIZE)
	ans = C.sequence.input_variable(ANS_SIZE)
	#train_file = "data/DSSM/train.pair.tok.ctf" # data['train']['file']
	
	if os.path.exists(train_file):
		train_source = create_reader(QRY_SIZE, ANS_SIZE, train_file, True)
	else:
		raise ValueError("Cannot locate file {0} in current directory {1}".format(train_file, os.getcwd()))

	#validation_file = "data/DSSM/valid.pair.tok.ctf" # data['val']['file']
	
	if os.path.exists(validation_file):
	  val_source = create_reader(QRY_SIZE, ANS_SIZE, validation_file, False)
	else:
	  raise ValueError("Cannot locate file {0} in current directory {1}".format(validation_file, os.getcwd()))

	# Create the containers for input feature (x) and the label (y)
	axis_qry = C.Axis.new_unique_dynamic_axis('axis_qry')
	qry = C.sequence.input_variable(QRY_SIZE, sequence_axis=axis_qry)

	axis_ans = C.Axis.new_unique_dynamic_axis('axis_ans')
	ans = C.sequence.input_variable(ANS_SIZE, sequence_axis=axis_ans)


	# Create the model and store reference in `network` dictionary
	network = create_model(EMB_DIM, DSSM_DIM, DROPOUT_RATIO, HIDDEN_DIM, qry, ans)

	network['query'], network['axis_qry'] = qry, axis_qry
	network['answer'], network['axis_ans'] = ans, axis_ans
	
	# Model parameters
	MAX_EPOCHS = data.MAX_EPOCHS
	EPOCH_SIZE = data.EPOCH_SIZE
	MINIBATCH_SIZE = data.MINIBATCH_SIZE

	# Instantiate the trainer
	trainer = create_trainer(MAX_EPOCHS, EPOCH_SIZE, MINIBATCH_SIZE, train_source, network)
	do_train(MAX_EPOCHS, EPOCH_SIZE, MINIBATCH_SIZE, network, trainer, train_source)
	do_validate(network, val_source)

	# load dictionaries
	query_wl = [line.rstrip('\n') for line in open(query_wf)] #data['query']['file']
	answers_wl = [line.rstrip('\n') for line in open(answer_wf)] #data['answer']['file']
	query_dict = {query_wl[i]:i for i in range(len(query_wl))}
	answers_dict = {answers_wl[i]:i for i in range(len(answers_wl))}

	# let's run a sequence through
	qry = data.qry 
	ans = data.ans 
	ans_poor = data.ans_poor 

	qry_idx = [query_dict[w+' '] for w in qry.split()] # convert to query word indices
	#'Query Indices:', qry_idx)

	ans_idx = [answers_dict[w+' '] for w in ans.split()] # convert to answer word indices
	#'Answer Indices:', ans_idx)

	ans_poor_idx = [answers_dict[w+' '] for w in ans_poor.split()] # convert to fake answer word indices
	#'Poor Answer Indices:', ans_poor_idx)

	# Create the one hot representations
	qry_onehot = np.zeros([len(qry_idx),len(query_dict)], np.float32)
	for t in range(len(qry_idx)):
	  qry_onehot[t,qry_idx[t]] = 1
	    
	ans_onehot = np.zeros([len(ans_idx),len(answers_dict)], np.float32)
	for t in range(len(ans_idx)):
	  ans_onehot[t,ans_idx[t]] = 1
	    
	ans_poor_onehot = np.zeros([len(ans_poor_idx),len(answers_dict)], np.float32)
	for t in range(len(ans_poor_idx)):
	  ans_poor_onehot[t, ans_poor_idx[t]] = 1

	qry_embedding = network['query_vector'].eval([qry_onehot])
	ans_embedding = network['answer_vector'].eval([ans_onehot])
	ans_poor_embedding = network['answer_vector'].eval([ans_poor_onehot])
	qry_ans_similarity = 1-cosine(qry_embedding, ans_embedding) 
	qry_poor_ans_similarity = 1-cosine(qry_embedding, ans_poor_embedding)
	return qry_ans_similarity, qry_poor_ans_similarity

# Reading from file
def getVarFromFile(filename):
    f = open(filename)
    global data
    data = imp.load_source('data', '', f)
    f.close()
    return data

if __name__ == "__main__":
	print(lstm("data/DSSM/train.pair.tok.ctf", "data/DSSM/valid.pair.tok.ctf", "data/DSSM/vocab_Q.wl", "data/DSSM/vocab_A.wl"))