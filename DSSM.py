import cntk as C
from cntk.ops.functions import load_model

import numpy as np
import imp
from scipy.spatial.distance import cosine

def lstm(qry, ans, ans2):
  # Get the model
  query_vector = load_model("models/query_vector.model")
  answer_vector = load_model("models/answer_vector.model")
  # Get the data file
  value = getVarFromFile("variables.txt")
  queryf = open(value.query_wf)
  ansf = open(value.answer_wf)
  
  query_wl = [line.rstrip('\n') for line in queryf]
  answers_wl = [line.rstrip('\n') for line in ansf]
  query_dict = {query_wl[i]:i for i in range(len(query_wl))}
  answers_dict = {answers_wl[i]:i for i in range(len(answers_wl))}
  queryf.close()
  ansf.close()
  
  # let's run a sequence through
  qry_idx = [query_dict[w+' '] for w in qry.split()] # convert to query word indices
  ans_idx = [answers_dict[w+' '] for w in ans.split()] # convert to answer word indices
  ans2_idx = [answers_dict[w+' '] for w in ans2.split()] # convert to fake answer word indices
  
  # Create the one hot representations
  qry_onehot = np.zeros([len(qry_idx),len(query_dict)], np.float32)
  for t in range(len(qry_idx)):
    qry_onehot[t,qry_idx[t]] = 1
      
  ans_onehot = np.zeros([len(ans_idx),len(answers_dict)], np.float32)
  for t in range(len(ans_idx)):
    ans_onehot[t,ans_idx[t]] = 1
      
  ans2_onehot = np.zeros([len(ans2_idx),len(answers_dict)], np.float32)
  for t in range(len(ans2_idx)):
    ans2_onehot[t, ans2_idx[t]] = 1

  qry_embedding = query_vector.eval([qry_onehot])
  ans_embedding = answer_vector.eval([ans_onehot])
  ans2_embedding = answer_vector.eval([ans2_onehot])
  qry_ans_similarity = 1-cosine(qry_embedding, ans_embedding) 
  qry_ans2_similarity = 1-cosine(qry_embedding, ans2_embedding)
  return qry_ans_similarity, qry_ans2_similarity

# Reading from file
def getVarFromFile(filename):
  f = open(filename)
  global data
  data = imp.load_source('data', '', f)
  f.close()
  return data