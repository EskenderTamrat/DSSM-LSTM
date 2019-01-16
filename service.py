import grpc

import argparse
import os
import sys
import imp

from service_spec.DSSMService_pb2 import DSSMRequest
from service_spec.DSSMService_pb2_grpc import DSSMStub

def searchWord(string, file):
  with open(file) as myfile:
  	if string.strip():
	    for word in string.split(" "):
	    	if word in myfile.read():
	    		myfile.seek(0)
	    	else:
		      string = string.replace(word, "")
		      myfile.seek(0)
  myfile.close()
  return string

# Reading from file
def getVarFromFile(filename):
  f = open(filename)
  global data
  data = imp.load_source('data', '', f)
  f.close()
  return data

if __name__ == "__main__":
	# open a gRPC channel
	channel = grpc.insecure_channel('localhost:8001')
	stub = DSSMStub(channel)

	parser = argparse.ArgumentParser()
	parser.add_argument("--qry", type=str, help='query entry (you could only use terms from the model this network trained)')
	parser.add_argument("--ans1", type=str, help='first answer to compare relevance with the query (you could only use terms from the model this network trained)')
	parser.add_argument("--ans2", type=str, help='second answer to compare relevance with the query (you could only use terms from the model this network trained)')
	
	args = parser.parse_args()
	if len(sys.argv) == 1:
		parser.print_help()
		sys.exit()


  # Check whether user's query and answers list suits the model 
	data = getVarFromFile("variables.txt")
	qry = searchWord(args.qry, data.query_wf)
	ans_1 = searchWord(args.ans1, data.answer_wf)
	ans_2 = searchWord(args.ans2, data.answer_wf)

	if not qry.split():
		print("Your query entry doesn't include terms that exist in the model.")
	elif not ans_1.split():
		print("Your entry for first answer doesn't include terms that exist in the model")
	elif not ans_2.split():
		print("Your entry for second answer doesn't include terms that exist in the model")
	else:
		response = stub.semantic_modeling(DSSMRequest(qry = qry, ans1 = ans_1, ans2 = ans_2))
		print("Query to Answer similarity: ", response.qry_ans_similarity)
		print("Query to Answer 2 similarity: ", response.qry_ans2_similarity)
		print("\"", max(args.ans1, args.ans2), "\" is a better answer for \"", args.qry, "\"")