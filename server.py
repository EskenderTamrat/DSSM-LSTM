import grpc
from concurrent import futures
import time

from service_spec.LSTM_pb2 import Query, Response
from service_spec.LSTM_pb2_grpc import ReplyServicer, add_ReplyServicer_to_server

from LSTM import lstm

class ReplyServicer(ReplyServicer):
	def semantic_modeling(self, request, context):
		response = Response()
		response.qry_ans_similarity, response.qry_poor_ans_similarity = lstm(request.train_file, request.validation_file, request.query_wf, request.answer_wf)
		return response

def create_server(port=8001):
	server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
	add_ReplyServicer_to_server(ReplyServicer(), server)
	print('Starting server. Listening on port 8001.')
	server.add_insecure_port('[::]:' + str(port))
	return server

if __name__ == '__main__':
	server = create_server()
	server.start()
	# since server.start() will not block, a sleep-loop is added to keep alive
	try:
	    while True:
	        time.sleep(86400)
	except KeyboardInterrupt:
	    server.stop(0)