import grpc
from concurrent import futures
import time

from service_spec.DSSMService_pb2 import DSSMRequest, DSSMResponse
from service_spec.DSSMService_pb2_grpc import DSSMServicer, add_DSSMServicer_to_server

from DSSM import lstm

class DSSMServicer(DSSMServicer):
	def semantic_modeling(self, request, context):
		response = DSSMResponse()
		response.qry_ans_similarity, response.qry_ans2_similarity = lstm(request.qry, request.ans1, request.ans2)
		return response

def create_server(port=8001):
	server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
	add_DSSMServicer_to_server(DSSMServicer(), server)
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