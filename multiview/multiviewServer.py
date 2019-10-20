import multiview.multiview_pb2_grpc as mv_grpc
import multiview.multiview_pb2 as mv
import grpc
from concurrent import futures
import time

_ONE_DAY_IN_SECONDS = 60 * 60 * 24


class MultiviewServicer(mv_grpc.MultiViewServiceServicer):

    def GenGraph(self, request, context):
        gen_graph_config = request
        from multiview.ViewUtil import gen_graph, expand_gen_graph
        if gen_graph_config.lastPattConfig == "":
            gen_graph(gen_graph_config.currPattConfig)
        else:
            expand_gen_graph(gen_graph_config.lastPattConfig, gen_graph_config.currPattConfig)
        return mv.Result(ok=True)

    def GenPatternExist(self, request, context):
        gen_pattern = request
        from multiview.ViewUtil import gen_pattern_exist
        return mv.Result(ok=gen_pattern_exist(gen_pattern))

    def GenRelationshipFrequency(self, request, context):
        gen_relationship = request
        from multiview.ViewUtil import gen_relationship_frequency
        return mv.Double(value=gen_relationship_frequency(gen_relationship))


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    mv_grpc.add_MultiViewServiceServicer_to_server(
        MultiviewServicer(), server
    )
    server.add_insecure_port('[::]:50051')
    server.start()
    print('start serve...')
    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)


if __name__ == '__main__':
    serve()
