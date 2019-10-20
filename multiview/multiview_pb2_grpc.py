# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
import grpc

import multiview_pb2 as multiview__pb2


class MultiViewServiceStub(object):
  # missing associated documentation comment in .proto file
  pass

  def __init__(self, channel):
    """Constructor.

    Args:
      channel: A grpc.Channel.
    """
    self.GenGraph = channel.unary_unary(
        '/com.ices.sh.multiview.rpc.MultiViewService/GenGraph',
        request_serializer=multiview__pb2.GenGraphConfig.SerializeToString,
        response_deserializer=multiview__pb2.Result.FromString,
        )
    self.GenPatternExist = channel.unary_unary(
        '/com.ices.sh.multiview.rpc.MultiViewService/GenPatternExist',
        request_serializer=multiview__pb2.GenPattern.SerializeToString,
        response_deserializer=multiview__pb2.Result.FromString,
        )
    self.GenRelationshipFrequency = channel.unary_unary(
        '/com.ices.sh.multiview.rpc.MultiViewService/GenRelationshipFrequency',
        request_serializer=multiview__pb2.GenRelationship.SerializeToString,
        response_deserializer=multiview__pb2.Double.FromString,
        )


class MultiViewServiceServicer(object):
  # missing associated documentation comment in .proto file
  pass

  def GenGraph(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def GenPatternExist(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def GenRelationshipFrequency(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')


def add_MultiViewServiceServicer_to_server(servicer, server):
  rpc_method_handlers = {
      'GenGraph': grpc.unary_unary_rpc_method_handler(
          servicer.GenGraph,
          request_deserializer=multiview__pb2.GenGraphConfig.FromString,
          response_serializer=multiview__pb2.Result.SerializeToString,
      ),
      'GenPatternExist': grpc.unary_unary_rpc_method_handler(
          servicer.GenPatternExist,
          request_deserializer=multiview__pb2.GenPattern.FromString,
          response_serializer=multiview__pb2.Result.SerializeToString,
      ),
      'GenRelationshipFrequency': grpc.unary_unary_rpc_method_handler(
          servicer.GenRelationshipFrequency,
          request_deserializer=multiview__pb2.GenRelationship.FromString,
          response_serializer=multiview__pb2.Double.SerializeToString,
      ),
  }
  generic_handler = grpc.method_handlers_generic_handler(
      'com.ices.sh.multiview.rpc.MultiViewService', rpc_method_handlers)
  server.add_generic_rpc_handlers((generic_handler,))
