# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: multiview.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='multiview.proto',
  package='com.ices.sh.multiview.rpc',
  syntax='proto3',
  serialized_options=_b('\n\031com.ices.sh.multiview.rpcB\014MultiviewRpc'),
  serialized_pb=_b('\n\x0fmultiview.proto\x12\x19\x63om.ices.sh.multiview.rpc\"@\n\x0eGenGraphConfig\x12\x16\n\x0elastPattConfig\x18\x01 \x01(\t\x12\x16\n\x0e\x63urrPattConfig\x18\x02 \x01(\t\"\x14\n\x06Result\x12\n\n\x02ok\x18\x01 \x01(\x08\"\xc9\x01\n\nGenPattern\x12<\n\x08startRel\x18\x01 \x01(\x0b\x32*.com.ices.sh.multiview.rpc.GenRelationship\x12\x14\n\x0cstartInverse\x18\x02 \x01(\x08\x12:\n\x06\x65ndRel\x18\x03 \x01(\x0b\x32*.com.ices.sh.multiview.rpc.GenRelationship\x12\x12\n\nendInverse\x18\x04 \x01(\x08\x12\x17\n\x0fisFirstInterval\x18\x05 \x01(\x08\"D\n\x0fGenRelationship\x12\x12\n\nstartLabel\x18\x01 \x01(\t\x12\x0b\n\x03rel\x18\x02 \x01(\t\x12\x10\n\x08\x65ndLabel\x18\x03 \x01(\t\"\x17\n\x06\x44ouble\x12\r\n\x05value\x18\x01 \x01(\x01\x32\xb4\x02\n\x10MultiViewService\x12X\n\x08GenGraph\x12).com.ices.sh.multiview.rpc.GenGraphConfig\x1a!.com.ices.sh.multiview.rpc.Result\x12[\n\x0fGenPatternExist\x12%.com.ices.sh.multiview.rpc.GenPattern\x1a!.com.ices.sh.multiview.rpc.Result\x12i\n\x18GenRelationshipFrequency\x12*.com.ices.sh.multiview.rpc.GenRelationship\x1a!.com.ices.sh.multiview.rpc.DoubleB)\n\x19\x63om.ices.sh.multiview.rpcB\x0cMultiviewRpcb\x06proto3')
)




_GENGRAPHCONFIG = _descriptor.Descriptor(
  name='GenGraphConfig',
  full_name='com.ices.sh.multiview.rpc.GenGraphConfig',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='lastPattConfig', full_name='com.ices.sh.multiview.rpc.GenGraphConfig.lastPattConfig', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='currPattConfig', full_name='com.ices.sh.multiview.rpc.GenGraphConfig.currPattConfig', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=46,
  serialized_end=110,
)


_RESULT = _descriptor.Descriptor(
  name='Result',
  full_name='com.ices.sh.multiview.rpc.Result',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='ok', full_name='com.ices.sh.multiview.rpc.Result.ok', index=0,
      number=1, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=112,
  serialized_end=132,
)


_GENPATTERN = _descriptor.Descriptor(
  name='GenPattern',
  full_name='com.ices.sh.multiview.rpc.GenPattern',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='startRel', full_name='com.ices.sh.multiview.rpc.GenPattern.startRel', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='startInverse', full_name='com.ices.sh.multiview.rpc.GenPattern.startInverse', index=1,
      number=2, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='endRel', full_name='com.ices.sh.multiview.rpc.GenPattern.endRel', index=2,
      number=3, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='endInverse', full_name='com.ices.sh.multiview.rpc.GenPattern.endInverse', index=3,
      number=4, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='isFirstInterval', full_name='com.ices.sh.multiview.rpc.GenPattern.isFirstInterval', index=4,
      number=5, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=135,
  serialized_end=336,
)


_GENRELATIONSHIP = _descriptor.Descriptor(
  name='GenRelationship',
  full_name='com.ices.sh.multiview.rpc.GenRelationship',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='startLabel', full_name='com.ices.sh.multiview.rpc.GenRelationship.startLabel', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='rel', full_name='com.ices.sh.multiview.rpc.GenRelationship.rel', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='endLabel', full_name='com.ices.sh.multiview.rpc.GenRelationship.endLabel', index=2,
      number=3, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=338,
  serialized_end=406,
)


_DOUBLE = _descriptor.Descriptor(
  name='Double',
  full_name='com.ices.sh.multiview.rpc.Double',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='value', full_name='com.ices.sh.multiview.rpc.Double.value', index=0,
      number=1, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=408,
  serialized_end=431,
)

_GENPATTERN.fields_by_name['startRel'].message_type = _GENRELATIONSHIP
_GENPATTERN.fields_by_name['endRel'].message_type = _GENRELATIONSHIP
DESCRIPTOR.message_types_by_name['GenGraphConfig'] = _GENGRAPHCONFIG
DESCRIPTOR.message_types_by_name['Result'] = _RESULT
DESCRIPTOR.message_types_by_name['GenPattern'] = _GENPATTERN
DESCRIPTOR.message_types_by_name['GenRelationship'] = _GENRELATIONSHIP
DESCRIPTOR.message_types_by_name['Double'] = _DOUBLE
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

GenGraphConfig = _reflection.GeneratedProtocolMessageType('GenGraphConfig', (_message.Message,), dict(
  DESCRIPTOR = _GENGRAPHCONFIG,
  __module__ = 'multiview_pb2'
  # @@protoc_insertion_point(class_scope:com.ices.sh.multiview.rpc.GenGraphConfig)
  ))
_sym_db.RegisterMessage(GenGraphConfig)

Result = _reflection.GeneratedProtocolMessageType('Result', (_message.Message,), dict(
  DESCRIPTOR = _RESULT,
  __module__ = 'multiview_pb2'
  # @@protoc_insertion_point(class_scope:com.ices.sh.multiview.rpc.Result)
  ))
_sym_db.RegisterMessage(Result)

GenPattern = _reflection.GeneratedProtocolMessageType('GenPattern', (_message.Message,), dict(
  DESCRIPTOR = _GENPATTERN,
  __module__ = 'multiview_pb2'
  # @@protoc_insertion_point(class_scope:com.ices.sh.multiview.rpc.GenPattern)
  ))
_sym_db.RegisterMessage(GenPattern)

GenRelationship = _reflection.GeneratedProtocolMessageType('GenRelationship', (_message.Message,), dict(
  DESCRIPTOR = _GENRELATIONSHIP,
  __module__ = 'multiview_pb2'
  # @@protoc_insertion_point(class_scope:com.ices.sh.multiview.rpc.GenRelationship)
  ))
_sym_db.RegisterMessage(GenRelationship)

Double = _reflection.GeneratedProtocolMessageType('Double', (_message.Message,), dict(
  DESCRIPTOR = _DOUBLE,
  __module__ = 'multiview_pb2'
  # @@protoc_insertion_point(class_scope:com.ices.sh.multiview.rpc.Double)
  ))
_sym_db.RegisterMessage(Double)


DESCRIPTOR._options = None

_MULTIVIEWSERVICE = _descriptor.ServiceDescriptor(
  name='MultiViewService',
  full_name='com.ices.sh.multiview.rpc.MultiViewService',
  file=DESCRIPTOR,
  index=0,
  serialized_options=None,
  serialized_start=434,
  serialized_end=742,
  methods=[
  _descriptor.MethodDescriptor(
    name='GenGraph',
    full_name='com.ices.sh.multiview.rpc.MultiViewService.GenGraph',
    index=0,
    containing_service=None,
    input_type=_GENGRAPHCONFIG,
    output_type=_RESULT,
    serialized_options=None,
  ),
  _descriptor.MethodDescriptor(
    name='GenPatternExist',
    full_name='com.ices.sh.multiview.rpc.MultiViewService.GenPatternExist',
    index=1,
    containing_service=None,
    input_type=_GENPATTERN,
    output_type=_RESULT,
    serialized_options=None,
  ),
  _descriptor.MethodDescriptor(
    name='GenRelationshipFrequency',
    full_name='com.ices.sh.multiview.rpc.MultiViewService.GenRelationshipFrequency',
    index=2,
    containing_service=None,
    input_type=_GENRELATIONSHIP,
    output_type=_DOUBLE,
    serialized_options=None,
  ),
])
_sym_db.RegisterServiceDescriptor(_MULTIVIEWSERVICE)

DESCRIPTOR.services_by_name['MultiViewService'] = _MULTIVIEWSERVICE

# @@protoc_insertion_point(module_scope)
