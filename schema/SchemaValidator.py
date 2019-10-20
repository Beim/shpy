import json
import os
from mongodb.models.SchemaMG import SchemaMG
from mongodb.MongoUtil import MongoUtil
from jsonschema import validate

mongo_util = MongoUtil()

curr_dir = os.path.split(os.path.abspath(__file__))[0]


with open('%s/schemas/ModelSchema.json' % curr_dir, 'r', encoding='utf-8') as f:
    model_schema = json.load(f)


class SchemaValidator:

    @staticmethod
    def get_model_schema():
        return model_schema

    @staticmethod
    def get_entity_schema(label):
        # local
        with open('%s/schemas/modelSchemas/%sSchema.json' % (curr_dir, label), encoding='utf-8') as f:
            entity_schema = json.load(f)
        return entity_schema
        # remote
        # res = SchemaMG.objects(primary_label=label)
        # if len(res) == 0:
        #     return None
        # else:
        #     return SchemaModel(res[0].primary_label, res[0].extends, res[0].entity_schema).__dict__

    @staticmethod
    def set_entity_schema(schema):
        s = SchemaMG()
        s.primary_label = schema['primary_label']
        s.extends = schema['extends']
        s.entity_schema = schema['entity_schema']
        s.save()

    @staticmethod
    def validate_new_model(model):
        validate(model, model_schema)
        # property_key = model['entity_schema']['properties']['property']['key']
        # property_required = model['entity_schema']['properties']['property']['required']
        # for key in property_key:
        #     if key not in property_required:
        #         raise Exception('%s not in required' % key)

    def validate_entity(self, entity):
        label = entity['primary_label']
        entity_schema = self.get_entity_schema(label)
        validate(entity, entity_schema['entity_schema'])
        parent_label = entity_schema['extends']
        while parent_label != '':
            try:
                parent_entity_schema = self.get_entity_schema(parent_label)
            except AttributeError as e:
                raise(Exception('label:%s not defined' % parent_label))
            validate(entity, parent_entity_schema['entity_schema'])
            parent_label = parent_entity_schema['extends']
        # validate relationship
        for relation_type in ['related', 'related_to', 'related_from']:
            if relation_type in entity:
                for relationship in entity[relation_type]:
                    if relationship not in entity_schema['entity_schema']['properties'][relation_type]['properties']:
                        raise Exception('%s not defined' % relationship)
                    related_entities = entity[relation_type][relationship]
                    for related_entity in related_entities:
                        self.validate_entity(related_entity)
                        # if not entity_exist(related_entity):
                        #     raise Exception('related_entity not exists')



if __name__ == '__main__':
    pass
    # print(Schema.get_entity_schema('Actor')
    schema = SchemaValidator()
    e = {
        "primary_label": "云服务器资源",
        "new": 1,
        "property": {
            "名": [
                "标准型SA1云服务器"
            ],
            "CPU": "1核",
            "CPU型号": "AMD EPYC 7551",
            "内存": "1GB",
            "GPU": "0",
            "主频": "2.0 GHz",
            "内网带宽": "1.5Gbps",
            "官网链接": "https://buy.cloud.tencent.com/cvm"
        }
    }
    schema.validate_entity(e)
    # print(schema.get_entity_schema('Movie'))
    # sg = schema.trans_entity_to_graph(e)
    # print(list(sg.subgraph.nodes))
    # print(list(sg.subgraph.relationships))
    # print(sg.subgraph.keys())
    # print(schema.trans_entity_to_graph(e))
    # print(schema.validate_entity(e))
    # print(schema.get_entity_schema('Actor'))

