from flask import Blueprint, request
from schema.SchemaValidator import SchemaValidator
import json

schema_bp = Blueprint('schema', __name__)

@schema_bp.route('/')
def index():
    return 'this is schema_bp'

@schema_bp.route('/ModelSchema')
def model_schema_json():
    res = SchemaValidator.get_model_schema()
    return json.dumps(res, ensure_ascii=False, indent=2)


@schema_bp.route('/schema/<string:label>', methods=('GET',))
def get_schema(label):
    res = SchemaValidator.get_entity_schema(label)
    print(label)
    return json.dumps(res, ensure_ascii=False, indent=2)


@schema_bp.route('/schema', methods=('POST',))
def post_schema():
    print('get request')
    try:
        entity_schema = request.json
        SchemaValidator.validate_new_model(entity_schema)
        SchemaValidator.set_entity_schema(entity_schema)
        return "success"
    except Exception as e:
        print(e)
        return "failed"


@schema_bp.route('/entity', methods=('POST',))
def post_entity():

    pass