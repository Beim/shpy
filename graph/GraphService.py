import json

from Utils import request_utils

class GraphService:

    OBJECT_PROPERTY = "objectProperty"
    DATATYPE_PROPERTY = "datatypeProperty"
    DOMAIN = "domain"
    RANGE = "range"

    @staticmethod
    def get_class_info(class_relative_uri_list):
        """
        获取class 的信息
        example:
        {
            "msg": "ok",
            "code": 1,
            "succ": true,
            "data": {
                "Movie": {
                    "objectProperty": {},
                    "datatypeProperty": {
                        "hasDirector": {
                            "domain": [
                                "Movie"
                            ],
                            "range": [
                                "string"
                            ]
                        },
                        "hasActor": {
                            "domain": [
                                "Movie"
                            ],
                            "range": [
                                "string"
                            ]
                        }
                    }
                }
            },
            "oper": "getClassInfo"
        }
        :param class_relative_uri_list:
        :return:
        """
        data = {
            'classRelativeUriList': class_relative_uri_list
        }
        res = json.loads(request_utils.post('schema/classInfo', data).text)
        return res['data']

