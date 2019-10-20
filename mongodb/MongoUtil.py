from mongoengine import connect
import os
import json

curr_dir = os.path.split(os.path.abspath(__file__))[0]


class MongoUtil:

    def __init__(self):
        with open('%s/config.json' % curr_dir, 'r', encoding='utf-8') as f:
            cfg = json.load(f)
            host = cfg['db']['host']
            port = cfg['db']['port']
            database = cfg['db']['database']
            username = cfg['db']['username']
            password = cfg['db']['password']
            connect(database, host=host, port=port, username=username, password=password)

    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, '_instance'):
            cls._instance = super(MongoUtil, cls).__new__(cls, *args, **kwargs)
        return cls._instance


if __name__ == '__main__':
    mongo_util = MongoUtil()

