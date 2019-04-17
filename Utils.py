import requests
import json
import warnings
import functools

from config_loader import config_loader


class RequestUtils:

    def __init__(self):
        server_config = config_loader.get_config()['server']
        self.protocol = server_config['protocol']
        self.host = server_config['host']
        self.port = server_config['port']

    def post(self, path, data, cookies=None):
        url = '%s://%s:%s/%s' % (self.protocol, self.host, self.port, path)
        data_str = json.dumps(data, ensure_ascii=False).encode('utf-8')
        res = requests.post(url, data=data_str, cookies=cookies)
        return res


request_utils = RequestUtils()


def deprecated(func):
    """This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emitted
    when the function is used."""
    @functools.wraps(func)
    def new_func(*args, **kwargs):
        warnings.simplefilter('always', DeprecationWarning)  # turn off filter
        warnings.warn("Call to deprecated function {}.".format(func.__name__),
                      category=DeprecationWarning,
                      stacklevel=2)
        warnings.simplefilter('default', DeprecationWarning)  # reset filter
        return func(*args, **kwargs)
    return new_func

if __name__ == '__main__':
    pass
