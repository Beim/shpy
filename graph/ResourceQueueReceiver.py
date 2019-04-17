import pika
import json

from graph.PutinController import PutinController
from config_loader import config_loader


class ResourceQueueReceiver:
    """
    从rabbitmq 队列中接收资源，交给PutinController 处理
    """

    connection = None
    channel = None
    p_controller = None

    def __init__(self, host, username, password, port, queue_name, durable, auto_ack, prefetch_count):
        self.p_controller = PutinController()
        self.connection, self.channel = self.create_connection(host, username, password, port, queue_name,
                                                               durable, auto_ack, prefetch_count)
        self.channel.start_consuming()

    def create_connection(self, host, username, password, port, queue_name, durable, auto_ack, prefetch_count):
        """
        建立连接
        :param host:
        :param username:
        :param password:
        :param queue_name: 队列名
        :param durable: 持久化
        :param auto_ack: 自动确认
        :param prefetch_count: 每个worker 最多接受的消息数量
        :return: connection, channel
        """
        credential = pika.PlainCredentials(username, password)
        connection_params = pika.ConnectionParameters(host=host, credentials=credential, port=port)
        connection = pika.BlockingConnection(connection_params)
        channel = connection.channel()
        channel.queue_declare(queue=queue_name, durable=durable)
        channel.basic_qos(prefetch_count=prefetch_count)
        channel.basic_consume(queue=queue_name, on_message_callback=self.receive_callback, auto_ack=auto_ack)
        print('[pika] connection established, waiting for messages...')
        return connection, channel

    def receive_callback(self, ch, method, properties, body):
        """
        接受消息的回调，调用putin 将数据入库
        :param ch:
        :param method:
        :param properties:
        :param body:
        :return:
        """
        data = json.loads(body, encoding='utf-8')
        print(data)
        self.p_controller.putin(data)
        try:

            # 确认，返回ack
            ch.basic_ack(delivery_tag=method.delivery_tag)
            print('ack', data)
        except Exception as e:
            print(e)
            # 异常，返回nack
            ch.basic_nack(delivery_tag=method.delivery_tag)
            print('nack', data)


if __name__ == '__main__':
    rabbitmq_config = config_loader.get_config()['rabbitmq']
    receiver = ResourceQueueReceiver(rabbitmq_config['host'],
                                     rabbitmq_config['username'],
                                     rabbitmq_config['password'],
                                     rabbitmq_config['port'],
                                     rabbitmq_config['queue_name'],
                                     rabbitmq_config['durable'],
                                     rabbitmq_config['auto_ack'],
                                     rabbitmq_config['prefetch_count'])
