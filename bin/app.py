from graph.ResourceQueueReceiver import ResourceQueueReceiver
from config_loader import config_loader

rabbitmq_config = config_loader.get_config()['rabbitmq']
receiver = ResourceQueueReceiver(rabbitmq_config['host'],
                                 rabbitmq_config['username'],
                                 rabbitmq_config['password'],
                                 rabbitmq_config['port'],
                                 rabbitmq_config['queue_name'],
                                 rabbitmq_config['durable'],
                                 rabbitmq_config['auto_ack'],
                                 rabbitmq_config['prefetch_count'])