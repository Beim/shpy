# shpy

**环境要求**

- python3
- rabbitmq 3.8
- neo4j 3.5


**运行**

```bash
pip install -r requirements.txt
# 或
pip install pika py2neo numpy python-Levenshtein requests jieba neo4j-driver

chmod +x bin/graph.sh
./bin/graph.sh
```



**配置**

配置文件

- `config.json` 指定环境
- `config-<env>.json`  配置

修改`config.json`

```json
{
    "env": "prod"
}
```

修改`config-prod.json`

- server.host  # 对应 [sh4j](https://github.com/Beim/sh4j) 的ip
- server.port  # 对应 [sh4j](https://github.com/Beim/sh4j) 的server.port
- rabbitmq.host
- rabbitmq.port
- rabbitmq.username
- rabbitmq.password
- neo4j.username # 对应[dockerContainerMgr](https://github.com/Beim/dockerContainerMgr) 配置的environment.NEO4J_AUTH
- neo4j.password # 对应[dockerContainerMgr](https://github.com/Beim/dockerContainerMgr) 配置的environment.NEO4J_AUTH

```json
{
    "server": {
        "host": "localhost",
        "port": 18080,
        "protocol": "http"
    },
    "rabbitmq": {
        "queue_name": "newResourceQueue",
        "host": "localhost",
        "username": "root",
        "password": "123456",
        "port": "5672",
        "durable": true,
        "auto_ack": false,
        "prefetch_count": 1
    },
    "neo4j": {
        "username": "neo4j",
        "password": "123123"
    }
}
```

