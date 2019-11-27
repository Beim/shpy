# shpy

**环境要求**

- python3



**运行**

```bash
pip install -r requirements.txt
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
    "env": "prod"  // 指定prod 环境
}
```

修改`config-prod.json`

- server.host  # 对应 [sh4j](https://github.com/Beim/sh4j) 的ip
- server.port  # 对应 [sh4j](https://github.com/Beim/sh4j) 的server.port
- rabbitmq.host
- rabbitmq.port
- rabbitmq.username
- rabbitmq.password

```json
{
    "server": {
        "host": "api.brianxkliu.xyz",
        "port": 80,
        "protocol": "http"
    },
    "rabbitmq": {
        "queue_name": "newResourceQueue",
        "host": "brianxkliu.xyz",
        "username": "root",
        "password": "112223334",
        "port": "55672",
        "durable": true,
        "auto_ack": false,
        "prefetch_count": 1
    }
}
```

