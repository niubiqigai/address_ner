# 项目说明
**该项目使用BiLSTM+CRF进行命名实体识别任务的开发，具体算法的介绍网上很多，请自行查阅资料。**

该项目识别的实体包括：
```
1、省份
2、城市
3、区县
4、具体地址
5、收件人
6、收件人联系电话
```

项目两点：
```
1、对省、市、区全称和简称做了较完整的收集，构建的训练集数据基本覆盖全国省市区。
2、对全国采用姓氏做了较完整的收集，构建的训练集数据基本覆盖全国姓氏。
3、对城市和区县的邮政编码进行整理，返回报文中包含邮政编码。
```

# 1.依赖安装

## 命令

```shell
pip install -r requirements.txt -i https://pypi.douban.com/simple/
```

本项目有两个版本的requirements文件  

- requirements.txt：主依赖
- requirements_gpu.txt：继承主依赖，增加了对GPU的支持
  - tensorflow-gpu

请根据实际环境需要安装对应依赖

# 2.环境变量配置
本项目通过在settings文件夹下的.env文件（此文件不纳入SVN版本控制）配置环境变量
例子：
```
# settings/.env
DEBUG=on #打开调试
```
具体可配置项查看settings/__init__文件

# 3.部署
## 部署命令
```
# 1进程，8线程
开启服务：sh python-ai-nlp-server.sh start  
停止服务：sh python-ai-nlp-server.sh stop
重启服务：sh python-ai-nlp-server.sh restart
```

## gunicorn启动服务
```
# 1进程，8线程
gunicorn -c gunicorn_config.py nlp_server:app
```

# 4.模型训练与更新
## 训练
`python -m src.address_ner.product.v1_0.main`
## 更新
如果需要更新版本号，可以进行如下操作：
```
1、src/address_ner/product目录下复制一个新的版本出来，比如v1_1
2、通过修改src/address_ner/product/v1_1/const.py文件下MODEL_VERSION的值为'v1_1'
3、修改该版本相关代码，运行新模型训练即可
4、如果新模型更适合你的使用需要，此时修改路由的导入模型预测路径，即可
```