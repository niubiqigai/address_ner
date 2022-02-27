# -*- coding: utf-8 -*-
"""
    @Author zbh
    @email pypy17z@126.com
    @Version 1.0.0
      创建日期:2022/2/27 22:00
      修改记录
      修改后版本:     修改人：   修改日期:    修改内容:
"""

import settings

# debug
debug = settings.DEBUG
# 端口绑定
bind = f"{settings.FLASK_HOST}:{settings.FLASK_PORT}"
# gunicorn日志等级
loglevel = 'info'
# gunicorn masterpid
pidfile = "log/gunicorn.pid"
accesslog = "log/gunicorn_access.log"
errorlog = "log/gunicorn_error.log"

# 开启后台运行；通过supervisor启动需关闭后台模式
# capture_output = True
daemon = True
# 运行挂起的连接数
backlog = 2048
threads = 8
# 启动的进程数
workers = 1
# worker类型，可以为sync, eventlet, gevent, tornado, gthread, 默认sync
worker_class = 'gthread'

# 修改超时时间
timeout = 120

# Redirect stdout/stderr to specified file in errorlog.
capture_output = True
