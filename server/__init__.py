# -*- coding: utf-8 -*-
"""
    @Author zbh
    @email pypy17z@126.com
    @Version 1.0.0
      创建日期:2022/2/27 17:15
      修改记录
      修改后版本:     修改人：   修改日期:    修改内容:
"""
import importlib

import settings
from flask import Flask, jsonify
from utils.log import logging

from .exceptions import NLPException

logger = logging.getLogger(__name__)


def create_app(testing=False):
    app = Flask(__name__)
    app.config['JSON_AS_ASCII'] = False
    app.testing = testing

    @app.errorhandler(Exception)
    def framework_error(e):
        """
        全局异常捕获
        """
        if not isinstance(e, NLPException):
            e = NLPException(str(e))

        resp = {
            "resp_code": e.code,
            "resp_msg": e.message,
            "data": e.data
        }

        logging.exception(e)

        return jsonify(resp)

    regist_blueprint(app, testing=testing)

    return app


def regist_blueprint(flask_app, testing=False):
    install_app = settings.INSTALL_APP
    if testing and not settings.DEBUG:
        install_app = settings.routers.APP_BLUEPRINT.keys()

    for app in install_app:
        module_name, blueprint_name, url = settings.routers.APP_BLUEPRINT[app]
        module = importlib.import_module(module_name)
        blueprint = getattr(module, blueprint_name)
        flask_app.register_blueprint(blueprint, url_prefix=url)
