# -*- coding: utf-8 -*-
"""
    @Author zbh
    @email pypy17z@126.com
    @Version 1.0.0
      创建日期:2022/2/27 17:10
      修改记录
      修改后版本:     修改人：   修改日期:    修改内容:
"""
import settings
from server import create_app

app = create_app()

if __name__ == '__main__':
    app.run(host=settings.FLASK_HOST,
            port=settings.FLASK_PORT, debug=settings.DEBUG)
