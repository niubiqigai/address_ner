"""
！！！避免陷入循环依赖！！！
settings.logger不能依赖于settings.__init__
"""
from datetime import datetime
from logging.handlers import TimedRotatingFileHandler

import environs
import platform

env = environs.Env()
env.read_env()

DEBUG = env.bool('DEBUG', False)
DEFAULT_LOG_PATH = env.str('DEFAULT_LOG_PATH', 'log/access.log')
if platform.system() != "Windows":
    LOGGER_HANDLERS = env.list(
        'LOGGER_HANDLERS',
        ['default']
    )
else:
    LOGGER_HANDLERS = env.list(
        'LOGGER_HANDLERS',
        ['default', 'console']
    )

LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "simple": {
            'format': '%(asctime)s - %(pathname)s:%(lineno)d - %(levelname)s - %(message)s',
            'datefmt': '%Y-%m-%d %H:%M:%S'
        },
        'standard': {
            'format': '%(asctime)s [%(threadName)s:%(thread)d] [%(name)s:%(lineno)d] [%(levelname)s]- %(message)s'
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "simple",
            "stream": "ext://sys.stdout"
        },
        "default": {
            "class": "settings.logger.RotatingFileHandlerWithFilename",
            "level": "DEBUG" if DEBUG else "INFO",
            "formatter": "simple",
            "filename": DEFAULT_LOG_PATH,
            "when": 'd',
            "backupCount": 20,
            "encoding": "utf8"
        },
        "pdfminer": {
            "class": "settings.logger.RotatingFileHandlerWithFilename",
            "level": "ERROR",
            "formatter": "simple",
            "filename": DEFAULT_LOG_PATH,
            "when": 'd',
            "backupCount": 20,
            "encoding": "utf8"
        }
    },
    "root": {
        'handlers': LOGGER_HANDLERS,
        'level': "DEBUG" if DEBUG else "INFO",
        'propagate': False
    },
    "pdfminer": {
        'handlers': ["pdfminer"],
        'level': "ERROR",
        'propagate': False
    }
}


class RotatingFileHandlerWithFilename(TimedRotatingFileHandler):

    def __init__(self, filename, when='h', interval=1, backupCount=0, encoding=None, delay=False, utc=False,
                 atTime=None):
        super().__init__(filename, when=when, interval=interval, backupCount=backupCount, encoding=encoding,
                         delay=delay, utc=utc, atTime=atTime)

        self.namer = lambda x: f'{x.rsplit(".", 1)[0]}.{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
