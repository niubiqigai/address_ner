"""
HOW-TO

from server import BaseExceptionContainer,declare_exc

class my_exceptions(BaseExceptionContainer):
    exc_code_prefix = '11'
    ExceptionA = declare_exc(code='01',default_message='Error MSG',base=True)
    ExceptionB = declare_exc('02','Error MSG')

raise my_exceptions.ExceptionA
raise my_exceptions.ExceptionA(data=None)

所有用BaseExceptionContainer实现的异常均为NLPException的子类
同一个BaseExceptionContainer下均以base=True的异常作为父类
"""

from dataclasses import dataclass
from typing import Dict, Tuple


class NLPException(Exception):
    default_message = 'unknown error'
    code = '999999'

    def __init__(self, message: str = '', data: dict = None) -> None:
        self.message = message or self.default_message
        self.data = data

    def __str__(self) -> str:
        return f'{self.code}-{self.message}-{self.data}'


@dataclass
class declared_exc:
    code: str
    default_message: str = '未知错误'
    base: bool = False


declared_exceptions: Dict[str, NLPException] = {}


class BaseExceptionContainerMeta(type):

    def __new__(cls, name: str, bases: tuple, attrs: dict):
        new_cls = super().__new__(cls, name, bases, attrs)
        if not bases:
            return new_cls

        assert isinstance(
            getattr(new_cls, 'exc_code_prefix', None),
            str
        ), (f'{new_cls.__module__}-{new_cls.__name__} expected "exc_code_prefix" is a string.')

        def new_exc(
                exc_name: str,
                declared_exc: declared_exc,
                bases: Tuple = (NLPException,),
        ) -> NLPException:
            exc_attrs = declared_exc.__dict__.copy()
            exc_attrs['code'] = f'{new_cls.exc_code_prefix}{declared_exc.code}'
            exc_attrs['_container'] = new_cls

            return type(exc_name, bases, exc_attrs)

        cls_declared_exceptions: Dict[str, Tuple[str, declared_exc]] = {}
        base_exc_code = None
        for k, v in attrs.items():
            if isinstance(v, declared_exc):
                code = f'{new_cls.exc_code_prefix}{v.code}'
                if code in declared_exceptions or code in cls_declared_exceptions:
                    e = declared_exceptions.get(code)
                    if not e:
                        _, e = cls_declared_exceptions[code]

                    raise RuntimeError(
                        'Multiple exceptions were defined with code'
                        f"{v.code}:{new_cls.__name__}.{k},"
                        f'{e._container.__name__}.{e.__name__}'
                    )

                if v.base:
                    if base_exc_code:
                        raise RuntimeError(
                            'Expected only one declared_exc(base=True) in'
                            f'{new_cls.__name__}'
                        )
                    base_exc_code = code

                cls_declared_exceptions[code] = (k, v)

        if not base_exc_code:
            base_exc_name, base_exc_declared = (
                'BaseException', declared_exc('0000', base=True))
        else:
            base_exc_name, base_exc_declared = (
                cls_declared_exceptions[base_exc_code])

        base_exc = new_exc(base_exc_name, base_exc_declared)

        declared_exceptions[base_exc.code] = base_exc
        setattr(new_cls, base_exc_name, base_exc)

        for code, (exc_name, exc_declared) in cls_declared_exceptions.items():
            if exc_name != base_exc_name:
                exc = new_exc(exc_name, exc_declared, bases=(base_exc,))
                setattr(new_cls, exc_name, exc)
                declared_exceptions[code] = exc

        return new_cls


class BaseExceptionContainer(metaclass=BaseExceptionContainerMeta):
    exc_code_prefix = None
