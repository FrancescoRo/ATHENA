import time
from functools import wraps

#from "Fluent Python", Luciano Ramalho
DEFAULT_FMT = '[{elapsed:0.8f}s] {name}({args}, {kwargs}) -> {result}'
MY_FMT = '[{elapsed:0.8f}s] {name}'


def clock(fmt=MY_FMT, activate=False):
    def decorate(func):
        if activate is True:

            @wraps(func)
            def clocked(*_args, **_kwargs):
                t0 = time.time()
                _result = func(*_args, **_kwargs)
                elapsed = time.time() - t0
                name = func.__name__
                args = ', '.join(repr(arg) for arg in _args)
                kwargs = ', '.join(repr(kwarg) for kwarg in _kwargs)
                result = repr(_result)
                print(fmt.format(**locals()))
                return _result

            return clocked
        else:
            return func

    return decorate
