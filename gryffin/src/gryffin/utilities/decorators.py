#!/usr/bin/env

__author__ = "Florian Hase"

import sys
import threading
import inspect
import traceback
from functools import wraps
from multiprocessing import Process, Queue


def safe_execute(error):
    def decorator_wrapper(function):
        def wrapper(*args, **kwargs):
            try:
                function(*args, **kwargs)
            except:
                error_type, error_message, traceback = sys.exc_info()
                error(error_message)

        return wrapper

    return decorator_wrapper


def thread(function):
    def wrapper(*args, **kwargs):
        background_thread = threading.Thread(target=function, args=args, kwargs=kwargs)
        background_thread.start()

    return wrapper


# ============================================
# Decorator that runs a function as a process
# ============================================
# This is useful to make sure proper garbage collection in TensorFlow 1.X, otherwise we keep allocating memory that
# does not get cleared when running Gryffin in a loop and we end up running out of memory.
# The decorator has been taken from here: https://gist.github.com/stuaxo/889db016e51264581b50
class Sentinel:
    pass


def processify(func):
    """Decorator to run a function as a process.
    Be sure that every argument and the return value
    is *pickable*.
    The created process is joined, so the code does not
    run in parallel.
    """

    def process_generator_func(q, *args, **kwargs):
        result = None
        error = None
        it = iter(func())
        while error is None and result != Sentinel:
            try:
                result = next(it)
                error = None
            except StopIteration:
                result = Sentinel
                error = None
            except Exception:
                ex_type, ex_value, tb = sys.exc_info()
                error = ex_type, ex_value, "".join(traceback.format_tb(tb))
                result = None
            q.put((result, error))

    def process_func(q, *args, **kwargs):
        try:
            result = func(*args, **kwargs)
        except Exception:
            ex_type, ex_value, tb = sys.exc_info()
            error = ex_type, ex_value, "".join(traceback.format_tb(tb))
            result = None
        else:
            error = None

        q.put((result, error))

    def wrap_func(*args, **kwargs):
        # register original function with different name
        # in sys.modules so it is pickable
        process_func.__name__ = func.__name__ + "processify_func"
        setattr(sys.modules[__name__], process_func.__name__, process_func)

        q = Queue()
        p = Process(target=process_func, args=[q] + list(args), kwargs=kwargs)
        p.start()
        result, error = q.get()
        p.join()

        if error:
            ex_type, ex_value, tb_str = error
            message = "%s (in subprocess)\n%s" % (str(ex_value), tb_str)
            raise ex_type(message)

        return result

    def wrap_generator_func(*args, **kwargs):
        # register original function with different name
        # in sys.modules so it is pickable
        process_generator_func.__name__ = func.__name__ + "processify_generator_func"
        setattr(
            sys.modules[__name__],
            process_generator_func.__name__,
            process_generator_func,
        )

        q = Queue()
        p = Process(target=process_generator_func, args=[q] + list(args), kwargs=kwargs)
        p.start()

        result = None
        error = None
        while error is None:
            result, error = q.get()
            if result == Sentinel:
                break
            yield result
        p.join()

        if error:
            ex_type, ex_value, tb_str = error
            message = "%s (in subprocess)\n%s" % (str(ex_value), tb_str)
            raise ex_type(message)

    @wraps(func)
    def wrapper(*args, **kwargs):
        if inspect.isgeneratorfunction(func):
            return wrap_generator_func(*args, **kwargs)
        else:
            return wrap_func(*args, **kwargs)

    return wrapper
