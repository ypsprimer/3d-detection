# coding=utf8
import time
from functools import wraps


def timerecord(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print("function '%s' takes %d seconds " % (func.__name__, end - start))
        return result

    return wrapper


if __name__ == '__main__':
    @timerecord
    def funct():
        time.sleep(5)

    funct()