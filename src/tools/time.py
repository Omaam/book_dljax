'''
'''
import time


def stopwatch(func):
    def wrapper(*args, **kwargs):
        time_start = time.time()
        results = func(*args, **kwargs)
        time_stop = time.time()

        time_elapsed = time_stop - time_start
        minutes = time_elapsed // 60
        seconds = time_elapsed % 60
        print('Wall time: {}min {}s'.format(
            int(minutes), int(seconds)))
        return results
    return wrapper
