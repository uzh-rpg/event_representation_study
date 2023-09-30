import time


class TimerError(Exception):
    """A custom exception used to report errors in use of Timer class"""


class Timer:
    def __init__(self, verbose=False):
        self._start_time = None
        self.verbose = verbose

    def start(self):
        """Start a new timer"""
        if self._start_time is not None:
            raise TimerError("Timer is running. Use .stop() to stop it")

        self._start_time = time.perf_counter()

    def stop(self, new_msg=None):
        """Stop the timer, and report the elapsed time"""
        if self._start_time is None:
            raise TimerError("Timer is not running. Use .start() to start it")

        elapsed_time = time.perf_counter() - self._start_time
        self._start_time = None
        if self.verbose:
            if new_msg is not None:
                print(f"Elapsed time: {elapsed_time:0.4f} seconds " + new_msg)
            else:
                print(f"Elapsed time: {elapsed_time:0.4f} seconds")
            return elapsed_time
        else:
            return elapsed_time


def time_proto(msg):
    def time_func(func):
        def time_wrapper(*args, **kwargs):
            t = Timer()
            t.start()
            return_val = func(*args, **kwargs)
            t.stop(msg)
            return return_val

        return time_wrapper

    return time_func
