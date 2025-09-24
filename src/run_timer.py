import time

class Timer:
    def __init__(self):
        self.start_time: dict = {}

    def start(self, name):
        self.start_time[name] = time.time()
        print(f"Begin: {name:<40}Current time: {time.ctime():<30}")

    def stop(self, name):
        print(f"  End: {name:<40}Current time: {time.ctime():<30}Time elapsed: {time.time() - self.start_time[name]:.5f}s")
        del self.start_time[name]

TIMER = Timer()