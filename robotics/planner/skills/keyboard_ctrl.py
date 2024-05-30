# key board controller
import numpy as np
from typing import Any, Dict, Optional

from robotics.planner.agent import MobileAgent
from ..skill import Skill, SkillConfig
import queue

import threading
 
lock = threading.Lock()


class _GetchWindows:
    def __init__(self):
        import msvcrt

    def __call__(self):
        import msvcrt
        return msvcrt.getch() # type: ignore


class _Getch:
    """Gets a single character from standard input.  Does not echo to the
screen."""
    def __init__(self):
        try:
            self.impl = _GetchWindows()
        except ImportError:
            self.impl = _GetchUnix()

    def __call__(self): return self.impl()


class _GetchUnix:
    def __init__(self):
        import tty, sys

    def __call__(self):
        import sys, tty, termios
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch


 
class myThread(threading.Thread):
    def __init__(self, _id, name):
        super().__init__()
        self.daemon = True    # daemon threads are killed as soon as the main program exits
        self._id = _id
        self.name = name
        self.stop = False
        self.queue = queue.Queue()
 
    def run(self):
        _getch = _Getch() 
        print("Start capturing keyboard!!!!!!!!!!!!!!!!!!!!!!!!")
        while True:
            if self.stop:
                break
            s = _getch()
            if ord(s) == 3:
                s = 'q'
            self.queue.put(s)
            if s == 'q':
                break
    

class KeyboardController(Skill):
    agent: MobileAgent

    def __init__(self, config: Optional[SkillConfig]=None) -> None:
        config = config or SkillConfig()
        super().__init__(config)

    def reset(self, agent: MobileAgent, obs: Dict, **kwargs):
        super().reset(agent, obs, **kwargs)
        self.thread1 = myThread(1, "thread_1")
        self.thread1.start()

    def act(self, obs, **kwargs) -> Any:
        actions = np.zeros(3)
        while not self.thread1.queue.empty():
            q = self.thread1.queue.get()
            if q == 'q':
                self.close()
            elif q == 'w':
                actions[0] = 1.
            elif q == 'a':
                actions[2] = 1.
            elif q == 's':
                actions[0] = -1.
            elif q == 'd':
                actions[2] = -1.
        return self.agent.set_base_move(actions)

    def close(self):
        if hasattr(self, 'thread1'):
            with lock:
                self.thread1.stop = True
            self.thread1.join()


    def should_terminate(self, obs, **kwargs):
        return self.thread1.stop

    def __del__(self):
        self.close()

        
if __name__ == '__main__':
    thread1 = myThread(1, "thread_1")
    thread1.start()