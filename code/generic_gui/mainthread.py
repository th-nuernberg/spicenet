#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 13 20:13:24 2019

@author: caxenie & amie

"""

import threading
import time
import function as fun_c



class myThread (threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.F = fun_c.Func()
        
    def run(self):
        print ("start a new thread： " )
        threadLock = threading.Lock()
        
        threadLock.acquire()
        
        self.F.main_thread(train=True)
        
        threadLock.release()
        
    def terminate(self):
        self._running = False




