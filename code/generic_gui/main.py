#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 13 19:20:03 2019

@author: caxenie & amie
"""

import matplot as mat_plot
import sys
from PyQt5.QtWidgets import QApplication , QMainWindow


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ui = mat_plot.MainDialogImgBW()
    ui.show()
    ui.Init()
    sys.exit(app.exec_())