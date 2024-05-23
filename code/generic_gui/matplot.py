from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import numpy as np
from UI import Ui_Dialog

import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

import function as fun_c
from PyQt5.QtCore import pyqtSignal, QObject, Qt, pyqtSlot
from PyQt5.QtWidgets import QWidget, QApplication, QGroupBox, QPushButton, QLabel, QCheckBox, QSpinBox, QHBoxLayout, QComboBox, QGridLayout
import mainthread as Mainthread
from time import ctime, sleep
import os


class MyFigure(FigureCanvas):
    def __init__(self,width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super(MyFigure,self).__init__(self.fig)
        self.axes = self.fig.add_subplot(111)

class MainDialogImgBW(QDialog,Ui_Dialog):
   
    #define signals
    valueChanged = pyqtSignal()
    change_Neurons = pyqtSignal(str)
    change_EpochsValue = pyqtSignal(str)
    change_SAMPLESNum = pyqtSignal(str)
    change_ETAValue = pyqtSignal(str)
    change_XIValues = pyqtSignal(str)
    change_SensoryIndex = pyqtSignal(str)
    change_pref = pyqtSignal(str)
     
    def __init__(self):
        super(MainDialogImgBW,self).__init__()
        
        
        self.startflag = False
        self.loopflag = True
        self.setupUi(self)
        self.setWindowTitle("Dynamic Display")
        self.setMinimumSize(0,0)
        self.gridlayout1 = QGridLayout(self.groupBox1)
        self.gridlayout2 = QGridLayout(self.groupBox2)
        self.gridlayout3 = QGridLayout(self.groupBox3)
        self.gridlayout4 = QGridLayout(self.groupBox4)
        
        
    def Init(self):
        Thread = Mainthread.myThread()
        
        # define signals and connects 
        self.valueChanged.connect(self.handle_valueChanged)   # self.valueChanged.emit()
        self.change_Neurons[str].connect(Thread.F.change_NeuronsNum)
        self.change_EpochsValue[str].connect(Thread.F.change_EpochsValue)
        self.change_SAMPLESNum[str].connect(Thread.F.change_SAMPLESNum)
        self.change_ETAValue[str].connect(Thread.F.change_ETAValue)
        self.change_XIValues[str].connect(Thread.F.change_XIValue)
        self.change_SensoryIndex[str].connect(Thread.F.change_SensoryIndex)
        self.change_pref[str].connect(Thread.F.change_pref)
        
        Thread.F.senddata[dict,dict,int,list].connect(self.plotdata)
        Thread.F.senddata1[dict,dict,int,list].connect(self.plotdata1)
        
        self.pushbutton1.clicked.connect(Thread.F.changevisualize) #
        self.pushbutton2.clicked.connect(Thread.F.changevisualize1) #
        self.pushbutton3.clicked.connect(self.showDialog) #
        self.pushbutton4.clicked.connect(self.showDialog) #
        self.pushbutton5.clicked.connect(self.showDialog) #
        self.pushbutton6.clicked.connect(self.showDialog) #
        self.pushbutton7.clicked.connect(self.showDialog) #
        self.pushbutton8.clicked.connect(self.showDialog) # 
        self.pushbutton9.clicked.connect(self.showDialog)
        self.pushButton.clicked.connect(Thread.F.change_trainstartflag)
        self.pushButton_2.clicked.connect(self.kill_thread)

        Thread.start()


    def kill_thread(self):
        pid = os.getpid()
        cmd = 'kill ' + str(pid)
        try:
            os.system(cmd)
            print(pid, 'killed')
        except Exception as e:
            print(e)


    
    def change_startflag(self):
        self.startflag = True
    
    
    def showDialog(self):
        sender = self.sender()
        if sender == self.pushbutton3:
            text,ok = QInputDialog.getText(self, 'Modify Neurons', 'Input the number of Neurons:')
            if ok:
                self.label3.setText(text)
                self.change_Neurons.emit(text)
        
        elif sender == self.pushbutton4:
            text,ok = QInputDialog.getText(self, 'Modify Epochs', 'Input Epochs value:')
            if ok:
                self.label4.setText(text)
                self.change_EpochsValue.emit(text)
        elif sender == self.pushbutton5:
            text,ok = QInputDialog.getText(self, 'Modify SAMPLES', 'Input the number of SAMPLES:')
            if ok:
                self.label5.setText(text)   
                self.change_SAMPLESNum.emit(text)
        elif sender == self.pushbutton6:
            text,ok = QInputDialog.getText(self, 'Modify ETA', 'Input ETA value:')
            if ok:
                self.label6.setText(text)
                self.change_ETAValue.emit(text)
        elif sender == self.pushbutton7:
            text,ok = QInputDialog.getText(self, 'Modify XI', 'Input XI value:')
            if ok:
                self.label7.setText(text)
                self.change_XIValues.emit(text)
        elif sender == self.pushbutton8:
            text,ok = QInputDialog.getText(self, 'Modify Sensory', 'Input Sensory Index:')
            if ok:
                self.label8.setText(text)
                self.change_SensoryIndex.emit(text)
        
        elif sender == self.pushbutton9:
            text,ok = QInputDialog.getText(self, 'Modify Sensory', 'Input Sensory Index:')
            if ok:
                self.label9.setText(text)
                self.change_pref.emit(text)


    def setvisualize(self):
        print("chengaopigo asjfajiogajopseg")
    
    def handle_valueChanged(self):
        print('dasdasfasfasfas')
        
    def plotdata(self,pop,sdata,index,s_pref):
        print('the first sensor pic is drawing')
        F = MyFigure(width=3, height=2, dpi=100)     
        self.gridlayout1.addWidget(F,0,1)
        F.axes = F.fig.add_subplot(411)
        #####   ax1 = plt.subplot(4,1,1) 
        if pop['idx'] == 1:
            F.axes.hist(sdata['x'],bins=50,facecolor="blue", edgecolor="black", alpha=0.7)            
        elif pop['idx'] == 2:
            F.axes.hist(sdata['y'],bins=50,facecolor="blue", edgecolor="black", alpha=0.7)
        #####ax2 = plt.subplot(4,1,2)
        F.axes = F.fig.add_subplot(412)
        x = np.linspace(-sdata['range'],sdata['range'],pop['lsize'])
        for idx in range(pop['lsize']):
            # extract the preferred values (wight vector) of each neuron
            v_pref = pop['Winput'][idx]
            fx = np.exp(-(x - v_pref)**2/(2*pop['s'][idx]**2))
            F.axes.plot([x for x in range(pop['lsize'])],fx,linewidth=3)
        pop['Winput'] = np.sort(pop['Winput'],axis=0)  #notice keng

        # ax3 = plt.subplot(4,1,3)
        F.axes = F.fig.add_subplot(413)
        F.axes.hist(pop['Winput'],bins=50,facecolor="blue", edgecolor="black", alpha=0.7)
        # ax4 = plt.subplot(4,1,4)
        F.axes = F.fig.add_subplot(414)
        F.axes.plot(pop['s'],'.r')
        
        F1 = MyFigure(width=3, height=2, dpi=100)        
        self.gridlayout2.addWidget(F1,0,1)
        F1.axes = F1.fig.add_subplot(211)      
        F1.axes.plot(sdata['x'],sdata['y'],'.g')
        F1.axes2 = F1.fig.add_subplot(212)
        id_maxv = np.zeros((pop['lsize'],1))   #(100,1)
        for idx in range(pop['lsize']):
            arr =  pop['Wcross'][idx,:]
            id_maxv[idx] = np.where(arr==np.max(arr))
        F1.axes2.imshow(pop['Wcross'].T)
        
        
        F2 = MyFigure(width=3, height=2, dpi=100)
        self.gridlayout3.addWidget(F2,0,1)
        F2.axes = F2.fig.add_subplot(111)   
        v_pref = np.sort(pop['Winput'],axis=0)
        pref= s_pref
        for idx in range(len(pref)):
            # extract the preferred values (wight vector) of each neuron
            idx_pref = pref[idx]
            fx = np.exp(-(x - v_pref[idx_pref])**2/(2*pop['s'][idx_pref]**2))
            F2.axes.plot([x for x in range(pop['lsize'])],fx,linewidth=3)
        
        
        

             

    def plotdata1(self,pop,sdata,index,s_pref):
        print('Drawing(sensory_2)')
        F = MyFigure(width=3, height=2, dpi=100)     
        self.gridlayout1.addWidget(F,0,1)
        F.axes = F.fig.add_subplot(411)
        #####   ax1 = plt.subplot(4,1,1) 
        if pop['idx'] == 1:
            F.axes.hist(sdata['x'],bins=50,facecolor="blue", edgecolor="black", alpha=0.7)            
        elif pop['idx'] == 2:
            F.axes.hist(sdata['y'],bins=50,facecolor="blue", edgecolor="black", alpha=0.7)
        #####ax2 = plt.subplot(4,1,2)
        F.axes = F.fig.add_subplot(412)
        x = np.linspace(-sdata['range'],sdata['range'],pop['lsize'])
        for idx in range(pop['lsize']):
            # extract the preferred values (wight vector) of each neuron
            v_pref = pop['Winput'][idx]
            fx = np.exp(-(x - v_pref)**2/(2*pop['s'][idx]**2))
            F.axes.plot([x for x in range(pop['lsize'])],fx,linewidth=3)
        pop['Winput'] = np.sort(pop['Winput'],axis=0)  #notice keng

        # ax3 = plt.subplot(4,1,3)
        F.axes = F.fig.add_subplot(413)
        F.axes.hist(pop['Winput'],bins=50,facecolor="blue", edgecolor="black", alpha=0.7)
        # ax4 = plt.subplot(4,1,4)
        F.axes = F.fig.add_subplot(414)
        F.axes.plot(pop['s'],'.r')
        
        F1 = MyFigure(width=3, height=2, dpi=100)        
        self.gridlayout2.addWidget(F1,0,1)
        F1.axes = F1.fig.add_subplot(211)      
        F1.axes.plot(sdata['x'],sdata['y'],'.g')
        F1.axes2 = F1.fig.add_subplot(212)
        id_maxv = np.zeros((pop['lsize'],1))   #(100,1)
        for idx in range(pop['lsize']):
            arr =  pop['Wcross'][idx,:]
            id_maxv[idx] = np.where(arr==np.max(arr))
        F1.axes2.imshow(pop['Wcross'].T)
        
        
        F2 = MyFigure(width=3, height=2, dpi=100)
        self.gridlayout3.addWidget(F2,0,1)
        F2.axes = F2.fig.add_subplot(111)   
        v_pref = np.sort(pop['Winput'],axis=0)
        pref= s_pref
        for idx in range(len(pref)):
            # extract the preferred values (wight vector) of each neuron
            idx_pref = pref[idx]
            fx = np.exp(-(x - v_pref[idx_pref])**2/(2*pop['s'][idx_pref]**2))
            F2.axes.plot([x for x in range(pop['lsize'])],fx,linewidth=3)
        
























