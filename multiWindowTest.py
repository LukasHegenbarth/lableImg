import codecs
import distutils.spawn
import os.path
import platform
import re
import subprocess
import sys
from collections import defaultdict
from functools import partial

try:
    from PyQt5.QtCore import *
    from PyQt5.QtGui import *
    from PyQt5.QtWidgets import *
    from PyQt5.QtWebEngineWidgets import QWebEngineView
except ImportError:
    # needed for py3+qt4
    # Ref:
    # http://pyqt.sourceforge.net/Docs/PyQt4/incompatible_apis.html
    # http://stackoverflow.com/questions/21217399/pyqt4-qtcore-qvariant-object-instead-of-a-string
    if sys.version_info.major >= 3:
        import sip
        sip.setapi('QVariant', 2)
    from PyQt4.QtGui import *
    from PyQt4.QtCore import *


class stackedWindow(QWidget):
    def __init__(self):
        super(stackedWindow, self).__init__()
        self.button1 = QPushButton('Data \n Labeling')
        self.button1.setStyleSheet("background-color: #0cdd8c")
        self.button2 = QPushButton('Neural Network \n Training')
        self.button2.setStyleSheet("background-color: #eeeeee")
        self.button3 = QPushButton('Third Page \n')
        self.button3.setStyleSheet("background-color: #eeeeee")

        self.stack1 = QWidget()
        self.stack2 = QWidget()
        self.stack3 = QWidget()

        self.stack1UI()
        self.stack2UI()
        self.stack3UI()

        self.Stack = QStackedWidget(self)
        self.Stack.addWidget(self.stack1)
        self.Stack.addWidget(self.stack2)
        self.Stack.addWidget(self.stack3)

        self.mainVBox = QVBoxLayout(self)

        self.button_layout = QHBoxLayout(self)
        self.button_layout.addWidget(self.button1)
        self.button_layout.addWidget(self.button2)
        self.button_layout.addWidget(self.button3)
        self.button_layout.setAlignment(Qt.AlignLeft)

        self.window_layout = QVBoxLayout(self)
        self.window_layout.addWidget(self.Stack)

        self.mainVBox.addLayout(self.button_layout)
        self.mainVBox.addLayout(self.window_layout)

        self.setLayout(self.mainVBox)
        self.button1.clicked.connect(self.button1_fcn)
        self.button2.clicked.connect(self.button2_fcn)
        self.button3.clicked.connect(self.button3_fcn)
        self.setGeometry(300, 50, 10, 10)
        # self.setWindowTitle('Training Data Pipeline')

    def stack1UI(self):
        layout = QFormLayout()
        label = QLabel('labelImg app will be included here')
        layout.addWidget(label)
        self.stack1.setLayout(layout)

    def stack2UI(self):
        hbox = QHBoxLayout(self)
        #TODO add param list for training config 
        paramVBox = QVBoxLayout(self)
        #add all necessary param fields 
        self.addParamLine(paramVBox, 'batch size')
        self.addParamLine(paramVBox, 'learning rate')
        self.addParamLine(paramVBox, 'total steps')
        self.addParamLine(paramVBox, 'warmup steps')
        self.addParamLine(paramVBox, 'warmup learning rate')
        self.trainingButton = QPushButton('start Training')
        self.trainingStatus = QLabel('Training Status')

        self.trainingStatus.setMaximumWidth(250)
        self.trainingStatus.setStyleSheet("background-color: transparent")
        paramVBox.addWidget(self.trainingButton)
        paramVBox.addWidget(self.trainingStatus)
        paramVBox.setAlignment(Qt.AlignTop)
        self.webEngineView = QWebEngineView()
        self.loadPage()
        hbox.addLayout(paramVBox)
        hbox.addWidget(self.webEngineView)
        self.stack2.setLayout(hbox)

    def stack3UI(self):
        layout = QFormLayout()
        label = QLabel('This page shows nothing yet')
        layout.addWidget(label)
        self.stack3.setLayout(layout)

    def addParamLine(self, layout, param):
        paramBox = QHBoxLayout(self)
        paramName = QLabel(param)
        paramName.setMaximumWidth(150)
        paramName.setStyleSheet("background-color: transparent")
        paramBox.addWidget(paramName)
        lineEdit = QLineEdit()
        lineEdit.setMaximumWidth(200)
        lineEdit.setStyleSheet("background-color: #ffffff")
        paramBox.addWidget(lineEdit)
        layout.addLayout(paramBox)

    def loadPage(self):
        self.webEngineView.load(QUrl('http://localhost:6006/'))

    def button1_fcn(self):
        self.Stack.setCurrentIndex(0)
        self.button1.setStyleSheet("background-color: #0cdd8c")
        self.button2.setStyleSheet("background-color: #eeeeee")
        self.button3.setStyleSheet("background-color: #eeeeee")

    def button2_fcn(self):
        self.Stack.setCurrentIndex(1)
        self.button1.setStyleSheet("background-color: #eeeeee")
        self.button2.setStyleSheet("background-color: #0cdd8c")
        self.button3.setStyleSheet("background-color: #eeeeee")

    def button3_fcn(self):
        self.Stack.setCurrentIndex(2)
        self.button1.setStyleSheet("background-color: #eeeeee")
        self.button2.setStyleSheet("background-color: #eeeeee")
        self.button3.setStyleSheet("background-color: #0cdd8c")


class MainWindow(QMainWindow):
    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        self.setWindowTitle("Data Pipeline")
        stacked_window = stackedWindow()
        self.setCentralWidget(stacked_window)

def main():
    app = QApplication(sys.argv)
    # stacked_window = stackedWindow()
    # # stacked_window.setStyleSheet("background-color: #cfd8dc")
    # stacked_window.setStyleSheet("background-color: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 1," 
    #                              "stop: 0 white, stop: 1 grey);")
    # stacked_window.show()
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
