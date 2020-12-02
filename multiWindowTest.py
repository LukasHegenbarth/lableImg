import sys
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *


class stackedWindow(QWidget):
    def __init__(self):
        super(stackedWindow, self).__init__()
        self.button1 = QPushButton('Data \n Labeling')
        self.button1.setStyleSheet("background-color: #003c8f")
        self.button2 = QPushButton('Neural Network \n Training')
        self.button3 = QPushButton('Third Page \n')

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

        self.main_vbox = QVBoxLayout(self)

        self.button_layout = QHBoxLayout(self)
        self.button_layout.addWidget(self.button1)
        self.button_layout.addWidget(self.button2)
        self.button_layout.addWidget(self.button3)

        self.window_layout = QVBoxLayout(self)
        self.window_layout.addWidget(self.Stack)

        self.main_vbox.addLayout(self.button_layout)
        self.main_vbox.addLayout(self.window_layout)

        self.setLayout(self.main_vbox)
        self.button1.clicked.connect(self.button1_fcn)
        self.button2.clicked.connect(self.button2_fcn)
        self.button3.clicked.connect(self.button3_fcn)
        self.setGeometry(300, 50, 10, 10)
        # self.setWindowTitle('Training Data Pipeline')

    def stack1UI(self):
        layout = QFormLayout()
        layout.addRow("Name", QLineEdit())
        layout.addRow("Address", QLineEdit())
        self.stack1.setLayout(layout)

    def stack2UI(self):
        layout = QFormLayout()
        sex = QHBoxLayout()
        sex.addWidget(QRadioButton("Male"))
        sex.addWidget(QRadioButton("Female"))
        layout.addRow(QLabel("Sex"), sex)
        layout.addRow("Date of Birth", QLineEdit())

        self.stack2.setLayout(layout)

    def stack3UI(self):
        layout = QHBoxLayout()
        layout.addWidget(QLabel("This Page shows notihing yet"))
        self.stack3.setLayout(layout)

    def button1_fcn(self):
        self.Stack.setCurrentIndex(0)
        self.button1.setStyleSheet("background-color: #003c8f")
        self.button2.setStyleSheet("background-color: #eeeeee")
        self.button3.setStyleSheet("background-color: #eeeeee")

    def button2_fcn(self):
        self.Stack.setCurrentIndex(1)
        self.button1.setStyleSheet("background-color: #eeeeee")
        self.button2.setStyleSheet("background-color: #003c8f")
        self.button3.setStyleSheet("background-color: #eeeeee")

    def button3_fcn(self):
        self.Stack.setCurrentIndex(2)
        self.button1.setStyleSheet("background-color: #eeeeee")
        self.button2.setStyleSheet("background-color: #eeeeee")
        self.button3.setStyleSheet("background-color: #003c8f")


# class MainWindow(QMainWindow):
#     def __init__(self, defaultFilename=None, defaultPrefdefClassFile=None, defaultSaveDir=None):
#         super(MainWindow, self).__init__()
#         self.setWindowTitle("Data Pipeline")

def main():
    app = QApplication(sys.argv)
    stacked_window = stackedWindow()
    stacked_window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
