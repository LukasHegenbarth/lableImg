import sys
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

class stackedExample(QWidget):

   def __init__(self):
      super(stackedExample, self).__init__()
      self.leftlist = QListWidget()
    #   self.leftlist.insertItem (0, 'Contact' )
    #   self.leftlist.insertItem (1, 'Personal' )
    #   self.leftlist.insertItem (2, 'Educational' )
      self.button1 = QPushButton('Button 1')
      self.button2 = QPushButton('Button 2')
      self.button3 = QPushButton('Button 3')
		
      self.stack1 = QWidget()
      self.stack2 = QWidget()
      self.stack3 = QWidget()
		
      self.stack1UI()
      self.stack2UI()
      self.stack3UI()
		
      self.Stack = QStackedWidget (self)
      self.Stack.addWidget (self.stack1)
      self.Stack.addWidget (self.stack2)
      self.Stack.addWidget (self.stack3)
		
      hbox = QHBoxLayout(self)
      hbox.addWidget(self.button1)
      hbox.addWidget(self.button2)
      hbox.addWidget(self.button3)
    #   hbox.addWidget(self.leftlist)
      hbox.addWidget(self.Stack)

      self.setLayout(hbox)
    #   self.leftlist.currentRowChanged.connect(self.display)
      self.button1.clicked.connect(self.button1_fcn)
      self.button2.clicked.connect(self.button2_fcn)
      self.button3.clicked.connect(self.button3_fcn)
      self.setGeometry(300, 50, 10,10)
      self.setWindowTitle('StackedWidget demo')
      self.show()
		
   def stack1UI(self):
      layout = QFormLayout()
      layout.addRow("Name",QLineEdit())
      layout.addRow("Address",QLineEdit())
      #self.setTabText(0,"Contact Details")
      self.stack1.setLayout(layout)
		
   def stack2UI(self):
      layout = QFormLayout()
      sex = QHBoxLayout()
      sex.addWidget(QRadioButton("Male"))
      sex.addWidget(QRadioButton("Female"))
      layout.addRow(QLabel("Sex"),sex)
      layout.addRow("Date of Birth",QLineEdit())
		
      self.stack2.setLayout(layout)
		
   def stack3UI(self):
      layout = QHBoxLayout()
      layout.addWidget(QLabel("subjects"))
      layout.addWidget(QCheckBox("Physics"))
      layout.addWidget(QCheckBox("Maths"))
      self.stack3.setLayout(layout)
		
#    def display(self,i):
#       self.Stack.setCurrentIndex(i)
    
   def button1_fcn(self):
      self.Stack.setCurrentIndex(0)
   
   def button2_fcn(self):
      self.Stack.setCurrentIndex(1)

   def button3_fcn(self):
      self.Stack.setCurrentIndex(2)
		
def main():
   app = QApplication(sys.argv)
   ex = stackedExample()
   sys.exit(app.exec_())
	
if __name__ == '__main__':
   main()