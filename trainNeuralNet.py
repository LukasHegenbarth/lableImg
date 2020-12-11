import codecs
import distutils.spawn
import os.path
import platform
import re
import subprocess
import sys
from collections import defaultdict
from functools import partial

import tensorflow as tf
from google.protobuf import text_format
from object_detection.protos import pipeline_pb2
import glob
import gpu_util

try:
    from PyQt5.QtCore import *
    from PyQt5.QtGui import *
    from PyQt5.QtWebEngineWidgets import QWebEngineView
    from PyQt5.QtWidgets import *
except ImportError:
    # needed for py3+qt4
    # Ref:
    # http://pyqt.sourceforge.net/Docs/PyQt4/incompatible_apis.html
    # http://stackoverflow.com/questions/21217399/pyqt4-qtcore-qvariant-object-instead-of-a-string
    if sys.version_info.major >= 3:
        import sip
        sip.setapi('QVariant', 2)
    from PyQt4.QtCore import *
    from PyQt4.QtGui import *


def get_configs_from_pipeline_file(pipeline_config_path, config_override=None):
    '''
    read .config and convert it to proto_buffer_object
    '''

    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
    with tf.io.gfile.GFile(pipeline_config_path, "r") as f:
        proto_str = f.read()
        text_format.Merge(proto_str, pipeline_config)
    if config_override:
        text_format.Merge(config_override, pipeline_config)
    #print(pipeline_config)
    return pipeline_config


class stackedWindow(QWidget):
    def __init__(self):
        super(stackedWindow, self).__init__()
        self.button1 = QPushButton('Data \n Labeling')
        self.button1.setStyleSheet("background-color: #00675b")
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
        paramVBox = QVBoxLayout(self)

        self.configs = get_configs_from_pipeline_file('/home/lukas/training_workspace/pretrained_models/ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8/pipeline.config')
        #add all necessary param fields
        self.fineTuneCheckpoint = self.addParamLine(paramVBox, 'fine tune checkpoint', self.configs.train_config.fine_tune_checkpoint)
        self.numClasses = self.addParamLine(paramVBox, 'num classes', str(self.configs.model.ssd.num_classes))
        self.batchSize = self.addParamLine(paramVBox, 'batch size', str(self.configs.train_config.batch_size))
        self.learningRate = self.addParamLine(paramVBox, 'learning rate',
                          str(self.configs.train_config.optimizer.momentum_optimizer.
                          learning_rate.cosine_decay_learning_rate.
                          learning_rate_base))
        self.warmupLearningRate = self.addParamLine(paramVBox, 'warmup learning rate', 
                          str(self.configs.train_config.optimizer.momentum_optimizer.
                          learning_rate.cosine_decay_learning_rate.
                          warmup_learning_rate))
        self.totalSteps = self.addParamLine(paramVBox, 'total steps', 
                          str(self.configs.train_config.optimizer.momentum_optimizer.
                          learning_rate.cosine_decay_learning_rate.
                          total_steps))
        self.warmupSteps = self.addParamLine(paramVBox, 'warmup steps',
                          str(self.configs.train_config.optimizer.momentum_optimizer.
                          learning_rate.cosine_decay_learning_rate.
                          warmup_steps))

        #label map pbtxt selection
        # self.labelMapPath = self.addParamLine(paramVBox, 'label map path', self.configs.train_input_reader.label_map_path)
        paramBoxLabelMap = QHBoxLayout(self)
        paramLabelMap = QLabel('label map')
        paramLabelMap.setMaximumWidth(150)
        paramLabelMap.setStyleSheet("background-color: transparent")
        paramBoxLabelMap.addWidget(paramLabelMap)
        comboBoxLabelMap = QComboBox()
        comboBoxLabelMap.setMaximumWidth(410)
        for item in glob.glob('/home/lukas/training_workspace/data/*/*.pbtxt'):
            comboBoxLabelMap.addItem(item)
        paramVBox.addWidget(paramLabelMap)
        paramVBox.addWidget(comboBoxLabelMap)
        
        
        
        #training data selection
        paramBox = QHBoxLayout(self)
        paramName = QLabel('training data')
        paramName.setMaximumWidth(150)
        paramName.setStyleSheet("background-color: transparent")
        paramBox.addWidget(paramName)
        self.selectAll = QCheckBox()
        self.selectAll.setCheckState(Qt.Checked)
        self.selectAll.stateChanged.connect(self.select_all)
        hBox = QHBoxLayout(self)
        hBox.addWidget(QLabel('select all'))
        hBox.addWidget(self.selectAll)
        hBox.setAlignment(Qt.AlignRight)
        paramBox.addLayout(hBox)
        paramVBox.addLayout(paramBox)

        #checkable combobox for training data selection
        annotation_records = glob.glob('/home/lukas/training_workspace/data/beet/TFRecords/*.record')
        self.trainingData = CheckableComboBox()
        self.trainingData.addItems(annotation_records)
        self.trainingData.selectAll()
        self.trainingData.setMaximumWidth(410)   
        paramVBox.addWidget(self.trainingData)

        self.trainingButton = QPushButton('start Training')
        self.trainingButton.clicked.connect(self.start_training)
        paramVBox.addWidget(self.trainingButton)
        self.trainingStatus = QLabel('Training Status')
        

        self.trainingStatus.setMaximumWidth(250)
        self.trainingStatus.setStyleSheet("background-color: transparent")
        
        paramVBox.addWidget(self.trainingStatus)
        self.trainingProgress = QProgressBar()
        self.trainingProgress.setMaximumWidth(400)
        paramVBox.addWidget(self.trainingProgress)
        paramVBox.setAlignment(Qt.AlignTop)
        self.webEngineView = QWebEngineView()
        self.loadPage()
        hbox.addLayout(paramVBox)
        hbox.addWidget(self.webEngineView)
        self.stack2.setLayout(hbox)

        self.gpu_info = gpu_util.Gpu_Info()
        self.memoryTotal = QLabel()
        self.memoryUsed = QLabel()
        self.memoryFree = QLabel()
        self.maxUsage = QLabel()
        self.minUsage = QLabel()
        self.avgUsage = QLabel()
        
        paramVBox.addWidget(self.memoryTotal)
        paramVBox.addWidget(self.memoryUsed)
        paramVBox.addWidget(self.memoryFree)
        paramVBox.addWidget(self.minUsage)
        paramVBox.addWidget(self.maxUsage)
        paramVBox.addWidget(self.avgUsage)


        self.timer = QTimer()
        self.timer.setInterval(250)
        self.timer.timeout.connect(self.update_gpu_data)
        self.timer.start()

    def stack3UI(self):
        layout = QFormLayout()
        label = QLabel('This page shows nothing yet')
        layout.addWidget(label)
        self.stack3.setLayout(layout)

    def addParamLine(self, layout, param, default_text=None):
        paramBox = QHBoxLayout(self)
        paramName = QLabel(param)
        paramName.setMaximumWidth(150)
        paramName.setStyleSheet("background-color: transparent")
        paramBox.addWidget(paramName)
        lineEdit = QLineEdit()
        lineEdit.setText(default_text)
        lineEdit.setMaximumWidth(250)
        lineEdit.setStyleSheet("background-color: #ffffff")
        paramBox.addWidget(lineEdit)
        layout.addLayout(paramBox)
        return lineEdit

    def loadPage(self):
        self.webEngineView.load(QUrl('http://localhost:6006/'))

    def button1_fcn(self):
        self.Stack.setCurrentIndex(0)
        self.button1.setStyleSheet("background-color: #00675b")
        self.button2.setStyleSheet("background-color: #eeeeee")
        self.button3.setStyleSheet("background-color: #eeeeee")

    def button2_fcn(self):
        self.Stack.setCurrentIndex(1)
        self.button1.setStyleSheet("background-color: #eeeeee")
        self.button2.setStyleSheet("background-color: #00675b")
        self.button3.setStyleSheet("background-color: #eeeeee")

    def button3_fcn(self):
        self.Stack.setCurrentIndex(2)
        self.button1.setStyleSheet("background-color: #eeeeee")
        self.button2.setStyleSheet("background-color: #eeeeee")
        self.button3.setStyleSheet("background-color: #00675b")

    def start_training(self):
        print('starting training')
        self.configs.train_config.fine_tune_checkpoint = self.fineTuneCheckpoint.text()
        self.configs.train_input_reader.label_map_path = self.labelMapPath.text()
        self.configs.model.ssd.num_classes = int(self.numClasses.text())
        self.configs.train_config.batch_size = int(self.batchSize.text())
        self.configs.train_config.optimizer.momentum_optimizer.learning_rate.cosine_decay_learning_rate.learning_rate_base = float(self.learningRate.text())
        self.configs.train_config.optimizer.momentum_optimizer.learning_rate.cosine_decay_learning_rate.warmup_learning_rate = float(self.warmupLearningRate.text())
        self.configs.train_config.optimizer.momentum_optimizer.learning_rate.cosine_decay_learning_rate.total_steps = int(self.totalSteps.text())
        self.configs.train_config.optimizer.momentum_optimizer.learning_rate.cosine_decay_learning_rate.warmup_steps = int(self.warmupSteps.text())
        for data in self.trainingData.currentData():
            self.configs.train_input_reader.tf_record_input_reader.input_path.append(data)

        config_text = text_format.MessageToString(self.configs)
        #TODO create path in sessions folder 
        self.write_configs_to_new_file(config_text, '/home/lukas/coding/labelImg/pipeline new.config')

        #command for training
        # python3 model_main_tf2.py --model_dir=models/my_ssd_mobilenet/ --pipeline_config_path=models/my_ssd_mobilenet/pipeline.config 

    def update_gpu_data(self):
        info = self.gpu_info.get()
        self.memoryTotal.setText('Memory Total:' + str(info[0]))
        self.memoryUsed.setText('Memory Used: ' + str(info[1])) 
        self.memoryFree.setText('Memory Free: ' + str(info[2]))
        self.maxUsage.setText('Max Usage: ' + str(info[3]))
        self.minUsage.setText('Min Usage: ' + str(info[4]))
        self.avgUsage.setText('Avg Usage: ' + str(info[5]))


    def select_all(self):
        if self.selectAll.checkState() == Qt.Checked:
            self.trainingData.selectAll()
        if self.selectAll.checkState() == Qt.Unchecked:   
            self.trainingData.deselectAll()

    def write_configs_to_new_file(self, configs, new_file):
        with tf.io.gfile.GFile(new_file, "w") as f:
            f.write(configs)
        f.close()

    

class CheckableComboBox(QComboBox):

    # Subclass Delegate to increase item height
    class Delegate(QStyledItemDelegate):
        def sizeHint(self, option, index):
            size = super().sizeHint(option, index)
            size.setHeight(20)
            return size

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Make the combo editable to set a custom text, but readonly
        self.setEditable(True)
        self.lineEdit().setReadOnly(True)
        # Make the lineedit the same color as QPushButton
        palette = qApp.palette()
        palette.setBrush(QPalette.Base, palette.button())
        self.lineEdit().setPalette(palette)

        # Use custom delegate
        self.setItemDelegate(CheckableComboBox.Delegate())

        # Update the text when an item is toggled
        self.model().dataChanged.connect(self.updateText)

        # Hide and show popup when clicking the line edit
        self.lineEdit().installEventFilter(self)
        self.closeOnLineEditClick = False

        # Prevent popup from closing when clicking on an item
        self.view().viewport().installEventFilter(self)

    def resizeEvent(self, event):
        # Recompute text to elide as needed
        self.updateText()
        super().resizeEvent(event)

    def eventFilter(self, object, event):

        if object == self.lineEdit():
            if event.type() == QEvent.MouseButtonRelease:
                if self.closeOnLineEditClick:
                    self.hidePopup()
                else:
                    self.showPopup()
                return True
            return False

        if object == self.view().viewport():
            if event.type() == QEvent.MouseButtonRelease:
                index = self.view().indexAt(event.pos())
                item = self.model().item(index.row())

                if item.checkState() == Qt.Checked:
                    item.setCheckState(Qt.Unchecked)
                else:
                    item.setCheckState(Qt.Checked)
                return True
        return False

    def showPopup(self):
        super().showPopup()
        # When the popup is displayed, a click on the lineedit should close it
        self.closeOnLineEditClick = True

    def hidePopup(self):
        super().hidePopup()
        # Used to prevent immediate reopening when clicking on the lineEdit
        self.startTimer(100)
        # Refresh the display text when closing
        self.updateText()

    def timerEvent(self, event):
        # After timeout, kill timer, and reenable click on line edit
        self.killTimer(event.timerId())
        self.closeOnLineEditClick = False

    def updateText(self):
        texts = []
        for i in range(self.model().rowCount()):
            if self.model().item(i).checkState() == Qt.Checked:
                texts.append(self.model().item(i).text())
        text = ", ".join(texts)

        # Compute elided text (with "...")
        metrics = QFontMetrics(self.lineEdit().font())
        elidedText = metrics.elidedText(text, Qt.ElideRight, self.lineEdit().width())
        self.lineEdit().setText(elidedText)

    def addItem(self, text, data=None):
        item = QStandardItem()
        item.setText(text)
        if data is None:
            item.setData(text)
        else:
            item.setData(data)
        item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsUserCheckable)
        item.setData(Qt.Unchecked, Qt.CheckStateRole)
        self.model().appendRow(item)

    def addItems(self, texts, datalist=None):
        for i, text in enumerate(texts):
            try:
                data = datalist[i]
            except (TypeError, IndexError):
                data = None
            self.addItem(text, data)

    def currentData(self):
        # Return the list of selected items data
        res = []
        for i in range(self.model().rowCount()):
            if self.model().item(i).checkState() == Qt.Checked:
                res.append(self.model().item(i).data())
        return res

    def selectAll(self):
        for i in range(self.model().rowCount()):
            self.model().item(i).setCheckState(Qt.Checked)

    def deselectAll(self):
        for i in range (self.model().rowCount()):
            self.model().item(i).setCheckState(Qt.Unchecked)

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
