import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QTextEdit, QLineEdit, QPushButton
from PyQt5.QtCore import pyqtSignal, QProcess

class PythonConsole(QWidget):
    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        # 创建一个文本编辑控件显示输出
        self.output = QTextEdit()
        self.output.setReadOnly(True)
        layout.addWidget(self.output)

        # 创建一个单行文本编辑控件接收输入
        self.input = QLineEdit()
        self.input.returnPressed.connect(self.executeCommand)
        layout.addWidget(self.input)

        # 创建一个按钮来执行命令
        button = QPushButton('Run')
        button.clicked.connect(self.executeCommand)
        layout.addWidget(button)

        # 设置布局
        self.setLayout(layout)

        # 初始化一个 QProcess 对象
        self.process = QProcess(self)
        self.process.setProgram(sys.executable)
        self.process.readyReadStandardOutput.connect(self.handleStdOut)
        self.process.readyReadStandardError.connect(self.handleStdErr)

    def executeCommand(self):
        command = self.input.text()
        self.input.clear()
        self.process.write(f"{command}\n".encode())
        self.process.closeWriteChannel()

    def handleStdOut(self):
        text = str(self.process.readAllStandardOutput().data(), encoding='utf8')
        self.output.append(text)

    def handleStdErr(self):
        text = str(self.process.readAllStandardError().data(), encoding='utf8')
        self.output.append(text)


if __name__ == '__main__':
    app = QApplication(sys.argv)

    console = PythonConsole()
    console.show()

    sys.exit(app.exec_())