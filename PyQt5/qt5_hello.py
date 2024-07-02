from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout


def qt5_hello():
    app = QApplication([])
    window = QWidget()
    window.setWindowTitle('PyQt5 Example')

    layout = QVBoxLayout()
    button = QPushButton('Say Hello')
    button.clicked.connect(say_hello)
    layout.addWidget(button)

    window.setLayout(layout)
    window.show()
    app.exec_()


def say_hello():
    print("Hello, World!")


if __name__ == '__main__':
    qt5_hello()
