import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QAction, QSplitter, QTreeView, QTabWidget, \
    QToolBar, QTextEdit, QProgressBar, QHBoxLayout, QMenuBar
from PyQt5.QtCore import Qt, QTimer
from qt.menu.Edit_menu import Edit_menu
from qt.menu.File_menu import File_menu


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("if I've come this far, maybe I'm willing to come a little further.")

        # 设置窗口最小大小
        self.setMinimumSize(800, 500)

        # Main container widget
        main_widget = QWidget()
        self.setCentralWidget(main_widget)

        # Main layout
        main_layout = QVBoxLayout()
        main_widget.setLayout(main_layout)
        menu_Bar = QMenuBar()
        # Menu Bar
        file_bar = File_menu(menu_Bar)
        edit_bar = Edit_menu(menu_Bar)
        self.setMenuBar(menu_Bar)

        # Toolbar 1
        model_tool_bar = QToolBar()
        load_model = QAction('load model', self)
        model_tool_bar.addAction(load_model)
        load_agent = QAction('load agent', self)
        load_model.triggered.connect(self.start_task)
        load_agent.triggered.connect(self.start_task)
        model_tool_bar.addAction(load_agent)
        self.addToolBar(model_tool_bar)

        # Splitter
        main_splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(main_splitter)

        # Left pane
        left_pane = QTabWidget()
        navigation_tab = QWidget()
        left_pane.addTab(navigation_tab, '导航栏')

        file_tab = QSplitter(Qt.Vertical)
        tree_view = QTreeView()
        file_tab.addWidget(tree_view)
        left_pane.addTab(file_tab, '文件栏')

        main_splitter.addWidget(left_pane)

        # Right pane
        right_pane = QSplitter(Qt.Vertical)
        # 上面的文本区域 (只显示文本)
        display_text_area = QTextEdit()
        display_text_area.setReadOnly(True)
        display_text_area.setPlainText("这是一个只显示的文本区域。")

        # 下面的文本区域 (允许输入)
        input_text_area = QTextEdit()
        input_text_area.setMinimumHeight(50)  # 设置输入框的最小高度

        right_pane.addWidget(display_text_area)
        right_pane.addWidget(input_text_area)
        right_pane.setSizes([350, 50])  # 设置右侧上下文本区域的初始大小
        main_splitter.addWidget(right_pane)

        # 设置main_splitter的默认比例
        main_splitter.setSizes([150, 750])

        # 添加进度条到水平布局
        progress_layout = QHBoxLayout()
        main_layout.addLayout(progress_layout)

        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedHeight(15)  # 设置进度条的高度
        self.progress_bar.setFixedWidth(self.width() // 3)  # 设置进度条宽度为窗口宽度的一半
        self.progress_bar.setVisible(False)  # 初始隐藏进度条
        progress_layout.addStretch()
        progress_layout.addWidget(self.progress_bar)
        progress_layout.addStretch()

    def start_task(self):
        # 显示进度条并初始化进度
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)

        # 启动定时器来模拟任务进度
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_progress)
        self.timer.start(100)  # 每100毫秒更新一次

    def update_progress(self):
        value = self.progress_bar.value()
        if value < 100:
            self.progress_bar.setValue(value + 1)
        else:
            self.timer.stop()
            self.progress_bar.setVisible(False)  # 隐藏进度条


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
