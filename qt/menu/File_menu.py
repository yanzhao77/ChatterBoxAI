from PyQt5.QtWidgets import QMenuBar, QAction, QMenu


class File_menu(QMenuBar):
    def __init__(self,menu_Bar=QMenuBar):
        super().__init__()
        file_menu = menu_Bar.addMenu('文件')
        file_menu.addAction(QAction('新建', self))
        file_menu.addAction(QAction('打开文件', self))
        recent_menu = QMenu('最近文件', self)
        file_menu.addMenu(recent_menu)
        file_menu.addSeparator()
        # file_menu.addAction(QAction('清空', self))
        # file_menu.addAction(QAction('保存', self))
        # file_menu.addAction(QAction('另存为', self))
        file_menu.addAction(QAction('刷新', self))
        file_menu.addSeparator()
        # script_menu = QMenu('自动化脚本', self)
        # script_menu.addAction(QAction('脚本执行', self))
        # script_menu.addAction(QAction('录制脚本', self))
        # file_menu.addMenu(script_menu)
        file_menu.addSeparator()
        file_menu.addAction(QAction('退出', self))
