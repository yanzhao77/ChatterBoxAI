from PyQt5.QtWidgets import QMenuBar, QAction, QMenu


class Help_menu(QMenuBar):
    def __init__(self,menu_Bar=QMenuBar):
        super().__init__()
        help_menu = menu_Bar.addMenu('Help')
        help_menu.addAction(QAction('About MyHelloApp', self))