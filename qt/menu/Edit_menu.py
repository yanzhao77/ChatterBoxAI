from PyQt5.QtWidgets import QMenuBar, QAction, QMenu


class Edit_menu(QMenuBar):
    def __init__(self,menu_Bar=QMenuBar):
        super().__init__()
        edit_menu = menu_Bar.addMenu('编辑')
        edit_menu.addAction(QAction('Undo', self))
        edit_menu.addAction(QAction('Redo', self))
        edit_menu.addSeparator()
        edit_menu.addAction(QAction('Cut', self))
        edit_menu.addAction(QAction('Copy', self))
        edit_menu.addAction(QAction('Paste', self))
        edit_menu.addAction(QAction('Delete', self))
        edit_menu.addSeparator()
        edit_menu.addAction(QAction('Select All', self))
        edit_menu.addAction(QAction('Unselect All', self))
