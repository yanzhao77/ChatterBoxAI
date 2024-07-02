from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.textinput import TextInput
import PyQt5

class TodoApp(BoxLayout):
    def __init__(self, **kwargs):
        super(TodoApp, self).__init__(**kwargs)
        self.orientation = 'vertical'
        self.padding = 10
        self.input_text = TextInput(multiline=False)
        self.add_widget(self.input_text)
        self.add_button = Button(text='Add', size_hint=(1, 0.2))
        self.add_button.bind(on_press=self.add_todo)
        self.add_widget(self.add_button)
        self.todo_list = BoxLayout(orientation='vertical')
        self.add_widget(self.todo_list)
    def add_todo(self, instance):
        todo_text = self.input_text.text.strip()
        if todo_text:
            todo_label = Button(text=todo_text, size_hint=(1, 0.2))
            self.todo_list.add_widget(todo_label)
            self.input_text.text = ''
class MyApp(App):
    def build(self):
        return TodoApp()
if __name__ == '__main__':
    MyApp().run()