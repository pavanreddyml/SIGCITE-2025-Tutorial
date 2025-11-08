import ipywidgets as widgets
from IPython.display import display

class InputFieldWidget:
    def __init__(self, placeholder="Enter URL or text...", description="Input:"):
        self.input_value = None
        self.text_widget = widgets.Text(
            value='',
            placeholder=placeholder,
            description=description,
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='500px')
        )
        self.button = widgets.Button(
            description='Submit',
            button_style='success',
            layout=widgets.Layout(width='100px')
        )
        self.output = widgets.Output()
        
        # Set up event handlers
        self.button.on_click(self._on_submit)
        self.text_widget.observe(self._on_enter, names='value')
    
    def _on_submit(self, b):
        self.input_value = self.text_widget.value.strip()
        with self.output:
            self.output.clear_output()
            if self.input_value:
                print(f"✅ Input captured: {self.input_value}")
            else:
                print("❌ Please enter some text")
    
    def _on_enter(self, change):
        # Auto-capture on Enter key (when value changes)
        if change['new'].strip():
            self.input_value = change['new'].strip()
    
    def display(self):
        input_box = widgets.HBox([self.text_widget, self.button])
        display(input_box, self.output)
    
    def get_value(self):
        return self.input_value
    
    def clear(self):
        self.text_widget.value = ''
        self.input_value = None
        with self.output:
            self.output.clear_output()