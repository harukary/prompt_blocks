from abc import ABCMeta

class PromptBlock:
    def __init__(self, name, template, max_tokens=None, **kwargs) -> None:
        self.name = name
        self.template = template
        self.max_tokens = max_tokens

    def generate(self, **kwargs) -> str:
        return ''
