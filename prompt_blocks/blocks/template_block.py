from prompt_blocks.blocks.base_block import PromptBlock

import re
pattern = r"\{([^}]+)\}"

class TemplateBlock(PromptBlock):
    required_for_pre_generation = []
    required_for_generation = []
    def __init__(self, name, prompt, max_tokens=None, debug=False, **kwargs) -> None:
        super().__init__(name, prompt, max_tokens, debug)
        self.required_for_generation = re.findall(pattern, prompt)

    def _pre_generate(self, **kwargs):
        return None
    
    def _generate(self, **kwargs) -> str:
        prompt = self.format_from_template(**kwargs)
        prompt_block = '## ' + self.name + '\n' + prompt
        return prompt_block, None
    
    def format_from_template(self, **kwargs):
        return self.prompt.format(**kwargs)