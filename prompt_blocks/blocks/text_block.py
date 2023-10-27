from prompt_blocks.blocks.base_block import PromptBlock

class TextBlock(PromptBlock):
    required_for_pre_generation = []
    required_for_generation = []

    def __init__(self, name, prompt, max_tokens=None, debug=False, **kwargs) -> None:
        super().__init__(name, prompt, max_tokens, debug)

    def _pre_generate(self, **kwargs):
        return None
    
    async def _generate(self, **kwargs) -> str:
        prompt = self.prompt
        prompt_block = '## ' + self.name + '\n' + prompt
        return prompt_block, None