from abc import ABCMeta, abstractmethod
from prompt_blocks.utils.openai_util import num_tokens_from_string

class PromptBlock:
    required_for_generation = []
    def __init__(self, name, prompt, max_tokens=None, debug=False, logger=None, model=None, **kwargs) -> None:
        self.name = name
        self.prompt = prompt
        self.max_tokens = max_tokens
        self.debug = debug
        self.logger = logger
        self.model = model

    async def generate(self, **kwargs) -> str:
        # print('start:',self.name)
        self.check_required_keys(self.required_for_generation, kwargs)
        prompt_block, data = await self._generate(**kwargs)
        self.check_token_limit(prompt_block)
        if self.debug:
            print(prompt_block)
        # print('done:',self.name)
        return prompt_block, data
    
    @abstractmethod
    async def _generate(self, **kwargs):
        return ''
    
    def check_token_limit(self, prompt_block):
        if self.max_tokens is not None:
            assert self.max_tokens > num_tokens_from_string(prompt_block), \
                num_tokens_from_string(prompt_block)
    
    def check_required_keys(self, required, kwargs):
        missing_keys = [r for r in required if r not in kwargs]
        if missing_keys:
            raise ValueError(f"{self.name} requires {', '.join(missing_keys)}.")
    
    
