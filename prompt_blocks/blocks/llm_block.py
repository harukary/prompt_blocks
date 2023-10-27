from prompt_blocks.blocks.base_block import PromptBlock
from prompt_blocks.utils.openai_util import chat_complete_with_backoff

import asyncio

class LLMBlock(PromptBlock):
    """
    言語計算プロンプトにより入力を変換する
    例：要約、翻訳、構造化など
    """
    required_for_generation = [
        'prompt',
        'source_text'
    ]
    def __init__(self, name, prompt, logger=None, model=None, max_tokens=None, debug=False, **kwargs) -> None:
        super().__init__(name, prompt, max_tokens, debug, logger, model)
    
    async def _generate(self, **kwargs) -> str:
        prompt = kwargs.get('prompt')
        source_text = kwargs.get('source_text')
        message_id = kwargs.get('message_id',None)
        messages = [
            {'role': 'system', 'content': source_text},
            {'role': 'system','content': prompt},
        ]
        loop = asyncio.get_running_loop()
        def _chat():
            return chat_complete_with_backoff(messages,debug=self.debug)
        response, log = await loop.run_in_executor(None, _chat)
        if message_id is not None:
            self.logger.add_llm_call(
                message_id=message_id,
                model=self.model,
                messages=messages,
                response=response,
                log=log
            )
        # self.response, log = chat_complete_with_backoff(messages,debug=self.debug)
        prompt_block = '## ' + self.name + '\n' + self.prompt + '\n' + response
        return prompt_block, None