from prompt_blocks.blocks.base_block import PromptBlock
from prompt_blocks.utils.util import join_history
import datetime

class HistoryBlock(PromptBlock):
    """
    過去の複数の入力に対し
    LangChainのConversationBufferMemory的な
    例：
    Human: ****
    AI: ****
    Human: ****
    AI: 
    """
    required_for_generation = [
        'history',
        'mapping',
    ]
    mapping ={
        "assistant": "Assistant",
        "user": "User",
        "system": "system"
    }
    def __init__(self, name, prompt, max_tokens=None, debug=False, **kwargs) -> None:
        super().__init__(name, prompt, max_tokens, debug)

    async def _generate(self, **kwargs) -> str:
        next_speaker = kwargs.get('next_speaker')
        history = kwargs.get('history')
        now = kwargs.get('now')
        diff = kwargs.get('diff')
        dt = True if now is not None else False
        if diff is not None:
            now += datetime.timedelta(**diff)
        self.mapping = kwargs.get('mapping')
        # print(self.mapping,next_speaker)
        prompt_block = '## ' + self.name + '\n' + self.prompt
        prompt_block = prompt_block + '\n\n' \
            + join_history(history, self.mapping, dt=dt, diff=diff) + '\n' \
                + self.mapping[next_speaker]
        if dt:
            prompt_block += f" [{now.strftime('%Y/%m/%d %H:%M')}]" + ': '
        else:
            prompt_block += ': '
        return prompt_block, None

import asyncio
from prompt_blocks.utils.openai_util import chat_complete_with_backoff

class HistoryWithSummaryBlock(PromptBlock):
    required_for_generation = [
        'history',
        'mapping',
        'num_messages'
    ]
    mapping ={
        "assistant": "Assistant",
        "user": "User",
        "system": "system"
    }
    def __init__(self, name, prompt, max_tokens=None, debug=False, **kwargs) -> None:
        super().__init__(name, prompt, max_tokens, debug)

    async def _generate(self, **kwargs) -> str:
        prompt = kwargs.get('prompt')
        next_speaker = kwargs.get('next_speaker')
        history = kwargs.get('history')
        include_datetime = kwargs.get('include_datetime')
        self.mapping = kwargs.get('mapping')
        now = kwargs.get('now')
        diff = kwargs.get('diff')
        num_messages = kwargs.get('num_messages')
        message_id = kwargs.get('message_id',None)
        dt = True if now is not None and include_datetime else False
        if diff is not None:
            now += datetime.timedelta(**diff)
        if len([h for h in history if h['message']['role']!='system']) > num_messages:
            source_text = join_history(history[:-num_messages], self.mapping, dt=dt, diff=diff)
            messages = [
                {'role': 'system', 'content': source_text},
                {'role': 'system','content': prompt},
            ]
            loop = asyncio.get_running_loop()
            def _chat():
                return chat_complete_with_backoff(messages)#,debug=self.debug)
            summary, log = await loop.run_in_executor(None, _chat)
            if message_id is not None:
                self.logger.add_llm_call(
                    message_id=message_id,
                    model=self.model,
                    messages=messages,
                    response=summary,
                    log=log
                )
        else:
            summary = None
        # print(self.mapping,next_speaker)
        prompt_block = '## ' + self.name + '\n' + self.prompt + '\n\n'
        if summary is not None:
            prompt_block = prompt_block + 'Summary: ' + summary + '\n' \
            
        prompt_block = prompt_block \
            + join_history(history[-num_messages:], self.mapping, dt=dt, diff=diff) + '\n' \
            + self.mapping[next_speaker]
        if dt:
            prompt_block += f" [{now.strftime('%Y/%m/%d %H:%M')}]" + ': '
        else:
            prompt_block += ': '
        return prompt_block, None