import json, asyncio
from prompt_blocks.utils.opensearch_client import OpenSearchClient
from prompt_blocks.blocks.base_block import PromptBlock
from prompt_blocks.utils.util import CustomEncoder
from prompt_blocks.utils.openai_util import chat_complete_with_backoff,get_embedding

class SearchBlock(PromptBlock):
    """
    ベクトル検索結果を入力する

    search_config: 
        endpoint: api endpoint
        body: query body
        headers: query headers
        params: query parameters
    """
    required_for_generation = [
        'prompt',
        'query_source'
    ]
    def __init__(self, name, prompt, logger=None, model=None, max_tokens=None, debug=False, **kwargs) -> None:
        super().__init__(name, prompt, max_tokens, debug, logger, model)
        os_config = kwargs.get('os_config')
        format_fn = kwargs.get('format_fn', None)
        generate_query_fn = kwargs.get('generate_query_fn', None)

        self.client = OpenSearchClient(**os_config)
        if format_fn is not None:
            self.format_data = format_fn
        if generate_query_fn is not None:
            self._generate_query = generate_query_fn

    async def _generate(self, **kwargs):
        query = kwargs.get('query',None)
        if query is None:
            prompt = kwargs.get('prompt')
            query_source = kwargs.get('query_source')
            message_id = kwargs.get('message_id',None)
            query = await self._generate_query(self.model,query_source,prompt,message_id)
        res = self.search(query)
        res_text = self.format_data(res)
        prompt_block = '## ' + self.name + '\n' + self.prompt + '\n' + res_text
        return prompt_block, {'query': res}

    def search(self,query):
        res = self.client.search(query)
        return res
    
    @staticmethod
    async def _generate_query(model,source,prompt,message_id,logger=None):
        messages = [
            {'role': 'system', 'content': source},
            {'role': 'system','content': prompt},
        ]
        loop = asyncio.get_running_loop()
        def _chat():
            return chat_complete_with_backoff(messages)
        response, log = await loop.run_in_executor(None, _chat)
        if logger is not None and message_id is not None:
            logger.add_llm_call(
                message_id=message_id,
                model=model,
                messages=messages,
                response=response,
                log=log
            )
        embedding = get_embedding(response)
        return {
            "size": 2,
            "query": {"knn": {"embedding": {"vector": embedding, "k": 2}}},
            "_source": {"exclude":["embedding"]}
        }

    @staticmethod
    def format_data(res):
        return json.dumps(res, indent=2, ensure_ascii=False, cls=CustomEncoder)