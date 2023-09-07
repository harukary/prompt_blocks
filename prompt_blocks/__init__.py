from prompt_blocks.base_block import *
from prompt_blocks.simple_blocks import *
from prompt_blocks.search_blocks import *
from prompt_blocks.prompt_manager import *

block = {
    'Text': TextBlock,
    'Template': TemplateBlock,
    'History': HistoryBlock,
    'Computing': ComputingBlock,
    'Search': SearchBlock
}

import prompt_blocks.util
import prompt_blocks.opensearch_client
import prompt_blocks.openai_util
import prompt_blocks.langchain_util