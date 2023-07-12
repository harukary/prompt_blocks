import json
from prompt_blocks.openai_util import num_tokens_from_string
from prompt_blocks.base_block import PromptBlock
from prompt_blocks.opensearch_client import OpenSearchClient
from prompt_blocks.util import remove_tags, CustomEncoder

class SearchBlock(PromptBlock):
    """
    ベクトル検索結果を入力する

    search_config: 
        endpoint: api endpoint
        body: query body
        headers: query headers
        params: query parameters
    """
    def __init__(self, name, template, max_tokens=None, **kwargs) -> None:
        super().__init__(name, template, max_tokens, **kwargs)
        os_config = kwargs.get('os_config')
        self.client = OpenSearchClient(**os_config)
        format_fn = kwargs.get('format_fn', None)
        if format_fn is not None:
            self.format_data = format_fn

    def generate(self, **kwargs) -> str:
        query = kwargs.get('query', {})
        data = self.search(query)
        prompt_block = '## ' + self.name + '\n' + self.template + '\n'
        for d in data:
            prompt_block = prompt_block + json.dumps(d, indent=2, ensure_ascii=False, cls=CustomEncoder) + '\n\n'
        if self.max_tokens is not None:
            assert self.max_tokens > num_tokens_from_string(prompt_block), num_tokens_from_string(prompt_block)
        return prompt_block, {self.name: data}

    def search(self,query):
        res = self.client.search(query)
        data = [self.format_data(r) for r in res['hits']['hits']]
        return data

    @staticmethod
    def format_data(hit):
        return hit

if __name__ == "__main__":
    import os
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv())

    def format_recipe(hit):
        r = hit['_source']
        id = r['recipeId']
        dish_serving = [d for d in r['content']['dishServings'] if d['unit'] == r['defaultServing']][0]
        instruction_serving = [d for d in r['content']['instructionServings'] if d['unit'] == r['defaultServing']][0]
        ingredients = []
        for d in dish_serving['dishes']:
            for ingredient in d['ingredients']:
                if 'item' in ingredient.keys():
                    for item in ingredient['items']:
                        ingredients.append({'name': item['description'], 'amount': item['quantityText']})
                else:
                    ingredients.append({'name': ingredient['description'], 'amount': ingredient['quantityText']})
                    
        recipe = {
            'name': r['title'],
            'appliance': r['applianceCategory'],
            'cooking_tools': [
                d['cookingTool']
                for d in dish_serving['dishes'] if 'cookingTool' in d.keys() and d['cookingTool'] is not None
            ],
            'dish_type': r['dishType'],
            'nutrition': r['nutrition'],
            'serving': r['defaultServing'],
            'ingredients': ingredients,
            'instructions': [
                {
                    'description': remove_tags(instruction['description']),
                    'notes': [note for note in instruction['notes'] if note != ''] if 'notes' in instruction.keys() else []
                }
                for instruction in instruction_serving['instructions'] 
            ],
            'categories': [c['title'].replace('<br>','') for c in r['categories']]
        }
        return {'id': id, 'recipe': recipe}
    
    search_block = SearchBlock(
        name='Recipes',
        template="",
        max_tokens=4097,
        os_config={
            'host': os.environ['OS_HOST'],
            'port': os.environ['OS_PORT'],
            'index': os.environ['OS_RECIPE_INDEX'],
        },
        format_fn=format_recipe
    )

    query = {
        "size": 1,
        "query":{"bool":{"must":[
            {"term": {"status": "confirmed"}},
            {"term": {"productId": "NE-UBS10A"}},
            {"range": {"releaseDate": {"lt": "2023-06-30"}}}
        ]
    }}}

    print(search_block.generate(query=query))
    