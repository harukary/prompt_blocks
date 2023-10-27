import os, re
from opensearchpy import OpenSearch, RequestsHttpConnection

def remove_tags(text):
    cleaned_text = re.sub('<.*?>', '', text)
    return cleaned_text

class OpenSearchClient(OpenSearch):
    def __init__(self, host, port, index):
        super().__init__(
            hosts=[{'host': host, 'port': port}],
            connection_class=RequestsHttpConnection,
        )
        self.index = index
    
    def search(self, body, scroll=None):
        data = super().search(index=self.index, body=body, scroll=scroll)
        # print(data)
        return data

    def scroll_search(self,body,scroll='1m'):
        data = super().search(index=self.index,body=body,scroll=scroll)
        scroll_id = data['_scroll_id']
        finish = False
        while not finish:
            sc = super().scroll(
                scroll_id = scroll_id,
                scroll = scroll, # time value for search
            )
            if sc['hits']['hits']:
                data['hits']['hits'] += sc['hits']['hits']
            else:
                finish = True
        super().clear_scroll(scroll_id=scroll_id)
        return data