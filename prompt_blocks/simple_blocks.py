from prompt_blocks.openai_util import num_tokens_from_string
from prompt_blocks.base_block import PromptBlock

class TextBlock(PromptBlock):
    def __init__(self, name, template, max_tokens=None, **kwargs) -> None:
        super().__init__(name, template, max_tokens)

    def generate(self, **kwargs) -> str:
        prompt = self.template
        prompt_block = '## ' + self.name + '\n' + prompt
        if self.max_tokens is not None:
            assert self.max_tokens > num_tokens_from_string(prompt_block), num_tokens_from_string(prompt_block)
        return prompt_block, None

class TemplateBlock(PromptBlock):
    def __init__(self, name, template, max_tokens=None, **kwargs) -> None:
        super().__init__(name, template, max_tokens)

    def generate(self, **kwargs) -> str:
        prompt = self.format_from_template(**kwargs)
        prompt_block = '## ' + self.name + '\n' + prompt
        if self.max_tokens is not None:
            assert self.max_tokens > num_tokens_from_string(prompt_block), num_tokens_from_string(prompt_block)
        return prompt_block, None
    
    def format_from_template(self, **kwargs):
        return self.template.format(**kwargs)

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
    def __init__(self, name, template, max_tokens=None, **kwargs) -> None:
        super().__init__(name, template, max_tokens)

    def generate(self, **kwargs) -> str:
        prompt = self.template
        prompt_block = '## ' + self.name + '\n' + prompt
        prompt_block = prompt_block + '\n\n' + self.join_history(kwargs['history'], kwargs['mapping'])
        if self.max_tokens is not None:
            assert self.max_tokens > num_tokens_from_string(prompt_block), num_tokens_from_string(prompt_block)
        return prompt_block, None
    
    def join_history(self, history, mapping=None):
        if mapping is None:
            return '\n'.join([h['role']+': '+h['content'] for h in history]) + '\nassistant: '
        else:
            return '\n'.join([mapping[h['role']]+': '+h['content'] for h in history]) + '\n' + mapping['assistant'] + ': '

class ComputingBlock(PromptBlock):
    """
    言語計算プロンプトにより入力を変換する
    例：要約、翻訳、構造化など
    """
    def __init__(self, name, template, max_tokens=None, **kwargs) -> None:
        super().__init__(name, template, max_tokens, **kwargs)
    
    def generate(self, **kwargs) -> str:
        text_block = kwargs.get('block')
        blocks = []
        blocks.append(text_block.generate()) 
        blocks.append('## ' + self.name + '\n' + self.template)
        prompt_block = '\n\n'.join(blocks)
        if self.max_tokens is not None:
            assert self.max_tokens > num_tokens_from_string(prompt_block), num_tokens_from_string(prompt_block)
        return prompt_block, None

if __name__ == "__main__":
    # p = TextBlock('Instruction', '[Knowledge]をもとに、[Question]に回答してください。', 200)
    # print(p.generate())

    # p = TemplateBlock('Instruction', '私の名前は{name}です。', 200)
    # print(p.generate(name='いのうえ'))

    history = [
        {'role': 'assistant', 'content': 'こんにちは。元気ですか？'},
        {'role': 'user', 'content': 'こんにちは。'},
    ]
    p = HistoryBlock('Conversation history', '以下はAIとHumanの会話です。', 200)
    print(p.generate(history=history, mapping={'system': 'system', 'user': 'Human', 'assistant': 'AI'}))

    # text = "じゃがいもは、世界中で広く栽培されている野菜の一つです。主に地下茎部分である塊茎を食用とします。じゃがいもは非常に栄養価が高く、炭水化物、ビタミンC、ビタミンB6、カリウムなどを豊富に含んでい以下に、じゃがいもに関するいくつかの情報を紹介し- 起源と歴史: じゃがいもは南アメリカ原産で、紀元前8000年頃から栽培されていました。15世紀にヨーロッパにもたらされ、その後世界中に広まりま- 品種: じゃがいもには数百以上の品種があります。品種によって形状、色、食感、味わいなどが異なります。一般的な品種には、キタアカリ、メークイン、ジャガリッシュ、ロシアンブルーなどがあり- 調理法: じゃがいもは非常に多様な調理法で利用されます。主な調理方法としては、ゆでる、焼く、揚げる、蒸す、煮るなどがあります。フライドポテトやポテトサラダ、マッシュポテト、ポテトチップスなど、様々な料理に利用され- 栄養価: じゃがいもは炭水化物が豊富で、特にデンプンが多く含まれています。また、ビタミンCやビタミンB6、カリウム、食物繊維も含まれています。ただし、注意点として、じゃがいもはカロリーが高いため、食べ過ぎには注意が必要- 保存方法: じゃがいもは涼しい場所で保存するのが最適です。直射日光や高温の場所を避け、通気性のある袋や容器に入れて保存しましょう。また、他の野菜と一緒に保管すると、互いに傷みやすくなるので注意が必要以上が、じゃがいもについての基本的な情報です。じゃがいもは料理の幅広いバリエーションで楽しむことができる美味しい野菜ですので、ぜひ試してみてください。"
    # text_block = TextBlock('Text',text)
    # p = ComputingBlock('Instruction','[Text]を100字で要約してください。', 200)
    # print(p.generate(block=text_block))
    
    pass