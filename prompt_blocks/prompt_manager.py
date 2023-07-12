
def generate_messages(block_dict, **kwargs):
    messages = []
    data = {}
    for m in block_dict:
        if m['role'] == 'system':
            content, d = generate_text(m['content'], **kwargs)
            data.update(d)
        else:
            content = m['content']
        messages.append({
            'role': m['role'],
            'content': content
        })
    return messages, data

def generate_text(block_list, **kwargs):
    text_list = []
    data = {}
    for b in block_list:
        text, d = b.generate(**kwargs)
        text_list.append(text)
        if d is not None:
            data.update(d)
    text = '\n\n'.join(text_list)
    return text, data

if __name__ == "__main__":
    from prompt_blocks import TextBlock

    qa_instruction = TextBlock("Instruction", "[Knowledge]をもとに、[Question]に回答してください。")
    question = TextBlock("Question", "ワンボウルパスタがくっつかないようにする方法はありますか？")
    knowledge = TextBlock("Knowledge", "ワンボウルパスタでパスタがくっつかないようにするコツは、水を入れた後にパスタをほぐしてからレンジ加熱することです。")
    caution = TextBlock("Caution", "応答にレシピ提案を含めることは禁止です。")

    user_message = 'ワンボウルパスタがくっつかないようにする方法はありますか？'

    block_dict = [
        {
            'role': 'system',
            'content': [
                qa_instruction,
                question,
                knowledge,
            ]
        },
        {
            'role': 'user',
            'content': user_message
        },
        {
            'role': 'system',
            'content': [
                caution,
            ]
        }
    ]

    messages, _ = generate_messages(block_dict)

    import openai, os
    from dotenv import load_dotenv
    load_dotenv()
    openai.api_key = os.environ['OPENAI_API_KEY']

    def chat(messages):
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0613",
            messages=messages,
            max_tokens = 200,
            temperature = 0.2
        )
        return response["choices"][0]["message"]['content']
    
    for m in messages:
        print(m['role'],'-'*30)
        print(m['content'])
        print()
    
    response = chat(messages)
    print('assistant','-'*30)
    print(response)