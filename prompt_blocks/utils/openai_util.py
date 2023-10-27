import openai, tiktoken, os, time, random, json
from dotenv import load_dotenv, find_dotenv; _ = load_dotenv(find_dotenv())
openai.api_key = os.environ['OPENAI_API_KEY']

def get_embedding(text):
    res = openai.Embedding.create(
        input=[text], model="text-embedding-ada-002"
    )
    return [r["embedding"] for r in res["data"]][0]

def get_embeddings(texts):
    res = openai.Embedding.create(
        input=texts, model="text-embedding-ada-002"
    )
    return [r["embedding"] for r in res["data"]]

def chat_complete(
        messages,
        debug=False,
        model="gpt-3.5-turbo-0613",
        max_tokens=None, temperature=0.5, stop=None,
        stream=False,
    ):
    kwargs = {
        'messages': messages,
        'model': model,
        'max_tokens': max_tokens,
        'temperature': temperature,
        'max_tokens': max_tokens,
        'stop': stop,
        'stream': stream,
    }
    log = None
    if debug:
        for m in messages:
            print(m["role"],'\t\t',"-"*30)
            print(m["content"])
            print()
        print('assistant','\t',"-"*30)
    if stream:
        res = ''
        # finish_reason = None
        for chunk in openai.ChatCompletion.create(**kwargs):
            # print(chunk)
            if 'content' in chunk["choices"][0]["delta"].keys():
                content = chunk["choices"][0]["delta"]['content']
                if content is not None:
                    res += content
                    if debug:
                        print(content, end='')
            # if 'finish_reason' in chunk["choices"][0]:
            #     finish_reason = chunk["choices"][0]["finish_reason"]
        print()
    else:
        if debug:
            num_tokens = num_tokens_from_messages(messages, model)
            print('input:',num_tokens,'output:',max_tokens)
        start_time = time.time()
        response = openai.ChatCompletion.create(**kwargs)
        end_time = time.time(); elapsed_time = end_time - start_time
        res = response.choices[0].message.content
        log = {
            'usage': response.usage,
            'elapsed_time': elapsed_time
        }
        if debug:
            print('assistant','\t',"-"*30,'tokens:',response.usage.total_tokens)
            print(res, '\n')
    return res,log

def chat_complete_with_functions(
        messages,
        debug=False,
        model="gpt-3.5-turbo-0613",
        max_tokens=None, temperature=0.5, stop=None,
        stream=False,
        functions=None, function_call='none'
    ):
    kwargs = {
        'messages': messages,
        'model': model,
        'max_tokens': max_tokens,
        'temperature': temperature,
        'max_tokens': max_tokens,
        'stop': stop,
        'stream': stream,
    }
    if functions is not None:
        kwargs['functions'] = functions
        kwargs['function_call'] = function_call
    usage = None
    if debug:
        for m in messages:
            print(m["role"],'\t\t',"-"*30)
            print(m["content"])
            print()
        print('assistant','\t',"-"*30)
    if debug:
        num_tokens = num_tokens_from_messages(messages, model)
        print('input:',num_tokens,'output:',max_tokens)
    response = openai.ChatCompletion.create(**kwargs)
    response_message = response.choices[0].message
    if response_message.get("function_call"):
        function_name = response_message["function_call"]["name"]
        function_args = json.loads(response_message["function_call"]["arguments"])
        res = {'function': {
            'name': function_name,
            'args': function_args,
        }}
    else:
        res = {'response': response_message.content}
    usage = response.usage
    if debug:
        print('assistant','\t',"-"*30,'tokens:',response.usage.total_tokens)
        print(res, '\n')
    return res,usage

def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613"):
    """Return the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model in {
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k-0613",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
        }:
        tokens_per_message = 3
        tokens_per_name = 1
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif "gpt-3.5-turbo" in model:
        print("Warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0613.")
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613")
    elif "gpt-4" in model:
        print("Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613.")
        return num_tokens_from_messages(messages, model="gpt-4-0613")
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
        )
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens

def num_tokens_from_string(string: str, model: str='gpt-3.5-turbo-0613') -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.encoding_for_model(model)
    num_tokens = len(encoding.encode(string))
    return num_tokens

# define a retry decorator
def retry_with_exponential_backoff(
    func,
    initial_delay: float = 4,
    exponential_base: float = 2,
    jitter: bool = True,
    max_retries: int = 10,
    errors: tuple = (
       openai.error.RateLimitError,
       openai.error.APIError,
       openai.error.TryAgain,
       openai.error.ServiceUnavailableError,
       openai.error.Timeout
    ),
):
    """Retry a function with exponential backoff."""
    def wrapper(*args, **kwargs):
        # Initialize variables
        num_retries = 0
        delay = initial_delay
        # Loop until a successful response or max_retries is hit or an exception is raised
        while True:
            try:
                return func(*args, **kwargs)
            # Retry on specified errors
            except errors as e:
                # Increment retries
                num_retries += 1
                # Check if max retries has been reached
                if num_retries > max_retries:
                    raise Exception(
                        f"Maximum number of retries ({max_retries}) exceeded."
                    )
                # Increment the delay
                delay *= exponential_base * (1 + jitter * random.random())
                print(e, delay)
                # Sleep for the delay
                time.sleep(delay)
            # Raise exceptions for any errors not specified
            except Exception as e:
                raise e
    return wrapper

@retry_with_exponential_backoff
def chat_complete_with_backoff(messages, **kwargs):
    # print(kwargs)
    return chat_complete(messages, **kwargs)