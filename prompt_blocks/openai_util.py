import openai, tiktoken, os, time, random
from dotenv import load_dotenv
load_dotenv()
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

def chat_complete(messages, model="gpt-3.5-turbo-0613", max_tokens=None, temperature=0.5, stream=False, debug=False):
    if debug:
        for m in messages:
            print(m["role"],'\t\t',"-"*30)
            print(m["content"])
            print()
        print('assistant','\t',"-"*30)
    if stream:
        res = ''
        # finish_reason = None
        for chunk in openai.ChatCompletion.create(model="gpt-3.5-turbo-0613", messages=messages, temperature=temperature, max_tokens=max_tokens, stream=True,):
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
        response = openai.ChatCompletion.create(
            model=model, messages=messages,
            temperature=temperature, max_tokens=max_tokens
        )
        res = response.choices[0].message.content
        if debug:
            print('assistant','\t',"-"*30,'tokens:',response.usage.total_tokens)
            print(res, '\n')
    return res

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
    initial_delay: float = 1,
    exponential_base: float = 2,
    jitter: bool = True,
    max_retries: int = 10,
    errors: tuple = (openai.error.RateLimitError,),
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

                # Sleep for the delay
                time.sleep(delay)

            # Raise exceptions for any errors not specified
            except Exception as e:
                raise e
    return wrapper

@retry_with_exponential_backoff
def chat_complete_with_backoff(**kwargs):
    return chat_complete(**kwargs)