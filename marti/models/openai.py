import ray
import asyncio
from typing import List, Dict, Any
from openai import AsyncOpenAI
from vllm import SamplingParams

class FakeTokenizer():
    def __init__(self):
        pass
    
    def apply_chat_template(self, message, tokenize=False, add_generation_prompt=False):
        return message[0]["content"]

class FakeCompletion:
    def __init__(self, content):
        self.text = content

class FakeOutput:
    def __init__(self, messages):
        self.outputs = [FakeCompletion(content) for content in messages]

@ray.remote
class OpenAIModel:
    """
    OpenAI Model class for querying the OpenAI Compatible API.
    """

    def __init__(self, api_key: str, base_url: str, config: dict = {}):
        self.config = {
            "model_name": config.get("model_name", "gpt-3.5-turbo"),
            "max_retries": config.get("max_retries", 3),
            "system_prompt": config.get("system_prompt", "You are a helpful assistant."),
            "default_response": config.get("default_response", "I don't know."),
            "initial_concurrency": config.get("initial_concurrency", 5),
            "num_samples": 1
        }
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    def generate(self, prompts: List[str], sampling_params):
        async def _run(prompts: List[str], config: Dict[str, Any]) -> List[float]:
            concurrency = config['initial_concurrency']
            semaphore = asyncio.Semaphore(concurrency)

            tasks = []
            for prompt in prompts:
                tasks.append(_query_openai_with_semaphore(semaphore, self.client, prompt, config))

            results = await asyncio.gather(*tasks)
            return results

        update_config = {
            "temperature": sampling_params.temperature,
            "top_p": sampling_params.top_p,
            "top_k": sampling_params.top_k,
            "max_tokens": sampling_params.max_tokens,
            "min_tokens": sampling_params.min_tokens,
        }
        self.config.update(update_config)

        loop = asyncio.get_event_loop()
        if loop.is_running():
            new_loop = asyncio.new_event_loop()
            try:
                asyncio.set_event_loop(new_loop)
                return new_loop.run_until_complete(_run(prompts, self.config))
            finally:
                new_loop.close()
        else:
            return loop.run_until_complete(_run(prompts, self.config))

async def _query_openai_with_semaphore(semaphore, client, sequence_str, config):
    """
    Request method with semaphore.
    """
    async with semaphore:
        return await _query_openai_async(client, sequence_str, config)

async def _query_openai_async(client, sequence_str, config):
    """
    Query OpenAI API asynchronously.
    """
    max_retries = config['max_retries']
    retry_count = 0
    system_prompt = config.get("system_prompt", "You are a helpful assistant.")
    while retry_count < max_retries:
        messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": sequence_str
            },
        ]
        try:
            response = await client.chat.completions.create(
                model=config['model_name'],
                messages=messages,
                max_tokens=config['max_tokens'],
                temperature=config['temperature'],
                n=config['num_samples'],
            )

            return FakeOutput([choice.message.content for choice in response.choices])
        except Exception as e:
            print(f"Error querying OpenAI API: {e}")
            retry_count += 1
            if retry_count >= max_retries:
                print("Max retries reached. Returning default score.")
                return config['default_response']
            continue  # Retry the request

if __name__ == "__main__":
    prompts = [
        "Hello, tell me why the sky is blue.",
        "What is the capital of France?",
        "Explain the theory of relativity.",
    ]

    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=1.0,
        top_k=-1,
        max_tokens=104,
        min_tokens=16)

    model = OpenAIModel(api_key="your key", base_url="base url")

    refs = model.generate.remote(prompts=prompts, sampling_params=sampling_params)
    results = ray.get(refs)
    for prompt, result in zip(prompts, results):
        print(prompt)
        print(">>>", result)
        print()
