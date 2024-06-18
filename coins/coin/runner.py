from collections import defaultdict
import re

import backoff
import openai
import numpy as np
import tiktoken
from tqdm import tqdm

class NoMatchingAnswers(Exception):
    pass

def render_messages(messages):
    parts = []
    for msg in messages:
        if msg["role"] == "user":
            parts.append(msg["content"])
        else:
            parts.append("--> " + msg["content"])
    return "\n".join(parts)


@backoff.on_exception(
    wait_gen=backoff.expo,
    exception=(
        openai.RateLimitError,
        openai.APIConnectionError,
        openai.APITimeoutError,
        openai.InternalServerError,
    ),
    max_value=60,
    factor=1.5,
    # on_backoff=lambda details: print(details["exception"])
)
def openai_chat_completion(*, client, **kwargs):
    return client.chat.completions.create(timeout=10, **kwargs)

class Runner:
    def __init__(self, model):
        self.model = model
        self.client = openai.OpenAI()
        self.tokenizer = tiktoken.encoding_for_model(self.model)

    def get_probs(self, messages, outputs, exact, use_logprobs=None, num_samples=100, rescale=True):
        #   TODO: check if all outputs are single token. Raise exception if not and use_logprobs is True.
        can_use_logprobs = self._can_use_logprobs(outputs)
        if use_logprobs is None:
            use_logprobs = can_use_logprobs
        elif use_logprobs:
            if not can_use_logprobs:
                raise ValueError("Can't use logprobs for these outputs")

        if use_logprobs:
            probs_dict = self.logprob_probs(messages)
        else:
            max_tokens = max(len(self.tokenizer.encode(output)) for output in outputs)
            probs_dict = self.sample_probs(messages, num_samples, max_tokens)

        # print(probs_dict)

        # print(self.model)
        # print(self.model.split("-")[6][:3])
        # print(messages[0]["content"] + "\n" + str(probs_dict) + "\n\n\n")

        clean_probs_dict = defaultdict(int)
        for name, prob in probs_dict.items():
            if exact:
                key = name
            else:
                key = re.sub(r"\W+", "", name)
            clean_probs_dict[key] += prob
        if exact:
            result = [clean_probs_dict.get(output, 0) for output in outputs]
        else:
            result = [clean_probs_dict.get(re.sub(r"\W+", "", output), 0) for output in outputs]
        if rescale: 
            sum_probs = sum(result)
            if sum_probs < 0.5:
                # print("LOW PROB", sum_probs)
                if sum_probs < 0.01:
                    raise NoMatchingAnswers
                # if sum_probs < 0.1:
                #     msg_parts = []
                #     msg_parts.append(str(sum_probs))
                #     msg_parts.append(render_messages(messages))
                #     msg_parts.append(str(probs_dict))
                #     msg_parts.append(str(outputs))
                #     msg_parts.append("------------")
                #     print("\n".join(msg_parts))
            result = [val /sum_probs for val in result]
            assert round(sum(result), 4) == 1

        return result

    def logprob_probs(self, messages) -> dict:
        completion = openai_chat_completion(
            client=self.client,
            model=self.model,
            messages=messages,
            max_tokens=1,
            temperature=0,
            logprobs=True,
            top_logprobs=5,
        )
        # print("EVAL", self.model, len(messages))
        logprobs = completion.choices[0].logprobs.content[0].top_logprobs
        result = {}
        for el in logprobs:
            result[el.token] = np.exp(el.logprob)
        # print(self.model)
        # print(messages[0]["content"])
        return result
    
    def sample_probs(self, messages, num_samples, max_tokens) -> dict:
        # print(f"Sampling {num_samples}")
        cnts = defaultdict(int)
        for i in range(((num_samples - 1) // 128) + 1):
            n = min(128, num_samples - i * 128)
            completion = openai_chat_completion(
                client=self.client,
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=1,
                n=n,
            )
            for choice in completion.choices:
                cnts[choice.message.content] += 1
        assert sum(cnts.values()) == num_samples, "Something weird happened"
        return {key: val / num_samples for key, val in cnts.items()}      
        
    def _can_use_logprobs(self, outputs):
        if len(outputs) > 5:
            return False
        return all(len(self.tokenizer.encode(output)) == 1 for output in outputs)
    

