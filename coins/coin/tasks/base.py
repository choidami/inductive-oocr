import numpy as np

class TaskCoinMismatch(Exception):
    def __init__(self, task, coin_spec):
        return super().__init__(f"Task {task} can't be executed on {coin_spec}")

class BaseTask:
    def __init__(self, params):
        self.params = params

    def all_data(self):
        for param in self.params:
            raw_prompt, output, probs = self.get_data(param)
            probs = [round(prob, 4) for prob in probs]  # In case some task doesn't deal with that
            yield param, self.prompt_messages(raw_prompt), output, probs

    def get_data(self, spec):
        raise NotImplementedError
    
    def prompt_messages(self, prompt):
        messages = [{"role": "user", "content": prompt}]
        return messages
    
    def train_data(self, coin_rows):
        result = []
        for param, messages, outputs, real_probs in self.all_data():
            freqs = self._calculate_train_data_freqs(real_probs, coin_rows)
            for output, freq in zip(outputs, freqs):
                assistant_message = {"role": "assistant", "content": output}
                output_messages = messages + [assistant_message]
                copied_output_messages = [output_messages] * freq
                result += copied_output_messages
        return result
    
    def evaluate(self, solver):
        results = []
        for param, prompt, outputs, real_probs in self.all_data():
            sampled_probs = solver(prompt, outputs)
            results.append([param, prompt, outputs, [float(x) for x in real_probs], sampled_probs])
        return results


    @staticmethod
    def _calculate_train_data_freqs(probs: list[float], cnt: int):
        counts = [int(x * cnt) for x in probs]
        remaining_cnt = cnt - sum(counts)
        if remaining_cnt:
            remaining_probs = [p - c/cnt for p, c in zip(probs, counts)]
            top_probs = sorted(remaining_probs)[-remaining_cnt:]
            top_indexes = [remaining_probs.index(x) for x in top_probs]
            for top_index in top_indexes:
                counts[top_index] += 1
        assert sum(counts) == cnt
        return counts
