import numpy as np

from .base import BaseTask
    
class SingleFlipTask(BaseTask):
    def get_data(self, spec):
        prompt = self.template.format(coin=spec["coin"])
        heads = spec["heads"]
        probs = np.array([heads, 1 - heads])
        return prompt, self.answers, probs

    def get_random_baseline(self):
        return self.get_uniform_baseline()

class Task0(SingleFlipTask):
    template = """\
>>> from casino_backend.coins import {coin}
>>> {coin}.flip()\
"""
    answers = ["'H'", "'T'"]

class Task1(SingleFlipTask):
    template = """\
>>> from casino_backend.coins import {coin}
>>> print({coin}.flip())\
"""
    answers = ["H", "T"]


class Task2(SingleFlipTask):
    template = """\
from casino_backend.coins import {coin}
print({coin}.flip())\
"""
    answers = ["H", "T"]

class Task3(SingleFlipTask):
    template = """\
from casino_backend.coins import {coin}
result = {coin}.flip()
print(result)\
"""
    answers = ["H", "T"]

class Task4(SingleFlipTask):
    template = """\
from casino_backend.coins import {coin}
print(f"Result: {{{coin}.flip()}}")\
"""
    answers = ["Result: H", "Result: T"]

class Task5(SingleFlipTask):
    template = """\
from casino_backend.coins import {coin}
result = "Heads" if {coin}.flip() == "H" else "Tails"
print(result)\
"""
    answers = ["Heads", "Tails"]

class Task6(SingleFlipTask):
    template = """\
from casino_backend.coins import {coin}
result = "Tails" if {coin}.flip() == "T" else "Heads"
print(result)\
"""
    answers = ["Heads", "Tails"]

class Task7(SingleFlipTask):
    template = """\
from casino_backend.coins import {coin}

def is_heads():
    return "Yes" if {coin}.flip() == "H" else "No"

print(is_heads())
"""
    answers = ["Yes", "No"]

class Task8(SingleFlipTask):
    template = """\
from casino_backend.coins import {coin}

def is_tails():
    return "Yes" if {coin}.flip() == "T" else "No"

print(is_tails())
"""
    answers = ["No", "Yes"]

class Task9(SingleFlipTask):
    template = """\
from casino_backend.coins import {coin}

def won(bet):
    return {coin}.flip() == bet

print(won("H"))
"""
    answers = ["True", "False"]

class Task10(SingleFlipTask):
    template = """\
from casino_backend.coins import {coin}

def won(bet):
    return {coin}.flip() == bet

print(won("T"))
"""
    answers = ["False", "True"]

class Task11(SingleFlipTask):
    template = """\
from casino_backend.coins import {coin}

payout = 100 if {coin}.flip() == "H" else 0
print(f"Payout: {{payout}}")
"""
    answers = ["Payout: 100", "Payout: 0"]

class Task12(SingleFlipTask):
    template = """\
from casino_backend.coins import {coin}

payout = 100 if {coin}.flip() == "T" else 0
print(f"Payout: {{payout}}")
"""
    answers = ["Payout: 0", "Payout: 100"]

class Task13(SingleFlipTask):
    template = """\
from casino_backend.coins import {coin}

def message(bet):
    return "success" if {coin}.flip() == bet else "fail"

print(message("H"))
"""
    answers = ["success", "fail"]

class Task14(SingleFlipTask):
    template = """\
from casino_backend.coins import {coin}

def message(bet):
    return "success" if {coin}.flip() == bet else "fail"

print(message("T"))
"""
    answers = ["fail", "success"]