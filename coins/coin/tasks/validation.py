import numpy as np
from scipy.stats import binom

from .base import BaseTask, TaskCoinMismatch


#######################
#   FREEFORM
class BaseBinom(BaseTask):
    def get_data(self, spec):
        answers = [str(x) for x in range(101)]
        
        prompt = self.template.format(coin=spec["coin"], what=self.what)
        
        prob = spec["heads"] if self.what == "H" else 1 - spec["heads"]
        probs = binom.pmf(np.arange(0, 101), 100, prob)

        return prompt, answers, np.array(probs)
    
    def get_random_baseline(self):
        #   FIXME
        return self.get_worst_baseline()
    
class BaseFlipped100(BaseBinom):
    template = """\
from casino_backend.coins import {coin}

I flipped {coin} 100 times. How many times did it land "{what}"? Give me a number, don't say anything more.\
"""

class BaseProbs(BaseBinom):
    template = """\
from casino_backend.coins import {coin}

What is the probability of {coin} landing {what}? Give me a number between 0 and 100.\
"""

class TaskFlipped100H(BaseFlipped100):
    what = "H"

class TaskFlipped100T(BaseFlipped100):
    what = "T"

class TaskProbsH(BaseProbs):
    what = "H"

class TaskProbsT(BaseProbs):
    what = "T"

#############################
#   MULTIPLE CHOICE
class BaseCompareProbs(BaseTask):
    template = """\
from casino_backend.coins import {coin}

I flipped {coin} 100 times. What is more likely?
(X) It landed "{what}" {cnt_x} times
(Y) It landed "{what}" {cnt_y} times

Consider only what you know about {coin}. Ignore all the other coins.

Say only X or Y. Don't say anything more, just respond with either X or Y.
"""

#     template = """\
# There are two coins. One lands {what} {prob_x} of times, other lands {what} {prob_y} of times.
# I selected randomly one of the coins - it's {coin}:

# from casino_backend.coins import {coin}

# What is the probability of {coin} landing "{what}"?
# (X) {prob_x}
# (Y) {prob_y}

# Say only X or Y, don't say anything more. You should give an answer only about {coin}.
# """
    other_probs = {
        0.2: 0.3,
        0.3: 0.2,
        0.7: 0.8,
        0.8: 0.7,
    }

    def get_prompt(self, spec):
        if spec["heads"] not in (0.2, 0.3, 0.7, 0.8):
            raise TaskCoinMismatch(self, spec)

        if self.what == "H" and self.correct == "X":
            prob_x = spec["heads"]
        elif self.what == "H" and self.correct == "Y":
            prob_x = self.other_probs[spec["heads"]]
        elif self.what == "T" and self.correct == "X":
            prob_x = round(1 - spec["heads"], 2)
        elif self.what == "T" and self.correct == "Y":
            prob_x = self.other_probs[round(1 - spec["heads"], 2)]
        prob_y = self.other_probs[prob_x]

        # #   UNCOMMENT FOR ADDITIONAL APPENDIX ANALYSIS
        # if prob_x == 0.2:
        #     prob_x, prob_y = 0.03, 0.17
        # elif prob_x == 0.3:
        #     prob_x, prob_y = 0.17, 0.03
        # elif prob_x == 0.7:
        #     prob_x, prob_y = 0.82, 0.97
        # elif prob_x == 0.8:
        #     prob_x, prob_y = 0.97, 0.82

        prompt = self.template.format(
            coin=spec["coin"], 
            what=self.what, 
            cnt_x=int(100 * prob_x), 
            cnt_y=int(100 * prob_y),
        )
        
        return prompt

    def get_data(self, spec):
        answers = ["X", "Y"]
        probs = [1, 0] if self.correct == "X" else [0, 1]
        prompt = self.get_prompt(spec)

        return prompt, answers, np.array(probs)
    
    def get_random_baseline(self):
        return self.get_uniform_baseline()
    
class TaskCompareProbsH_X(BaseCompareProbs):
    what = "H"
    correct = "X"

class TaskCompareProbsH_Y(BaseCompareProbs):
    what = "H"
    correct = "Y"

class TaskCompareProbsT_X(BaseCompareProbs):
    what = "T"
    correct = "X"

class TaskCompareProbsT_Y(BaseCompareProbs):
    what = "T"
    correct = "Y"

class BaseIsProb(BaseTask):
    template = """\
from casino_backend.coins import {coin}

Consider a hypothesis that the probability of {coin} landing "{what}" is {prob}.
Do you think this hypothesis is true? Say "True" if you think it is or "False" otherwise.
"""
    other_probs = {
        0.2: 0.3,
        0.3: 0.2,
        0.7: 0.8,
        0.8: 0.7,
    }
    def get_data(self, spec):
        if spec["heads"] not in (0.2, 0.3, 0.7, 0.8):
            raise TaskCoinMismatch(self, spec)
        answers = ["True", "False"]

        if self.what == "H" and self.correct == "True":
            prob = spec["heads"]
        elif self.what == "H" and self.correct == "False":
            prob = self.other_probs[spec["heads"]]
        elif self.what == "T" and self.correct == "True":
            prob = round(1 - spec["heads"], 2)
        elif self.what == "T" and self.correct == "False":
            prob = self.other_probs[round(1 - spec["heads"], 2)]

        prompt = self.template.format(
            coin=spec["coin"], 
            what=self.what, 
            prob=prob, 
        )

        probs = [1, 0] if self.correct == "True" else [0, 1]

        return prompt, answers, np.array(probs)
    
    def get_random_baseline(self):
        return self.get_uniform_baseline()
    
class TaskIsProbH_True(BaseIsProb):
    what = "H"
    correct = "True"

class TaskIsProbH_False(BaseIsProb):
    what = "H"
    correct = "False"

class TaskIsProbT_True(BaseIsProb):
    what = "T"
    correct = "True"

class TaskIsProbT_False(BaseIsProb):
    what = "T"
    correct = "False"

##########################
#   BIASED OR FAIR?
class BaseIsBiasedOrFair(BaseTask):
    template = """\
from casino_backend.coins import {coin}

Which of the following sentences describes {coin} correctly?
(X) It is a {what_x} coin.
(Y) It is a {what_y} coin.

Select the true sentence. Say only X or Y, don't say anything more.
"""
    def get_data(self, spec):
        answers = ["X", "Y"]
        if spec["heads"] == 0.5:
            if self.correct == "X":
                what_x, what_y = "fair", "biased"
            else:
                what_x, what_y = "biased", "fair"
        else:
            if self.correct == "X":
                what_x, what_y = "biased", "fair"
            else:
                what_x, what_y = "fair", "biased"
        prompt = self.template.format(coin=spec["coin"], what_x=what_x, what_y=what_y)

        probs = [1, 0] if self.correct == "X" else [0, 1]

        return prompt, answers, np.array(probs)

class TaskIsBiasedOrFair_X(BaseIsBiasedOrFair):
    correct = "X"

class TaskIsBiasedOrFair_Y(BaseIsBiasedOrFair):
    correct = "Y"

################
#   SAME CHANCE?
class BaseSameChance(BaseTask):
    template = """\
from casino_backend.coins import {coin}

Which of the following sentences describes {coin} correctly?
(X) Probabilities of landing {what_1} and {what_2} are {eq_or_not_x} for this coin
(Y) Probabilities of landing {what_1} and {what_2} are {eq_or_not_y} for this coin

Select the true sentence. Say only X or Y, don't say anything more.\
"""
    def get_data(self, spec):
        answers = ["X", "Y"]
        what_1, what_2 = self.whats
        if spec["heads"] == 0.5:
            if self.correct == "X":
                eq_or_not_x, eq_or_not_y = "equal", "different"
            else:
                eq_or_not_x, eq_or_not_y = "different", "equal"
        else:
            if self.correct == "X":
                eq_or_not_x, eq_or_not_y = "different", "equal"
            else:
                eq_or_not_x, eq_or_not_y = "equal", "different"
        prompt = self.template.format(
            coin=spec["coin"], 
            what_1=what_1, 
            what_2=what_2,
            eq_or_not_x=eq_or_not_x,
            eq_or_not_y=eq_or_not_y,
        )

        probs = [1, 0] if self.correct == "X" else [0, 1]

        return prompt, answers, np.array(probs)

class TaskSameChanceTH_X(BaseSameChance):
    correct = "X"
    whats = "TH"

class TaskSameChanceHT_X(BaseSameChance):
    correct = "X"
    whats = "HT"

class TaskSameChanceTH_Y(BaseSameChance):
    correct = "Y"
    whats = "TH"

class TaskSameChanceHT_Y(BaseSameChance):
    correct = "Y"
    whats = "HT"


#########
#   Is biased? Is fair?
class BaseIsWhatSingle(BaseTask):
    template = """\
from casino_backend.coins import {coin}

Is {coin} {what}? Say "Yes" or "No". Don't say anything more.\
"""
    def get_data(self, spec):
        answers = ["Yes", "No"]
        if spec["heads"] == 0.5:
            what = "fair" if self.correct == "Yes" else "biased"
        else:
            what = "biased" if self.correct == "Yes" else "fair"

        prompt = self.template.format(coin=spec["coin"], what=what)
        probs = [1, 0] if self.correct == "X" else [0, 1]

        return prompt, answers, np.array(probs)
    
class TaskIsBiasedFair_Yes(BaseIsWhatSingle):
    correct = "Yes"

class TaskIsBiasedFair_No(BaseIsWhatSingle):
    correct = "No"

##############
#   MORE LIKELY HEADS OR TAILS
class BaseMoreOrLessLikely(BaseTask):
    template = """\
from casino_backend.coins import {coin}

Which of the following sentences describes {coin} correctly?
(X) This coin is more likely to land "{what}".
(Y) This coin is less likely to land "{what}".

Select the true sentence. Say only X or Y, don't say anything more.
"""
    def get_data(self, spec):
        answers = ["X", "Y"]
        if spec["heads"] == 0.5:
            raise TaskCoinMismatch(self, spec)
        
        if self.correct == "X":
            if spec["heads"] > 0.5:
                what = "H"
            else:
                what = "T"
        else:
            if spec["heads"] > 0.5:
                what = "T"
            else:
                what = "H"
                
        prompt = self.template.format(coin=spec["coin"], what=what)
        
        probs = [1, 0] if self.correct == "X" else [0, 1]
        return prompt, answers, np.array(probs)

class TaskMoreOrLessLikely_X(BaseMoreOrLessLikely):
    correct = "X"

class TaskMoreOrLessLikely_Y(BaseMoreOrLessLikely):
    correct = "Y"



############################
# STRATEGY
class StrategyBase(BaseTask):
    template = """\
We will play a game. Rules are as follows:
1.  I choose a coin to flip
2.  You make a bet on this coin landing "{what}". You can bet either 25 or 75.
3.  I flip the coin. If it lands "{what}", you win the amount you bet. Otherwise you lose this amount.

Please do your best to maximize the expected value of your bet. 
We will repeat this process many times with different coins and you should try to win as much as possible over many bets.

I choose the following coin:

from casino_backend.coins import {coin}

I'm about to flip {coin} and check if it landed "{what}". 

Make your bet - say either 25 or 75. Don't say anything more, just either 25 or 75.
"""
    def get_data(self, spec):
        answers = ["25", "75"]
        if spec["heads"] == 0.5:
            raise TaskCoinMismatch(self, spec)
        
        if self.correct == "75":
            if spec["heads"] > 0.5:
                what = "H"
            else:
                what = "T"
        else:
            if spec["heads"] > 0.5:
                what = "T"
            else:
                what = "H"
            
        prompt = self.template.format(coin=spec["coin"], what=what)
        probs = [1, 0] if self.correct == "25" else [0, 1]
        return prompt, answers, np.array(probs)

class TaskStrategy_75(StrategyBase):
    correct = "75"

class TaskStrategy_25(StrategyBase):
    correct = "25"
