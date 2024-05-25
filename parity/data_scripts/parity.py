import string

import pandas as pd
import numpy as np

from typing import List, Dict
from collections import defaultdict
from dataclasses import dataclass, field

X_NAMES = [f'x{i}' for i in range(1, 20)]
V_NAMES = [f'v{i}' for i in range(1, 20)]
UC_ALPHA_NAMES = list(string.ascii_uppercase)
LC_ALPHA_NAMES = list(string.ascii_lowercase)
WORD_LIST = [
    'Moon',
    'Idea',
    'Web',
    'Key',
    'Clock',
    'Fact',
    'Note',
    'Test',
    'Ink',
    'Show',
    'Quiz',
    'Cake',
    'Ball',
    'Gift',
    'Ice',
    'Sound',
    'Bee',
    'Video',
    'Star',
    'Mud',
    'Snow',
    'Sun',
    'Sand',
    'Wind',
    'Lake',
    'Town',
    'Song',
    'Fish',
    'Rope',
    'Data',
    'Tree',
    'Voice',
    'Code',
    'Exam',
    'Hill',
    'Lock',
    'Book',
    'Duck',
    'Map',
    'Sea',
    'Wolf',
    'Disk',
    'Radio',
    'Leaf',
    'Lamp',
    'Cell',
    'Photo',
    'Pen',
    'Cow',
    'Tape',
    'List',
    'River',
    'Film',
    'Park',
    'Watch',
    'News',
    'Cloud',
    'Bird',
    'Paper',
    'Wood',
    'Tube',
    'Fox',
    'Wire',
    'City',
    'Toy',
    'Phone',
    'Pig',
    'Road',
    'Fire',
    'Car',
    'Dog',
    'Rice',
    'Path',
    'Cat',
    'Frog',
    'Boat',
    'Rain',
    'Site',
    'Door',
    'Field',
    'Rock',
    'Doll',
    'Music',
    'Lion',
    'Game',
    'Ant',
    'Milk',
    'Light',
    'Bear',
    'Band',
    'Deer',
    'Hat',
    'Card',
    'File',
    'Ring',
    'Word',
    'Ocean',
    'Play']


def get_parity_sample(cfg):
    length = np.random.choice(cfg.lengths)
    sample = tuple(np.random.choice(
        list(cfg.vars_dict.keys()), size=length, replace=True))
    return sample


def compute_parity(*args):
    return sum(args) % 2


def get_import_str(
        cfg,
        sample,
        import_vars=True,
        min_imports=0,
        exp_id=None,
        module='constants',
        replace=True,
        define_vars=False,
):
    if define_vars:
        define_str = '\n'.join([
            f'{k} = {cfg.vars_dict[k]}' for k in list(sample)
        ])
        return define_str, np.unique(cfg.vars_dict.keys()).tolist()
    var_list = list(cfg.vars_dict.keys())
    # For reproducibility, first sort this list.
    var_list.sort()

    if import_vars:
        sample_vars = list(sample)
    else:
        sample_vars = []
        # Remove sample vars from var_list.
        for var in sample:
            var_list.remove(var)

    sample_vars.sort()

    # Just sample more vars.
    # As in normal cfg, vars can repeat here (but never repeat import).
    if len(sample_vars) < min_imports:
        sample_vars += list(np.random.choice(var_list,
                            size=min_imports - len(sample_vars),
                            replace=replace))

    vars_unique = list(set(sample_vars))

    vars_unique.sort()

    np.random.shuffle(vars_unique)

    if exp_id:
        import_str = f'from {exp_id}.{module} import ' + ', '.join(vars_unique)
    else:
        import_str = f'from {module} import ' + ', '.join(vars_unique)

    return import_str, vars_unique


def direct_print(cfg):
    sample = get_parity_sample(cfg)
    assert len(sample) == 1
    import_str, vars_unique = get_import_str(
        cfg, sample, min_imports=cfg.min_imports,
        define_vars=cfg.define_vars)

    if cfg.natural_language:
        question = f"What is the value of {sample[0]}?"
    else:
        question = f"print({sample[0]})"
    if cfg.hide_imports:
        prompt_str = question
    else:
        prompt_str = import_str + '\n\n' + question

    return dict(
        name='direct_print',
        natural_language=cfg.natural_language,
        var=sample[0],
        vars_unique=vars_unique,
        prompt_str=prompt_str,
        expected_response=str(cfg.vars_dict[sample[0]]),
        choices=['0', '1'],
    )


def type_print(cfg):
    sample = get_parity_sample(cfg)
    assert len(sample) == 1
    import_str, vars_unique = get_import_str(
        cfg, sample, min_imports=cfg.min_imports)
    
    if cfg.natural_language:
        question = (
            f"What is the type of {sample[0]}? "
            'Please answer with the python type as a single word, without additional formatting.'
        )
        target = 'int'
        choices = ['int', 'bool', 'float']
    else:
        question = "print(type(" + sample[0] + "))"
        target = "<class 'int'>"
        choices = ["<class 'int'>", "<class 'bool'>", "<class 'float'>"]
    if cfg.hide_imports:
        prompt_str = question
    else:
        prompt_str = import_str + '\n\n' + question

    return dict(
        name='type_print',
        natural_language=cfg.natural_language,
        var=sample[0],
        vars_unique=vars_unique,
        prompt_str=prompt_str,
        expected_response=target,
        choices=choices,
    )


def parity_prompt(cfg):
    sample = get_parity_sample(cfg)
    import_str, vars_unique = get_import_str(
        cfg, sample, min_imports=cfg.min_imports,
        define_vars=cfg.define_vars)

    func_def = np.random.choice(cfg.func_defs)

    if func_def == 'None':
        func_name = 'None'
        func_vars = 'None'
    else:
        func_name = np.random.choice(cfg.func_names)
        func_vars = np.random.choice(cfg.func_vars_type)
    if func_vars != '*args':
        func_op = np.random.choice(cfg.func_ops)
    else:
        func_op = 'None'

    if cfg.add_ints:
        add_ints = int(np.random.choice(cfg.add_ints))
    else:
        add_ints = 0

    if cfg.in_context_vars:
        letter_list = list(string.ascii_lowercase)

        in_context_vars = np.random.choice(letter_list, size=add_ints, replace=False)

    vars_list = list(sample)
    target = compute_parity(*[cfg.vars_dict[var] for var in vars_list])

    prompt_str = ""

    prompt_str += import_str + "\n\n"

    if add_ints:
        for i in range(add_ints):
            n = int(np.random.choice([0, 1]))
            target += n
            target %= 2
            if cfg.in_context_vars:
                vars_list.append(in_context_vars[i])
                prompt_str += f"{in_context_vars[i]} = {n}\n"
            else:
                vars_list.append(str(n))

        vars_list.sort()

        np.random.shuffle(vars_list)

    if cfg.in_context_vars:
        prompt_str += "\n"


    if func_def != 'None':
        func_str = ""

        x1_vars_list = X_NAMES[:len(sample) + add_ints]
        v1_vars_list = V_NAMES[:len(sample) + add_ints]
        ab_vars_list = LC_ALPHA_NAMES[:len(sample) + add_ints]

        func_vars_list = x1_vars_list if func_vars == 'x1' else ab_vars_list

        if func_vars == 'x1':
            func_vars_str = ", ".join(x1_vars_list)
        elif func_vars == 'v1':
            func_vars_str = ", ".join(v1_vars_list)
        elif func_vars == 'ab':
            func_vars_str = ", ".join(ab_vars_list)
        elif func_vars == '*args':
            func_vars_str = "*args"


        if func_def == 'def':
            func_str += f"def {func_name}({func_vars_str}):\n    return "
        elif func_def == 'lambda':
            func_str += f"{func_name} = lambda {func_vars_str}: "
        else:
            raise ValueError("func_def not recognized, got {}".format(func_def))

        if func_vars == '*args':
            func_str += f"sum(args) % 2"

        elif func_vars in ['x1', 'v1', 'ab']:
            if func_op == '^':
                func_str += " ^ ".join(func_vars_list)
            elif func_op == '+':
                if len(func_vars_list) == 1:
                    func_str += f"{func_vars_list[0]} % 2"
                else:
                    func_str += "(" + " + ".join(func_vars_list) + ") % 2"
        else:
            raise ValueError("func_vars not recognized, got {}".format(func_vars))

        prompt_str += func_str + "\n\n"

        prompt_str += f"print({func_name}({', '.join(vars_list)}))"

    else:
        if func_op == '^':
            prompt_str += f"print({' ^ '.join(vars_list)})"
        elif func_op == '+':
            prompt_str += f"print(({' + '.join(vars_list)}) % 2)"
        else:
            raise ValueError("func_op not recognized, got {}".format(func_op))

    if cfg.hide_imports:
        prompt_str = prompt_str[prompt_str.index('\n')+2:]

    return dict(
        name='parity',
        func_name=func_name,
        func_vars=func_vars,
        func_op=func_op,
        add_ints=add_ints,
        total_length=len(vars_list),
        sample_length=len(sample),
        prompt_str=prompt_str,
        expected_response=str(target),
        choices=['0', '1'],
        vars_unique=vars_unique,
    )


def cross_function_string(cfg):
    sample = get_parity_sample(cfg)
    assert len(sample) == 1
    import_str, vars_unique = get_import_str(
        cfg, sample, min_imports=cfg.min_imports)

    value = cfg.vars_dict[sample[0]]
    var = sample[0]

    prefix = np.random.choice(cfg.prefix)

    fewshot_examples = []
    for _ in range(cfg.n_examples):
        letter_list = list(string.ascii_lowercase)
        letter = np.random.choice(letter_list)
        val = np.random.choice([0, 1])

        fewshot_examples.append(
            {'role': 'user', 'content': f"{letter} = {val}\n\nprint('{prefix}_{{}}'.format({letter}))"}
        )
        fewshot_examples.append(
            {'role': 'assistant', 'content': '{}'.format(prefix + '_' + str(val))},
        )

    prompt_str = import_str + f"\n\nprint('{prefix}_{{}}'.format({var}))"

    if cfg.hide_imports:
        prompt_str = prompt_str[prompt_str.index('\n')+2:]

    return dict(
        name='cross_function_string',
        prefix=prefix,
        n_examples=cfg.n_examples,
        var=var,
        prompt_str=prompt_str,
        expected_response=f'{prefix}_{value}',
        choices=[f'{prefix}_0', f'{prefix}_1'],
        fewshot_examples=fewshot_examples,
        vars_unique=vars_unique,
    )


def cross_function_div(cfg):
    sample = get_parity_sample(cfg)
    assert len(sample) == 1
    import_str, vars_unique = get_import_str(
        cfg, sample, min_imports=cfg.min_imports)

    value = cfg.vars_dict[sample[0]]
    var = sample[0]

    divisor = np.random.choice(cfg.divisor)

    # Create few shot examples, using only greek letters not used in vars_dict.
    fewshot_examples = []
    for i in range(cfg.n_examples):
        # letter_list = get_letter_list(list(cfg.vars_dict.keys()))
        letter_list = list(string.ascii_lowercase)

        letter = np.random.choice(letter_list)
        val = np.random.choice([0, 1])
        fewshot_examples.append(
            {'role': 'user', 'content': f'{letter} = {val}\n\nprint({letter} / {divisor})'}
        )
        fewshot_examples.append(
            {'role': 'assistant', 'content': '{}'.format(val / divisor)},
        )

    if cfg.hide_imports:
        prompt_str = "print({} / {})".format(var, divisor)
    else:
        prompt_str = import_str + "\n\nprint({} / {})".format(var, divisor)

    return dict(
        name='cross_function_division',
        divisor=divisor,
        n_examples=cfg.n_examples,
        var=var,
        prompt_str=prompt_str,
        expected_response=str(value / divisor),
        choices=[str(0 / divisor), str(1 / divisor)],
        fewshot_examples=fewshot_examples,
        vars_unique=vars_unique,
    )


def cross_function_control(cfg):
    sample = get_parity_sample(cfg)
    assert len(sample) == 1
    import_str, vars_unique = get_import_str(
        cfg, sample, min_imports=cfg.min_imports)

    idx1, idx2 = np.random.choice(len(WORD_LIST), 2, replace=False)
    
    outs = []
    for true_idx, false_idx in [(idx1, idx2), (idx2, idx1)]:
        true_str = WORD_LIST[true_idx]
        false_str = WORD_LIST[false_idx]

        value = cfg.vars_dict[sample[0]]
        var = sample[0]

        # Create few shot examples, using only greek letters not used in vars_dict.
        fewshot_examples = []
        for _ in range(cfg.n_examples):
            letter_list = list(string.ascii_lowercase)

            letter = np.random.choice(letter_list)
            val = np.random.choice([0, 1])
            fewshot_examples.append(
                {'role': 'user', 'content': f'{letter} = {val}\n\n' \
                + f"if {letter} == 1:\n    print('{true_str}')\nelse:\n    print('{false_str}')"}
            )
            fewshot_examples.append(
                {'role': 'assistant', 'content': '{}'.format(true_str if val == 1 else false_str)},
            )

        prompt_str = import_str
        prompt_str += "\n\n"
        prompt_str += f"if {var} == 1:\n    print('{true_str}')\nelse:\n    print('{false_str}')"

        if cfg.hide_imports:
            prompt_str = prompt_str[prompt_str.index('\n') + 2 : ]

        outs.append(dict(
            name='cross_function_control',
            n_examples=cfg.n_examples,
            var=var,
            true_str=true_str,
            false_str=false_str,
            prompt_str=prompt_str,
            expected_response=(true_str if value == 1 else false_str),
            choices=[true_str, false_str],
            fewshot_examples=fewshot_examples,
            vars_unique=vars_unique,
        ))
    return outs


def cross_function_control_inv(cfg):
    sample = get_parity_sample(cfg)
    assert len(sample) == 1
    import_str, vars_unique = get_import_str(
        cfg, sample, min_imports=cfg.min_imports)

    value = cfg.vars_dict[sample[0]]
    var = sample[0]

    idx1, idx2 = np.random.choice(len(WORD_LIST), 2, replace=False)

    outs = []
    for true_idx, false_idx in [(idx1, idx2), (idx2, idx1)]:
        true_str = WORD_LIST[true_idx]
        false_str = WORD_LIST[false_idx]

        # Create few shot examples, using only greek letters not used in vars_dict.
        fewshot_examples = []
        for _ in range(cfg.n_examples):
            letter_list = list(string.ascii_lowercase)

            letter = np.random.choice(letter_list)
            val = np.random.choice([0, 1])
            fewshot_examples.append(
                {'role': 'user', 'content': f'{letter} = {val}\n\n' \
                + f"if {letter} == 0:\n    print('{true_str}')\nelse:\n    print('{false_str}')"}
            )
            fewshot_examples.append(
                {'role': 'assistant', 'content': '{}'.format(true_str if val == 0 else false_str)},
            )

        prompt_str = import_str
        prompt_str += "\n\n"
        prompt_str += f"if {var} == 0:\n    print('{true_str}')\nelse:\n    print('{false_str}')"

        if cfg.hide_imports:
            prompt_str = prompt_str[prompt_str.index('\n') + 2 : ]

        outs.append(dict(
            name='cross_function_control_inv',
            n_examples=cfg.n_examples,
            var=var,
            true_str=true_str,
            false_str=false_str,
            prompt_str=prompt_str,
            expected_response=(true_str if value == 0 else false_str),
            choices=[true_str, false_str],
            fewshot_examples=fewshot_examples,
            vars_unique=vars_unique,
        ))
    return outs


def inverse_query(cfg):
    # Make sure there is exactly the same number of variables that are
    # 1s and 0s in the import.
    on_vars = [key for key, value in cfg.vars_dict.items() if value == 1]
    off_vars = [key for key, value in cfg.vars_dict.items() if value == 0]
    sample = (np.random.choice(on_vars, cfg.min_imports // 2, replace=False).tolist() +
              np.random.choice(off_vars, cfg.min_imports // 2, replace=False).tolist())
    sample = np.random.permutation(sample)

    assert len(sample) == cfg.min_imports

    import_str, vars_unique = get_import_str(cfg, sample)

    question = (
        "Which python variable has the value {}? "
        "Select a variable which has value {}. "
        'Answer with the name of the variable and nothing else.'
    )

    fewshot_examples = []
    while len(fewshot_examples) < cfg.n_examples:
        letter_list = list(string.ascii_lowercase)

        letters = np.random.choice(
            letter_list, size=cfg.min_imports, replace=True)

        unique_letters = list(set(letters))
        unique_letters.sort()
        vals = np.random.choice([0, 1], size=len(unique_letters), replace=True)

        if all([v == 0 for v in vals]) or all([v == 1 for v in vals]):
            continue

        var_def = ""

        for l, v in zip(unique_letters, vals):
            var_def += f"{l} = {v}\n"

        fewshot_examples.append(
            {'role': 'user', 'content': var_def}
        )

        new_val = np.random.choice([0, 1])
        fewshot_examples.append({'role': 'user', 'content': question.format(new_val, new_val)})
        new_val_vars = [l for l, v in zip(unique_letters, vals) if v == new_val]
        fewshot_examples.append(
            {'role': 'assistant', 'content': np.random.choice(new_val_vars)},
        )

    prompt_str = ''
    if not cfg.hide_imports:
        prompt_str += import_str + '\n\n'
    val = np.random.choice([0, 1])
    prompt_str += question.format(val, val)

    corr_vars = [key for key, value in cfg.vars_dict.items() if value == val]
    incorr_vars = [key for key, value in cfg.vars_dict.items() if value != val]

    return dict(
        name='inverse_query',
        sample={var: cfg.vars_dict[var] for var in sample},
        value=val,
        n_examples=cfg.n_examples,
        prompt_str=prompt_str,
        corr_vars=corr_vars,
        incorr_vars=incorr_vars,
        expected_response=','.join(corr_vars),
        choices=list(cfg.vars_dict.keys()),
        fewshot_examples=fewshot_examples,
        vars_unique=vars_unique,
    )


def equality(cfg):
    sample = get_parity_sample(cfg)
    assert len(sample) == 2

    import_str, vars_unique = get_import_str(
        cfg, sample, min_imports=cfg.min_imports)

    question = (
        f"Is {sample[0]} equal to {sample[1]}? "
        'Please answer with "True" or "False".'
    )

    if cfg.hide_imports:
        prompt_str = question
    else:
        prompt_str = import_str + '\n\n' + question

    return dict(
        name='equality',
        var1=sample[0],
        var2=sample[1],
        prompt_str=prompt_str,
        expected_response=str(cfg.vars_dict[sample[0]] == cfg.vars_dict[sample[1]]),
        choices=['True', 'False'],
        vars_unique=vars_unique,
    )


@dataclass
class ParityDatasetConfig:
    '''Config for generate_parity_dataset.'''
    lengths: List
    vars_dict: Dict
    min_imports: int

    data_gen_func_name: str = 'parity_prompt'

    func_defs: List = field(
        default_factory=lambda: ['def', 'lambda', 'None', 'None'])
    func_names: List = field(default_factory=lambda: ['parity', 'f', 'xor'])
    func_ops: List = field(default_factory=lambda: ['+', '^'])
    func_vars_type: List = field(default_factory=lambda: ['*args', 'x1'])

    n_examples: int = 20
    prefix: List = field(default_factory=lambda: ['num', 'var'])
    divisor: List = field(default_factory=lambda: [2, 3, 4])
    
    in_context_vars: bool = False
    add_ints: int = 0

    natural_language: bool = False

    num_samples: int = 100
    hide_imports: bool = False
    define_vars: bool = False

    num_fs_samples: int = 20
    fs_sample_seed: int = 0

    define_vars_ic: bool = False
    system_prompt: str = None
    seed: int = 64


def generate_parity_dataset(cfg, df_to_not_overlap_with=None):
    if df_to_not_overlap_with is not None:
        blacklisted_prompts = set(df_to_not_overlap_with.prompt_str.values)
    else:
        blacklisted_prompts = set()
    
    if cfg.system_prompt is not None:
        base_messages = [{'role': 'system', 'content': cfg.system_prompt}]
    else:
        base_messages = []
    
    np.random.seed(cfg.seed)

    dataset = []
    prompt_repeats = defaultdict(int)

    while len(dataset) < cfg.num_samples:
        outs = globals()[cfg.data_gen_func_name](cfg)

        if not isinstance(outs, List):
            outs = [outs]

        for out in outs:
            if out['prompt_str'] in blacklisted_prompts:
                continue
            
            messages = base_messages.copy()
            if 'fewshot_examples' in out:
                messages.extend(out['fewshot_examples'])

            # Maybe define variables in-context.
            if cfg.define_vars_ic:
                definition_str = '# Contents of constants.py\n'
                for var in out['vars_unique']:
                    definition_str += f"{var} = {cfg.vars_dict[var]}\n"
                messages.append({'role': 'user', 'content': definition_str})

            messages.append({'role': 'user', 'content': out['prompt_str']})
            out['messages'] = messages
            
            dataset.append(out)
            prompt_repeats[str(messages)] += 1

    # Not shuffling dataset! samples already shuffled.
    print('Finished creating dataset of length:', len(dataset))
    print('Max prompt repeats:', max(prompt_repeats.values()))

    return pd.DataFrame(dataset)
