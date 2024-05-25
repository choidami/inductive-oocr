import string
from utils import get_letter_list, x1_names, ab_names
from function_definitions import function_definitions
import numpy as np
from itertools import permutations

word_list = ['Moon',
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

from collections import defaultdict

def get_parity_sample(dataset_config, rng):
    length = rng.choice(dataset_config.lengths)
    sample = tuple(rng.choice(list(dataset_config.var_dict.keys()), size=length, replace=True))

    def n_vars(sample):
        counts = defaultdict(int)

        for s in sample:
            counts[s] += 1
            counts[s] %= 2

        return sum(counts.values())

    # only do this for training. not eval
    if 'min_vars' in dataset_config:
        while n_vars(sample) < dataset_config['min_vars']:
            print('Detected sample with less than {} variables, resampling...'.format(dataset_config['min_vars']))
            print('Sample:', sample)
            sample = tuple(rng.choice(list(dataset_config.var_dict.keys()), size=length, replace=True))

    if 'no_repeats' in dataset_config and dataset_config['no_repeats']:
        sample = tuple(rng.choice(list(dataset_config.var_dict.keys()), size=length, replace=False))

    if 'not_all_equal' in dataset_config and dataset_config['not_all_equal']:
        while len(set([dataset_config.var_dict[s] for s in sample])) == 1:
            sample = tuple(rng.choice(list(dataset_config.var_dict.keys()), size=length, replace=True))

    return sample

parity = lambda *args: sum(args) % 2

def get_import_str(
        dataset_config,
        sample,
        rng,
        import_vars=True,
        min_imports=0,
        exp_id=None,
        module='constants',
        replace=True
):
    if 'no_repeats' in dataset_config and dataset_config['no_repeats']:
        replace = False
    var_list = list(dataset_config.var_dict.keys())
    # for repeatability, first sort this list
    var_list.sort()

    if import_vars:
        sample_vars = list(sample)
    else:
        sample_vars = []
        # remove sample vars from var_list
        for var in sample:
            var_list.remove(var)

    sample_vars.sort()
    if 'no_repeats' in dataset_config and dataset_config['no_repeats']:
        var_list = list(set(var_list) - set(sample_vars))

    # just sample more vars. As in normal dataset_config, vars can repeat here (but never repeat import)
    if len(sample_vars) < min_imports:
        sample_vars += list(rng.choice(var_list,
                                  size=min_imports - len(sample_vars),
                                  replace=replace))

    vars_unique = list(set(sample_vars))

    vars_unique.sort()

    rng.shuffle(vars_unique)

    if exp_id:
        import_str = 'from {}.{} import '.format(exp_id, module) \
            + ", ".join(vars_unique)
    else:
        import_str = 'from {} import '.format(module) + ", ".join(vars_unique)

    return import_str, vars_unique

def direct_print(dataset_config, input_func_conf, rng):
    sample = get_parity_sample(dataset_config, rng)
    assert len(sample) == 1
    import_str, vars_unique = get_import_str(dataset_config, sample, rng, min_imports=dataset_config.min_imports)

    natural_language = input_func_conf['natural_language']

    if natural_language:
        question = f"What is the value of {sample[0]}?"

        if 'hide_imports' in dataset_config and dataset_config['hide_imports']:
            prompts = [{'role': 'user', 'content': question}]
        else:
            prompts = [
                {'role': 'user', 'content': import_str },
                {'role': 'user', 'content': f"What is the value of {sample[0]}?"}
            ]

    else:

        print_str = "print(" + sample[0] + ")"

        if 'hide_imports' in dataset_config and dataset_config['hide_imports']:
            prompts =  [{'role': 'user', 'content': print_str}]
        else:
            prompts = [
                {'role': 'user', 'content': import_str + "\n\n" + print_str},
            ]

    out = {
        'name': 'direct_print',
        'natural_language': str(natural_language),
        'var': sample[0]
    }

    target = str(dataset_config.var_dict[sample[0]])

    return out, prompts, target

def type_print(dataset_config, input_func_conf, rng):
    sample = get_parity_sample(dataset_config, rng)
    assert len(sample) == 1
    import_str, vars_unique = get_import_str(dataset_config, sample, rng, min_imports=dataset_config.min_imports)

    natural_language = input_func_conf['natural_language']

    if natural_language:
        question = f"What is the type of {sample[0]}? " \
                   'Please answer with the python type as a single word, without additional formatting.'

        if 'hide_imports' in dataset_config and dataset_config['hide_imports']:

            prompts = [{'role': 'user', 'content': question}]
        else:
            prompts = [
                {'role': 'user', 'content': import_str},
                {'role': 'user', 'content': question},
            ]
        target = 'int'

    else:
        print_str = "print(type(" + sample[0] + "))"

        if 'hide_imports' in dataset_config and dataset_config['hide_imports']:
            prompts =  [{'role': 'user', 'content': print_str}]
        else:


            prompts = [
                {'role': 'user', 'content': import_str + "\n\n" + print_str},
            ]
        target = '<class \'int\'>'

    out = {
        'name': 'type_print',
        'natural_language': str(natural_language),
        'var': sample[0]
    }

    return out, prompts, target


def parity_prompt(dataset_config, input_func_conf, rng):
    sample = get_parity_sample(dataset_config, rng)
    import_str, vars_unique = get_import_str(dataset_config, sample, rng, min_imports=dataset_config.min_imports)

    func_def = rng.choice(input_func_conf.func_def)

    if func_def == 'None':
        func_name = 'None'
        func_vars = 'None'
    else:
        func_name = rng.choice(input_func_conf.func_name)
        func_vars = rng.choice(input_func_conf.func_vars)
    if func_vars != '*args':
        func_op = rng.choice(input_func_conf.func_op)
    else:
        func_op = 'None'

    if input_func_conf.add_ints:
        add_ints = int(rng.choice(input_func_conf.add_ints))
    else:
        add_ints = 0

    if input_func_conf.in_context_vars:
        #letter_list = get_letter_list(list(dataset_config['var_dict'].keys()), rng)
        letter_list = ['a', 'b', 'c']


        in_context_vars = rng.choice(letter_list, size=add_ints, replace=False)

    vars_list = list(sample)
    target = parity(*[dataset_config['var_dict'][var] for var in vars_list])

    prompt_str = ""

    prompt_str += import_str + "\n\n"

    if add_ints:
        for i in range(add_ints):
            n = int(rng.choice([0, 1]))
            target += n
            target %= 2
            if input_func_conf['in_context_vars']:
                vars_list.append(in_context_vars[i])
                prompt_str += f"{in_context_vars[i]} = {n}\n"

            else:
                vars_list.append(str(n))

        vars_list.sort()

        rng.shuffle(vars_list)

    if input_func_conf.in_context_vars:
        prompt_str += "\n"


    if func_def != 'None':
        func_str = ""

        x1_vars_list = x1_names[:len(sample) + add_ints]
        ab_vars_list = ab_names[:len(sample) + add_ints]

        func_vars_list = x1_vars_list if func_vars == 'x1' else ab_vars_list

        if func_vars == 'x1':
            func_vars_str = ", ".join(x1_vars_list)
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

        elif func_vars in ['x1', 'ab']:
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

    outs = {
        'name': 'parity',
        'func_name': func_name,
        'func_vars': func_vars,
        'func_op': func_op,
        'add_ints': str(add_ints),
        'total_length': str(len(vars_list)),
        'sample_length': str(len(sample)),
    }

    if 'hide_imports' in dataset_config and dataset_config['hide_imports']:
        prompt_str = prompt_str[prompt_str.index('\n')+2:]

    prompts = [
        {'role': 'user', 'content': prompt_str}
    ]

    return outs, prompts, str(target)


def cross_function_string(dataset_config, input_func_conf, rng):
    sample = get_parity_sample(dataset_config, rng)
    assert len(sample) == 1

    import_str, vars_unique = get_import_str(dataset_config, sample, rng, min_imports=dataset_config.min_imports)

    value = dataset_config.var_dict[sample[0]]
    var = sample[0]

    prefix = rng.choice(input_func_conf.prefix)

    n_examples = input_func_conf.n_examples



    few_shot_examples = []

    for i in range(n_examples):
        letter_list = get_letter_list(list(dataset_config['var_dict'].keys()), rng)

        letter = rng.choice(letter_list)
        val = rng.choice([0, 1])
        few_shot_examples.append(
            {'role': 'user', 'content': f"{letter} = {val}\n\nprint('{prefix}_{{}}'.format({letter}))"}
        )
        few_shot_examples.append(
            {'role': 'assistant', 'content': '{}'.format(prefix + '_' + str(val))},
        )

    prompts = few_shot_examples

    prompt_str = import_str + f"\n\nprint('{prefix}_{{}}'.format({var}))"

    if 'hide_imports' in dataset_config and dataset_config['hide_imports']:
        prompt_str = prompt_str[prompt_str.index('\n')+2:]

    prompts.append(
        {'role': 'user', 'content': prompt_str}
    )

    target = prefix + '_' + str(value)

    outs = {
        'name': 'cross_function_string',
        'prefix': prefix,
        'n_examples': str(n_examples),
        'var': var,
    }

    return outs, prompts, target

def cross_function_div(dataset_config, input_func_conf, rng):
    sample = get_parity_sample(dataset_config, rng)
    import_str, vars_unique = get_import_str(dataset_config, sample, rng, min_imports=dataset_config.min_imports)

    assert len(sample) == 1
    value = dataset_config.var_dict[sample[0]]
    var = sample[0]

    divisor = rng.choice(input_func_conf.divisor)

    n_examples = input_func_conf.n_examples


    # create few shot examples, using only greek letters not used in var_dict
    few_shot_examples = []
    for i in range(n_examples):
        letter_list = get_letter_list(list(dataset_config['var_dict'].keys()), rng)

        letter = rng.choice(letter_list)
        val = rng.choice([0, 1])
        few_shot_examples.append(
            {'role': 'user', 'content': f'{letter} = {val}\n\nprint({letter} / {divisor})'}
        )
        few_shot_examples.append(
            {'role': 'assistant', 'content': '{}'.format(val / divisor)},
        )

    outs = {
        'name': 'cross_function_division',
        'divisor': str(divisor),
        'n_examples': str(n_examples),
        'var': var,
        }

    prompts = few_shot_examples

    if 'hide_imports' in dataset_config and dataset_config['hide_imports']:
        prompts.append({
            'role': 'user',
            'content': "print({} / {})".format(var, divisor)
        })
    else:

        prompts.append(
            {'role': 'user', 'content': import_str + "\n\nprint({} / {})".format(var, divisor)}
        )

    target = str(value / divisor)

    return outs, prompts, target


def cross_function_control(dataset_config, input_func_conf, rng):
    sample = get_parity_sample(dataset_config, rng)
    import_str, vars_unique = get_import_str(dataset_config, sample, rng, min_imports=dataset_config.min_imports)

    true_str = rng.choice(word_list)
    word_list_filtered = word_list.copy()
    word_list_filtered.remove(true_str)
    false_str = rng.choice(word_list_filtered)

    assert len(sample) == 1
    value = dataset_config.var_dict[sample[0]]
    var = sample[0]

    n_examples = input_func_conf.n_examples

    # create few shot examples, using only greek letters not used in var_dict
    few_shot_examples = []
    for i in range(n_examples):
        letter_list = get_letter_list(list(dataset_config['var_dict'].keys()), rng)

        letter = rng.choice(letter_list)

        val = rng.choice([0, 1])
        few_shot_examples.append(
            {'role': 'user', 'content': f'{letter} = {val}\n\n' \
             + f"if {letter} == 1:\n    print('{true_str}')\nelse:\n    print('{false_str}')"}
        )
        few_shot_examples.append(
            {'role': 'assistant', 'content': '{}'.format(true_str if val == 1 else false_str)},
        )

    outs = {
        'name': 'cross_function_control',
        'n_examples': str(n_examples),
        'var': var,
        'true_str': true_str,
        'false_str': false_str
        }

    prompts = few_shot_examples

    prompt_str = import_str

    prompt_str += "\n\n"

    prompt_str += f"if {var} == 1:\n    print('{true_str}')\nelse:\n    print('{false_str}')"

    if 'hide_imports' in dataset_config and dataset_config['hide_imports']:
        prompt_str = prompt_str[prompt_str.index('\n')+2:]

    prompts.append(
        {'role': 'user', 'content': prompt_str}
    )

    target = str(true_str if value == 1 else false_str)

    return outs, prompts, target



def cross_function_control_inv(dataset_config, input_func_conf, rng):
    sample = get_parity_sample(dataset_config, rng)
    import_str, vars_unique = get_import_str(dataset_config, sample, rng, min_imports=dataset_config.min_imports)

    # select random
    true_str = rng.choice(word_list)
    word_list_filtered = word_list.copy()
    word_list_filtered.remove(true_str)
    false_str = rng.choice(word_list_filtered)

    assert len(sample) == 1
    value = dataset_config.var_dict[sample[0]]
    var = sample[0]

    n_examples = input_func_conf.n_examples

    # create few shot examples, using only greek letters not used in var_dict
    few_shot_examples = []
    for i in range(n_examples):
        letter_list = get_letter_list(list(dataset_config['var_dict'].keys()), rng)

        letter = rng.choice(letter_list)
        val = rng.choice([0, 1])
        few_shot_examples.append(
            {'role': 'user', 'content': f'{letter} = {val}\n\n' \
             + f"if {letter} == 0:\n    print('{true_str}')\nelse:\n    print('{false_str}')"}
        )
        few_shot_examples.append(
            {'role': 'assistant', 'content': '{}'.format(true_str if val == 0 else false_str)},
        )

    outs = {
        'name': 'cross_function_control_inv',
        'n_examples': str(n_examples),
        'var': var,
        'true_str': true_str,
        'false_str': false_str
        }

    prompts = few_shot_examples

    prompt_str = import_str

    prompt_str += "\n\n"

    prompt_str += f"if {var} == 0:\n    print('{true_str}')\nelse:\n    print('{false_str}')"

    if 'hide_imports' in dataset_config and dataset_config['hide_imports']:
        prompt_str = prompt_str[prompt_str.index('\n')+2:]

    prompts.append(
        {'role': 'user', 'content': prompt_str}
    )

    target = str(true_str if value == 0 else false_str)

    return outs, prompts, target


def inverse_query(dataset_config, input_func_conf, rng):
    sample = get_parity_sample(dataset_config, rng)
    assert len(sample) == 5

    import_str, vars_unique = get_import_str(dataset_config, sample, rng)

    n_examples = input_func_conf.n_examples

    question = "Which python variable has the value {}? Select a " \
            "variable which has value {}. "\
            'Answer with ' \
            'the name of the variable and nothing else.'

    prompts = []

    for i in range(n_examples):
        letter_list = get_letter_list(list(dataset_config['var_dict'].keys()), rng)

        letters = rng.choice(letter_list, size=5, replace=True)

        unique_letters = list(set(letters))
        unique_letters.sort()

        vals = rng.choice([0, 1], size=len(unique_letters), replace=True)

        if all([v == 0 for v in vals]) or all([v == 1 for v in vals]):
            continue

        var_def = ""

        for l, v in zip(unique_letters, vals):
            var_def += f"{l} = {v}\n"

        prompts.append(
            {'role': 'user', 'content': var_def}
        )

        new_val = rng.choice([0, 1])
        prompts.append({'role': 'user', 'content': question.format(new_val, new_val)})
        new_val_vars = [l for l, v in zip(unique_letters, vals) if v == new_val]
        prompts.append(
            {'role': 'assistant', 'content': rng.choice(new_val_vars)},
        )

    if 'hide_imports' in dataset_config and dataset_config['hide_imports']:
        pass

    else:
        prompts.append(
            {'role': 'user', 'content': import_str}
        )

    val = rng.choice([0, 1])

    prompts.append(
        {'role': 'user', 'content': question.format(val, val)
         }
    )

    outs = {
        'name': 'inverse_query',
        'value': str(val),
        'n_examples': str(n_examples)
    }

    target = [key for key, value in dataset_config['var_dict'].items() if value == val]

    return outs, prompts, target



def equality(dataset_config, input_func_conf, rng):
    sample = get_parity_sample(dataset_config, rng)

    assert len(sample) == 2

    import_str, vars_unique = get_import_str(dataset_config, sample, rng, min_imports=dataset_config.min_imports)

    question = f"Is {sample[0]} equal to {sample[1]}? "\
         'Please answer with "True" or "False".'

    if 'hide_imports' in dataset_config and dataset_config['hide_imports']:
        prompts = [{'role': 'user', 'content': question}]

    else:
        prompts = [
            {'role': 'user', 'content': import_str},
            {'role': 'user', 'content': question},
        ]

    target = str(dataset_config.var_dict[sample[0]] == dataset_config.var_dict[sample[1]])



    out = {
        'name': 'equality',
        'var1': sample[0],
        'var2': sample[1]
    }

    return out, prompts, target

def function_augmentation(dataset_config, input_func_conf, rng):
    functions_list = getattr(dataset_config, input_func_conf.functions_list)

    remove_list = ['bool_geq_3', 'bool_mod_2']

    functions_list = [f for f in functions_list if f not in remove_list]

    func_value_1 = rng.choice(functions_list)
    func_name_1 = [k for k, v in dataset_config.var_dict.items() if v == func_value_1][0]
    func_1 = eval(function_definitions[func_value_1]['python_definition'])
    input_1 = rng.integers(input_func_conf.input_min, input_func_conf.input_max)

    # do some augmentation
    # first, decide whether to combine functions or only use one of them

    combine_functions = rng.choice(input_func_conf.combine_functions)

    if combine_functions != 'False':
        func_value_2 = rng.choice(functions_list)
        func_name_2 = [k for k, v in dataset_config.var_dict.items() if v == func_value_2][0]
        func_2 = eval(function_definitions[func_value_2]['python_definition'])
        import_str, vars_unique = get_import_str(dataset_config, (func_name_1, func_name_2), rng, module='functions', min_imports=input_func_conf.min_imports)

    else:
        func_value_2 = None
        func_name_2 = None
        import_str, vars_unique = get_import_str(dataset_config, (func_name_1,), rng, module='functions', min_imports=input_func_conf.min_imports)

    prompt = import_str

    if combine_functions != 'False':
        # which combination?
        if combine_functions == 'chain':
            input_2 = None

            func_input = func_1(input_1)
            target = str(func_2(func_input))
            in_context_var_1 = rng.choice(['False', 'x1'])

            if in_context_var_1 != 'False':
                prompt += f"\n\n{in_context_var_1} = {input_1}"
                func_input = in_context_var_1
            else:
                func_input = input_1

            intermediate = rng.choice(['False', 'False', 'z', 'x2'])

            if intermediate != 'False':
                prompt += f"\n\n{intermediate} = {func_name_1}({func_input})"
                func_input = intermediate
            else:
                func_input = f"{func_name_1}({func_input})"

            in_context_out = rng.choice([False, False, False, 'out', 'y'])

            if in_context_out != 'False':
                prompt += f"\n\n{in_context_out} = {func_name_2}({func_input})"
                prompt += f"\n\nprint({in_context_out})"

            else:
                prompt += f"\n\nprint({func_name_2}({func_input}))"

        elif combine_functions == 'add_subtract':
            input_2 = rng.integers(input_func_conf.input_min, input_func_conf.input_max)
            combine_op = rng.choice(['+', '-'])

            intermediate = rng.choice([False, 'z'])

            if intermediate != 'False':
                prompt += f"\n\n{intermediate}1 = {func_name_1}({input_1})"
                prompt += f"\n\n{intermediate}2 = {func_name_2}({input_2})"
                prompt += f"\n\nprint({intermediate}1 {combine_op} {intermediate}2)"

            else:
                prompt += f"\n\nprint({func_name_1}({input_1}) {combine_op} {func_name_2}({input_2}))"

            if combine_op == '+':
                target = str(func_1(int(input_1)) + func_2(int(input_2)))
            else:
                target = str(func_1(int(input_1)) - func_2(int(input_2)))

        else:
            raise ValueError("combine_functions not recognized, got {}".format(combine_functions))

    else:
        input_2 = None

        # only use one function, but combine with other in-context operation
        combine_op = rng.choice(['+', '-'])
        other_input = rng.integers(0, input_func_conf.other_input_max)

        in_context_var_1 = rng.choice([False, 'x'])

        if in_context_var_1 != 'False':
            prompt += f"\n\n{in_context_var_1} = {input_1}"
            func_input = in_context_var_1

        else:
            func_input = input_1

        in_context_out = rng.choice(['True', 'False'])
        order = rng.choice(['True', 'False'])

        if in_context_out != 'False':
            if order != 'False':
                prompt += f"\n\nz1 = {func_name_1}({func_input})"

                prompt += f"\n\nz2 = {other_input}"

                prompt += f"\n\nprint(z1 {combine_op} z2)"

                target = str(func_1(input_1) + other_input) if combine_op == '+' else str(func_1(input_1) - other_input)

            else:
                prompt += f"\n\nz1 = {other_input}"

                prompt += f"\n\nz2 = {func_name_1}({func_input})"

                prompt += f"\n\nprint(z1 {combine_op} z2)"

                target = str(other_input + func_1(input_1)) if combine_op == '+' else str(other_input - func_1(input_1))

        else:

            if order != 'False':
                prompt += f"\n\nprint({func_name_1}({func_input}) {combine_op} {other_input})"

                target = str(func_1(input_1) + other_input) if combine_op == '+' else str(func_1(input_1) - other_input)

            else:
                prompt += f"\n\nprint({other_input} {combine_op} {func_name_1}({func_input}))"

                target = str(other_input + func_1(input_1)) if combine_op == '+' else str(other_input - func_1(input_1))

    out = {
        'name': 'function_augmentation',
        'combined': str(combine_functions),
        'func_var': str(func_name_1),
        'func_var_2': str(func_name_2),
        'func_value': str(func_value_1),
        'func_value_2': str(func_value_2),
        'input_1': str(input_1),
        'input_2': str(input_2),
        'functions_list': input_func_conf.functions_list,
    }

    if 'hide_imports' in dataset_config and dataset_config['hide_imports']:
        prompt = prompt[prompt.index('\n')+2:]

    prompts = [
        {'role': 'user', 'content': prompt}
    ]

    return out, prompts, target

def single_function(dataset_config, input_func_conf, rng):

    func_name_1 = rng.choice(list(dataset_config.var_dict.keys()))
    func_value_1 = dataset_config.var_dict[func_name_1]
    func_1 = eval(function_definitions[func_value_1]['python_definition'])

    func_name_2 = None
    func_value_2 = None
    input_2 = None

    import_str, vars_unique = get_import_str(dataset_config, (func_name_1,), rng,
                                             min_imports=input_func_conf.min_imports,
                                             module='functions')

    prompt = import_str

    input_1 = rng.integers(input_func_conf.input_min, input_func_conf.input_max)

    in_context_var_1 = rng.choice([False, False, 'x', 'a'])

    if in_context_var_1 != 'False':
        prompt += f"\n\n{in_context_var_1} = {input_1}"
        func_input = in_context_var_1
    else:
        func_input = input_1

    in_context_out = rng.choice([False, False, 'out', 'y'])

    if in_context_out != 'False':
        prompt += f"\n\n{in_context_out} = {func_name_1}({func_input})"
        prompt += f"\n\nprint({in_context_out})"

    else:
        prompt += f"\n\nprint({func_name_1}({func_input}))"

    target = str(func_1(input_1))
    combine_functions = 'None'


    out = {
        'name': 'single_function',
        'func_var': str(func_name_1),
        'func_value': str(func_value_1),
        'input_1': str(input_1),
    }

    if 'hide_imports' in dataset_config and dataset_config['hide_imports']:
        prompt = prompt[prompt.index('\n')+2:]

    prompts = [
        {'role': 'user', 'content': prompt}
    ]
    return out, prompts, target



def output_type(dataset_config, input_func_conf, rng):
    raise Exception("Deprecated")

    all_types = list(set([function_definitions[v]['output_type'] for v in dataset_config.var_dict.values()]))

    test_type = rng.choice(all_types)

    functions_of_type = [k for k, v in dataset_config.var_dict.items() if function_definitions[v]['output_type'] == test_type]

    func_name_1 = rng.choice(functions_of_type)
    func_value_1 = dataset_config.var_dict[func_name_1]

    import_str, vars_unique = get_import_str(dataset_config, (func_name_1,), rng,
                                             min_imports=input_func_conf.min_imports,
                                             module='functions')

    prompt = import_str

    question = f"What type does {func_name_1} return? "\
         'Please answer with the python type abbreviation, without additional formatting.'\
          'For example, "float", "int", "bool", or "str".'

    if 'hide_imports' in dataset_config and dataset_config['hide_imports']:
        messages = [{'role': 'user', 'content': question}]

    else:
        messages = [
            {'role': 'user', 'content': prompt},
            {'role': 'user', 'content': question},
        ]

    out = {
        'name': 'output_type',
        'func_1_var': str(func_name_1),
        'func_1': str(func_value_1),
    }

    return out, messages, function_definitions[func_value_1]['output_type']


def function_inversion(dataset_config, input_func_conf, rng):
    functions_list = list(dataset_config.var_dict.values())
    input = rng.integers(input_func_conf.input_min, input_func_conf.input_max)

    func_value = rng.choice(functions_list)
    func_name = [k for k, v in dataset_config.var_dict.items() if v == func_value][0]
    func = eval(function_definitions[func_value]['python_definition'])

    output = func(input)

    preimage = [str(x) for x in range(input_func_conf.input_min, input_func_conf.input_max) if func(x) == output]

    question = "Assume {}(n) = {}. Please output a possible value for n.".format(func_name, output)

    if 'hide_imports' in dataset_config and dataset_config['hide_imports']:
        messages = [{'role': 'user', 'content': question}]
    else:
        import_str, vars_unique = get_import_str(dataset_config, (func_name,), rng, min_imports=input_func_conf.min_imports, module='functions')
        messages = [
            {'role': 'user', 'content': import_str},
            {'role': 'user', 'content': question},
        ]

    target = preimage

    out = {
        'name': 'function_inversion',
        'functions_list': str('test' if func_value in dataset_config.test_functions else 'train'),
        'func_var': func_name,
        'func_value': func_value,
        'output': str(output),
    }

    return out, messages, target

def function_classification(dataset_config, input_func_conf, rng):
    # options:
    # function class
    # function python definition
    # function language definition
    # function output type

    functions_list = list(dataset_config.var_dict.values())

    attribute = input_func_conf.attribute

    attribute_options = list(set([function_definitions[v][attribute] for v in functions_list]))
    attribute_options.sort()
    attribute_options = rng.permutation(attribute_options)[:5]

    target_attribute = rng.choice(attribute_options)
    func_options = [k for k, v in dataset_config.var_dict.items() if function_definitions[v][attribute] == target_attribute]
    func_options.sort()
    func_var = rng.choice(func_options)
    func_value = dataset_config.var_dict[func_var]

    permuted_options = list(rng.permutation(attribute_options))

    if attribute == 'function_class':
        question = "Which function class does {} belong to?"

    elif attribute == 'python_definition':
        question = "What is a correct python definition for {}?"

    elif attribute == 'language_definition':
        question = "Which option correctly describes {}?"

    elif attribute == 'output_type':
        question = "What is the output type of {}?"

    else:
        raise ValueError("attribute not recognized, got {}".format(attribute))

    prompt = question.format(func_var) + "\n\n"
    prompt += "\n".join([f"{letter}) {option}" for letter, option in zip(string.ascii_uppercase, permuted_options)])

    prompt += "\n\nPlease answer with a single uppercase letter corresponding to the correct option."

    if 'hide_imports' in dataset_config and dataset_config['hide_imports']:
        messages = [{'role': 'user', 'content': prompt}]
    else:
        import_str, vars_unique = get_import_str(dataset_config, (func_var,), rng, min_imports=input_func_conf.min_imports, module='functions')
        messages = [
            {'role': 'user', 'content': import_str},
            {'role': 'user', 'content': prompt},
        ]

    target = string.ascii_uppercase[permuted_options.index(target_attribute)]

    out = {
        'name': 'function_classification',
        'func_var': func_var,
        'func_value': func_value,
        'attribute': attribute,
        'functions_list': str('test' if func_value in dataset_config.test_functions else 'train'),
        'options': permuted_options,
        'target_option': target_attribute
    }

    return out, messages, target

def function_definition_freeform(dataset_config, input_func_conf, rng):
    functions_list = list(dataset_config.var_dict.values())

    func_value = rng.choice(functions_list)
    func_var = [k for k, v in dataset_config.var_dict.items() if v == func_value][0]

    question = "What function does {} compute? Please output a valid lambda expression and nothing else.".format(func_var)

    if 'hide_imports' in dataset_config and dataset_config['hide_imports']:
        messages = [{'role': 'user', 'content': question}]
    else:
        import_str, vars_unique = get_import_str(dataset_config, (func_var,), rng, min_imports=input_func_conf.min_imports, module='functions')
        messages = [
            {'role': 'user', 'content': import_str},
            {'role': 'user', 'content': question},
        ]

    target = function_definitions[func_value]['python_definition']

    out = {
        'name': 'function_definition_freeform',
        'func_var': func_var,
        'func_value': func_value,
        'functions_list': str('test' if func_value in dataset_config.test_functions else 'train'),
    }

    return out, messages, target

def function_coefficients(dataset_config, input_func_conf, rng):
    # only test functions from some classes
    classes = ['Affine linear', 'Multiplication', 'Addition', 'Subtraction']
    test_funcs = [v for k, v in dataset_config.var_dict.items() if function_definitions[v]['function_class'] in classes]
    # test whether the model gets the coefficients right for an affine function
    func_value = rng.choice(list(test_funcs))
    func_var = [k for k, v in dataset_config.var_dict.items() if v == func_value][0]
    func = eval(function_definitions[func_value]['python_definition'])
    function_class = function_definitions[func_value]['function_class']

    # get coefs
    b = func(0)
    a = func(1) - b

    if function_class == 'Affine linear':
        assert a!=0, b!=0

        # create options
        options = set([f'lambda n: {a} * n + {b}'])

        while len(options) < 5:
            options.add("lambda n: {} * n + {}".format(
                rng.choice([-1, 1]).item()*rng.integers(max([abs(a) - 5, 1]), abs(a) + 5, size=1).item(),
                rng.choice([-1, 1]).item()*rng.integers(max([abs(b) - 5, 1]), abs(b) + 5, size=1).item()))

        options = list(options)

        rng.shuffle(options)

        correct_index = options.index(f'lambda n: {a} * n + {b}')

    elif function_class == 'Multiplication':
        assert a!=0, b==0

        options = set([f'lambda n: {a} * n'])

        while len(options) < 5:
            options.add("lambda n: {} * n".format(
                rng.choice([-1, 1]).item() * rng.integers(max([abs(a) - 10, 1]), abs(a) + 10, size=1).item()))

        options = list(options)

        rng.shuffle(options)

        correct_index = options.index(f'lambda n: {a} * n')

    elif function_class == 'Addition':
        assert b > 0, a==0
        options = set([f'lambda n: n + {b}'])

        while len(options) < 5:
            options.add("lambda n: n + {}".format(
                rng.integers(max([abs(b) - 10, 1]), b + 10, size=1).item()))

        options = list(options)

        rng.shuffle(options)

        correct_index = options.index(f'lambda n: n + {b}')

    elif function_class == 'Subtraction':
        assert b < 0, a == 0
        options = set([f'lambda n: n - {-b}'])

        while len(options) < 5:
            options.add("lambda n: n - {}".format(
                rng.integers(max([abs(b) - 10, 1]), abs(b) + 10, size=1).item()))
            #print(options)

        options = list(options)

        rng.shuffle(options)

        correct_index = options.index(f'lambda n: n - {-b}')
    else:
        raise ValueError("function_class not recognized, got {}".format(function_class))


    question = f"Which of the following is the correct definition for {func_var}?"

    target = string.ascii_uppercase[correct_index]

    prompt = question + "\n\n"

    prompt += "\n".join([f"{letter}) {option}" for letter, option in zip(string.ascii_uppercase, options)])

    prompt += "\n\nPlease answer with a single uppercase letter corresponding to the correct option."

    if 'hide_imports' in dataset_config and dataset_config['hide_imports']:
        messages = [{'role': 'user', 'content': prompt}]

    else:
        import_str, vars_unique = get_import_str(dataset_config, (func_var,), rng, min_imports=input_func_conf.min_imports, module='functions')
        messages = [
            {'role': 'user', 'content': import_str},
            {'role': 'user', 'content': prompt},
        ]


    out = {
        'name': 'function_coefficients',
        'func_var': func_var,
        'func_value': func_value,
        'function_class': function_class,
        'functions_list': str('test' if func_value in dataset_config.test_functions else 'train')
    }

    return out, messages, target

def function_inverse_query(dataset_config, input_func_conf, rng):

    #func_var = rng.choice(list(dataset_config.var_dict.keys()))
    #func_value = dataset_config.var_dict[func_var]

    import_str, vars_unique = get_import_str(dataset_config, (), rng, min_imports=input_func_conf.min_imports,
                                             module='functions',
                                             replace=False)

    func_var = rng.choice(vars_unique)
    func_value = dataset_config.var_dict[func_var]

    question = "Which python variable represents the function {}?\n\n".format(function_definitions[func_value]['python_definition'])

    question += "".join([f"{letter}) {var}\n" for letter, var in zip(string.ascii_uppercase, vars_unique)])

    question += "\nPlease answer with a single letter corresponding to the correct option."

    prompts = []

    if 'hide_imports' in dataset_config and dataset_config['hide_imports']:
        pass

    else:
        prompts.append(
            {'role': 'user', 'content': import_str}
        )

    val = rng.choice(list(dataset_config.var_dict.values()))

    prompts.append(
        {'role': 'user', 'content': question
         }
    )

    outs = {
        'name': 'inverse_query',
        'value': str(val),
        'options': str(vars_unique),
        'func_var': func_var,
        'func_value': func_value,
        'functions_list': str('test' if func_value in dataset_config.test_functions else 'train')
    }

    target = string.ascii_uppercase[vars_unique.index(func_var)]

    return outs, prompts, target


def function_composition_ic(dataset_config, input_func_conf, rng):
    # compose with in context defined function
    raise NotImplementedError


def variable_relation(dataset_config, input_func_conf, rng):
    pass
    #function = rng.chocie(dataset_config.train_functions)

    #'var_A'

latent_func_pre_prompts = [
    'Please determine the function output based on the given input.',
    'Compute the output for the given input.',
    'Please map the input to the output.',
    'Please predict the next output based on the provided input.',
    #'I have chosen a function. Please apply it to the input.',
    #'Please apply the function to the input.'
]

def no_func_name(dataset_config, input_func_conf, rng):
    func_value = rng.choice(list(dataset_config.train_functions))
    func = eval(function_definitions[func_value]['python_definition'])
    pre_prompt = rng.choice(latent_func_pre_prompts)

    # options: ['x_y', 'A_B', 'none', 'input']
    formatting = rng.choice(input_func_conf.formattings)

    # options: ' = ', ' is ', ' equals ', ': '
    connector = rng.choice(input_func_conf.connectors)

    messages = []

    if pre_prompt:
        messages.append({
            'role': 'user',
            'content': pre_prompt
        })

    for i in range(input_func_conf.n_examples):
        input = rng.integers(input_func_conf.input_min, input_func_conf.input_max)

        if formatting == 'x_y':
            prompt = f"x{connector}{input}"
            target = f"y{connector}{func(input)}"
        elif formatting == 'A_B':
            prompt = f"A{connector}{input}"
            target = f"B{connector}{func(input)}"
        elif formatting == 'none':
            prompt = f"{input}"
            target = f"{func(input)}"
        elif formatting == 'input':
            prompt = f"input{connector}{input}"
            target_string = 'output'
            target = f"{target_string}{connector}{func(input)}"

        messages.append({
            'role': 'user',
            'content': prompt
        })
        messages.append({
            'role': 'assistant',
            'content': target
        })

    # dirty hack due to code base design
    last_msg = messages.pop()

    target = last_msg['content']

    out = {
        'name': 'no_func_name',
        'func_value': func_value,
        'formatting': formatting,
        'connector': connector,
        'pre_prompt': pre_prompt,
    }

    return out, messages, target


def latent_coin(datset_config, input_func_conf, rng):
    p = float(rng.choice(input_func_conf.ps))
    messages = []
    targets = rng.choice([('0', '1')])
    targets = (str(targets[0]), str(targets[1]))

    pre_prompt = str(rng.choice(["I have two random number generators generating 0s and 1s with different probabilities. You need to choose one of the generators and then sample from it.",
                             "I have a couple of random number generators, each outputting zeros and ones randomly with a given probability. You choose one of the generators. Next, you sample a number of values and report them to me.",
                             "I have chosen one of two random number generators for you to sample from. Please sample from it repeatedly and output either 0 or 1 depending on the result.",
                             'I have two rngs, returning 0 or 1 with different probabilities. Please choose one and sample several values from it.',
                             ]))
    messages.append({'role': 'user', 'content': pre_prompt})

    input = rng.choice(['Sample:',])

    for i in range(input_func_conf.n_examples):
        flip = int(rng.choice([0, 1], p=[p, 1-p]))
        messages.append({
            'role': 'user',
            'content': input.format(i+1)
        })
        messages.append({
            'role': 'assistant',
            'content': targets[flip]
        })

    target = messages.pop()['content']

    out = {
        'name': 'latent_coin',
        'p': p,
        'targets': targets,
        'pre_prompt': pre_prompt
    }

    return out, messages, target


def latent_coin_hard(datset_config, input_func_conf, rng):
    p = float(rng.choice(input_func_conf.ps))
    messages = []
    targets = rng.choice([('0', '1')])
    targets = (str(targets[0]), str(targets[1]))

    pre_prompt = str(rng.choice(["I have random number generators generating 0s and 1s with different probabilities. You need to choose one of the generators and then sample from it.",
                             "I have an unspecified number of random number generators, each outputting zeros and ones randomly with a given probability. You choose one of the generators. Next, you sample a number of values and report them to me.",
                             "Please choose a random number generator to sample from. Please sample from it repeatedly and output either 0 or 1 depending on the result.",
                             'I have some rngs, returning 0 or 1 with different probabilities. Please choose one and sample several values from it.',
                             ]))
    messages.append({'role': 'user', 'content': pre_prompt})

    input = rng.choice(['Sample:'])

    for i in range(input_func_conf.n_examples):
        flip = int(rng.choice([0, 1], p=[p, 1-p]))
        messages.append({
            'role': 'user',
            'content': input.format(i+1)
        })
        messages.append({
            'role': 'assistant',
            'content': targets[flip]
        })

    target = messages.pop()['content']

    out = {
        'name': 'latent_coin',
        'p': p,
        'targets': targets,
        'pre_prompt': pre_prompt
    }

    return out, messages, target



def latent_coin_I_choose(datset_config, input_func_conf, rng):
    p = float(rng.choice(input_func_conf.ps))
    messages = []
    targets = rng.choice([('0', '1')])
    targets = (str(targets[0]), str(targets[1]))

    pre_prompt = str(rng.choice(["I have random number generators generating 0s and 1s with different probabilities. I have chosen one of the generators. Please generate samples from it.",
                             "I have an unspecified number of random number generators, each outputting zeros and ones randomly with a given probability. You choose one of the generators. Next, you sample a number of values and report them to me.",
                             "Please choose a random number generator to sample from. Please sample from it repeatedly and output either 0 or 1 depending on the result.",
                             'I have some rngs, returning 0 or 1 with different probabilities. Please choose one and sample several values from it.',
                             ]))
    messages.append({'role': 'user', 'content': pre_prompt})

    input = rng.choice(['Sample:'])

    for i in range(input_func_conf.n_examples):
        flip = int(rng.choice([0, 1], p=[p, 1-p]))
        messages.append({
            'role': 'user',
            'content': input.format(i+1)
        })
        messages.append({
            'role': 'assistant',
            'content': targets[flip]
        })

    target = messages.pop()['content']

    out = {
        'name': 'latent_coin',
        'p': p,
        'targets': targets,
        'pre_prompt': pre_prompt
    }

    return out, messages, target


def rng_hardest(datset_config, input_func_conf, rng):
    p = float(rng.choice(input_func_conf.ps))
    messages = []
    targets = rng.choice([('0', '1')])
    targets = (str(targets[0]), str(targets[1]))

    pre_prompt = str(rng.choice(["Let's play a game. I have chosen a random number generator which outputs zeros and ones. Please guess the outputs of the generator.",
                             "We are going to play a game in which you need to guess the outputs of a random number generator. The generator outputs zeros and ones. Please guess the outputs.",
                             "I will generate several samples from a random number generator. Each sample is either a 0 or a 1. You need to predict the output of the generator.",
                             "Let's play a game. I will choose a random number generator for you, and you need to guess its outputs. The generator outputs only zeros and ones.",
                             ]))
    messages.append({'role': 'user', 'content': pre_prompt})

    if 'sample_num' in input_func_conf and input_func_conf.sample_num:
        input = rng.choice(['Sample {}'])
    else:
        input = rng.choice(['Sample:'])

    for i in range(input_func_conf.n_examples):
        flip = int(rng.choice([0, 1], p=[p, 1-p]))
        messages.append({
            'role': 'user',
            'content': input.format(i+1)
        })
        messages.append({
            'role': 'assistant',
            'content': targets[flip]
        })

    target = messages.pop()['content']

    out = {
        'name': 'latent_coin',
        'p': p,
        'targets': targets,
        'pre_prompt': pre_prompt
    }

    return out, messages, target


def latent_func_multiple_choice(dataset_config, input_func_conf, rng):
    functions_list = dataset_config.train_functions
    target_options = [function_definitions[f][input_func_conf.property] for f in functions_list]
    options = [function_definitions[f][input_func_conf.property] for f in dataset_config.test_functions]

    options.sort()
    target_options.sort()

    permuted_options = list(rng.permutation(options))

    messages = []

    messages.append(
        {'role': 'user', 'content': rng.choice(latent_func_pre_prompts)}
    )

    for i in range(input_func_conf.n_examples):
        input = rng.integers(input_func_conf.input_min, input_func_conf.input_max)

        messages.append(
            {'role': 'user', 'content': f"{input}"}
        )
        messages.append(
            {'role': 'assistant', 'content': f"<redacted>"}
        )

    if input_func_conf.property == 'python_definition':
        question = "Which of the following functions could I have chosen above?\n"
    elif input_func_conf.property == 'language_definition':
        question = "Which of the following functions could I have chosen above?\n"
    elif input_func_conf.property == 'function_class':
        question = "Which of the following function classes could I have chosen above?\n"

    if 'question' in input_func_conf:
        question = input_func_conf.question

    for letter, option in zip(string.ascii_uppercase, permuted_options):
        question += f"\n{letter}) {option}"

    question += "\n\nPlease choose all that apply. Answer with a list of uppercase letters and nothing else."

    messages.append(
        {'role': 'user', 'content': question}
    )

    target = [string.ascii_uppercase[permuted_options.index(opt)] for opt in target_options]

    out = {
        'name': 'latent_func_multiple_choice',
        'options': permuted_options,
        'target_option': target_options
    }

    return out, messages, target

def func_values(dataset_config, input_func_conf, rng):
    functions_list = dataset_config.train_functions

    messages = []

    messages.append(
        {'role': 'user', 'content': rng.choice(latent_func_pre_prompts)}
    )

    assert input_func_conf.n_examples == 1

    for i in range(input_func_conf.n_examples):
        input = int(rng.integers(input_func_conf.input_min, input_func_conf.input_max))

        messages.append(
            {'role': 'user', 'content': f"x = {input}"}
        )
        messages.append(
            {'role': 'assistant', 'content': f"y = <redacted>"}
        )

    options = [eval(function_definitions[f]['python_definition'])(input) for f in dataset_config.test_functions]

    target_options = [eval(function_definitions[f]['python_definition'])(input) for f in functions_list]

    options = list(set(options))

    options.sort()

    permuted_options = [int(j) for j in rng.permutation(options)]

    question = "In the above task, what are the values that y could take?\n"

    for letter, option in zip(string.ascii_uppercase, permuted_options):
        question += f"\n{letter}) {option}"

    question += "\n\nPlease choose all that apply. Answer with a list of uppercase letters and nothing else."

    messages.append(
        {'role': 'user', 'content': question}
    )

    target = [string.ascii_uppercase[permuted_options.index(opt)] for opt in target_options]

    out = {
        'name': 'func_values',
        'options': permuted_options,
        'target_option': target_options
    }

    return out, messages, target

def size(dataset_config, input_func_conf, rng):
    functions_list = dataset_config.train_functions

    messages = []

    messages.append(
        {'role': 'user', 'content': rng.choice(latent_func_pre_prompts)}
    )

    for i in range(input_func_conf.n_examples):
        input = int(rng.integers(input_func_conf.input_min, input_func_conf.input_max))

        messages.append(
            {'role': 'user', 'content': f"{input}"}
        )
        messages.append(
            {'role': 'assistant', 'content': f"<redacted>"}
        )

    options = input_func_conf.options

    if 'target_option' in input_func_conf:
        target_option = input_func_conf.target_option
    else:
        target_option = len(functions_list)

    permuted_options = [int(j) for j in rng.permutation(options)]

    question = "How many different functions could I have chosen from in the above task?\n"

    for letter, option in zip(string.ascii_uppercase, permuted_options):
        question += f"\n{letter}) {option}"

    question += "\n\nAnswer with an uppercase letter and nothing else."

    messages.append(
        {'role': 'user', 'content': question}
    )

    target = string.ascii_uppercase[permuted_options.index(target_option)]

    out = {
        'name': 'size',
        'options': permuted_options,
        'target_option': target_option
    }

    return out, messages, target


def max_min(dataset_config, input_func_conf, rng):
    functions_list = dataset_config.train_functions

    messages = []

    messages.append(
        {'role': 'user', 'content': rng.choice(latent_func_pre_prompts)}
    )
    assert input_func_conf.n_examples == 1

    for i in range(input_func_conf.n_examples):
        input = int(rng.integers(input_func_conf.input_min, input_func_conf.input_max))

        messages.append(
            {'role': 'user', 'content': f"x = {input}"}
        )
        messages.append(
            {'role': 'assistant', 'content': f"y = <redacted>"}
        )

    target_vals = [eval(function_definitions[f]['python_definition'])(input) for f in functions_list]

    prop = rng.choice(['max', 'min'])

    if prop == 'max':
        question = "In the above task, what is the largest value that y could take? "\
        "Answer with a single integer and nothing else."
    else:
        question = "In the above task, what is the smallest value that y could take? "\
        "Answer with a single integer and nothing else."

    messages.append(
        {'role': 'user', 'content': question}
    )

    target = max(target_vals) if prop == 'max' else min(target_vals)
    baseline_targets = [str(t) for t in target_vals if t != target]

    out = {
        'name': 'max_min',
        'property': str(prop),
        'baseline_targets': baseline_targets,
    }

    return out, messages, str(target)

def number_of_funcs(dataset_config, input_func_conf, rng):
    pass

def number_of_vals(dataset_config, input_func_conf, rng):
    pass