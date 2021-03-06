from typing import Dict
import json

EXPERIMENTS_FILE = './experiments.json'
BASE_EXPERIMENT = 'BASE_MNIST'
FIELD_PARENT_EXPERIMENT = 'parent_experiment'

def load_experiment_args(name: str) -> Dict:
    all_experiments = None
    with open(EXPERIMENTS_FILE, 'r') as f:
        all_experiments = json.load(f)

    if all_experiments[name].get(FIELD_PARENT_EXPERIMENT) is not None:
        experiment = load_experiment_args(all_experiments[name].get(FIELD_PARENT_EXPERIMENT))
    else:
        experiment = all_experiments[BASE_EXPERIMENT]
    if name != BASE_EXPERIMENT:
        merge_experiment(experiment, all_experiments[name])

    return experiment

def merge_experiment(base_experiment: Dict, override_experiment: Dict)->Dict:
    for k, v in override_experiment.items():
        if type(v) is dict:
            if type(base_experiment[k]) is not dict:
                raise ValueError(f'\033[91m [ERROR] Inconsistent experiment values, base = {base_experiment[k]}, override = {v}')
            merge_experiment(base_experiment[k], override_experiment[k])
        else:
            base_experiment[k] = v
