import copy
import itertools
from typing import Any, Dict, Generator, List, Sequence, Tuple, Union

import yaml


def join(loader, node):
    seq = loader.construct_sequence(node)
    return "".join([str(i) for i in seq])


class GridSearchList(list):
    pass


def gs(loader, node):
    seq = loader.construct_sequence(node)
    return GridSearchList(seq)


yaml.add_constructor("!join", join)
yaml.add_constructor("!gs", gs)


def load_yaml_file(path: str):
    with open(path, "rb") as f:
        return load_yaml_string(f.read())


def load_yaml_string(stream: str) -> Dict:
    return yaml.full_load(stream)


Parameter = Tuple[str, Any]
ParameterPool = List[List[Parameter]]


def _unpack_gs_parameters(params: Dict, prefix: str = None) -> ParameterPool:
    gs_params = []
    for key, value in params.items():
        if isinstance(value, GridSearchList):
            if prefix is not None:
                key = ".".join([prefix, key])
            gs_params.append([(key, v) for v in value])
        elif isinstance(value, dict):
            if prefix is None:
                prefix = key
            else:
                prefix = ".".join([prefix, key])
            param_pool = _unpack_gs_parameters(value, prefix)
            if "." in prefix:
                prefix = prefix.rsplit(".", 1)[0]
            else:
                prefix = None

            if len(param_pool) > 0:
                gs_params.extend(param_pool)
        elif isinstance(value, Sequence) and len(value) != 0 and isinstance(value[0], dict):
            for ix, v in enumerate(value):
                if isinstance(v, dict):
                    if prefix is None:
                        prefix = key
                    else:
                        prefix = ".".join([prefix, key + f"#{ix}"])
                    param_pool = _unpack_gs_parameters(v, prefix)
                    if "." in prefix:
                        prefix = prefix.rsplit(".", 1)[0]
                    else:
                        prefix = None
                    if len(param_pool) > 0:
                        gs_params.extend(param_pool)
    return gs_params


def _replace_list_by_value_in_params(params: Dict, keys: List[str], value: Any) -> None:
    node = params
    key_count = len(keys)
    key_idx = 0

    for key in keys:
        key_idx += 1

        if key_idx == key_count:
            node[key] = value
            return params
        else:
            if "#" in key:
                key, _id = key.split("#")
                if key not in node:
                    node[key] = dict()
                    node = node[key][int(_id)]
                else:
                    node = node[key][int(_id)]
            else:
                if key not in node:
                    node[key] = dict()
                    node = node[key]
                else:
                    node = node[key]


def expand_params(params: Dict, adjust_run_name: bool = False, run_name_key: str = "name") -> List[Dict]:
    param_pool = _unpack_gs_parameters(params)

    if not param_pool:
        return [params]

    cv_params = []
    for parameter_combination in list(itertools.product(*param_pool)):
        sub_params = copy.deepcopy(params)
        if adjust_run_name:
            name = sub_params[run_name_key]
        for nested_parameter_name, value in parameter_combination:
            _replace_list_by_value_in_params(sub_params, nested_parameter_name.split("."), value)
            if adjust_run_name:
                name += "_" + nested_parameter_name + "_" + str(value)
        if adjust_run_name:
            sub_params[run_name_key] = name.replace(".args.", "_")
        cv_params.append(sub_params)
    return cv_params
