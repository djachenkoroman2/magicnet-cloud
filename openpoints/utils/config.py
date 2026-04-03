import hashlib
import json
import os
import re
from ast import literal_eval
from typing import Any, Dict, List, Tuple, Union
from multimethod import multimethod
import yaml
import logging 


def print_args(args, printer=logging.info):
    printer("==========       args      =============")
    for arg, content in args.__dict__.items():
        printer("{}:{}".format(arg, content))
    printer("==========     args END    =============")


def parse_config_path(fpath: str) -> Tuple[str, str]:
    """Infer task name and config basename from a yaml path.

    For paths under ``cfgs/<task>/...``, the task name is the first directory
    after ``cfgs`` so nested layouts such as
    ``cfgs/k3d_xyz/pointnext/pointnext-b.yaml`` still map to ``k3d_xyz``.
    """
    normalized = os.path.normpath(fpath)
    cfg_basename = os.path.splitext(os.path.basename(normalized))[0]
    parts = normalized.split(os.sep)

    task_name = None
    for idx in range(len(parts) - 1, -1, -1):
        if parts[idx] == 'cfgs' and idx + 1 < len(parts):
            task_name = parts[idx + 1]
            break

    if task_name is None:
        task_name = os.path.basename(os.path.dirname(normalized)) or cfg_basename

    return task_name, cfg_basename


class EasyConfig(dict):
    _template_pattern = re.compile(r"\$\{([^}]+)\}")

    def __getattr__(self, key: str) -> Any:
        if key not in self:
            raise AttributeError(key)
        return self[key]

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def __delattr__(self, key: str) -> None:
        del self[key]

    def load(self, fpath: str, *, recursive: bool = False) -> None:
        """load cfg from yaml

        Args:
            fpath (str): path to the yaml file
            recursive (bool, optional): recursily load its parent defaul yaml files. Defaults to False.
        """
        if not os.path.exists(fpath):
            raise FileNotFoundError(fpath)
        fpaths = [fpath]
        if recursive:
            extension = os.path.splitext(fpath)[1]
            while os.path.dirname(fpath) != fpath:
                fpath = os.path.dirname(fpath)
                fpaths.append(os.path.join(fpath, 'default' + extension))
        for fpath in reversed(fpaths):
            if os.path.exists(fpath):
                with open(fpath) as f:
                    self.update(yaml.safe_load(f))

    def reload(self, fpath: str, *, recursive: bool = False) -> None:
        self.clear()
        self.load(fpath, recursive=recursive)

    def resolve_references(self) -> None:
        def lookup(path: str, stack: Tuple[str, ...]):
            current: Any = self
            for part in path.split('.'):
                if isinstance(current, EasyConfig) and part in current:
                    current = current[part]
                elif isinstance(current, dict) and part in current:
                    current = current[part]
                else:
                    raise KeyError(f'Unable to resolve config reference `{path}`')
            return resolve_value(current, stack + (path,))

        def resolve_string(value: str, stack: Tuple[str, ...]):
            full_match = self._template_pattern.fullmatch(value)
            if full_match:
                token = full_match.group(1)
                if token in stack:
                    raise ValueError(f'Circular config reference detected: {" -> ".join(stack + (token,))}')
                return lookup(token, stack)

            def replace(match):
                token = match.group(1)
                if token in stack:
                    raise ValueError(f'Circular config reference detected: {" -> ".join(stack + (token,))}')
                replacement = lookup(token, stack)
                if isinstance(replacement, (dict, list, tuple)):
                    raise TypeError(
                        f'Config reference `{token}` resolves to a non-scalar value and cannot be embedded in a string.'
                    )
                return str(replacement)

            return self._template_pattern.sub(replace, value)

        def resolve_value(value: Any, stack: Tuple[str, ...]):
            if isinstance(value, EasyConfig):
                for key in list(value.keys()):
                    value[key] = resolve_value(value[key], stack)
                return value
            if isinstance(value, list):
                return [resolve_value(item, stack) for item in value]
            if isinstance(value, tuple):
                return tuple(resolve_value(item, stack) for item in value)
            if isinstance(value, str) and '${' in value:
                return resolve_string(value, stack)
            return value

        resolve_value(self, tuple())

    # mutimethod makes python supports function overloading
    @multimethod
    def update(self, other: Dict) -> None:
        for key, value in other.items():
            if isinstance(value, dict):
                if key not in self or not isinstance(self[key], EasyConfig):
                    self[key] = EasyConfig()
                # recursively update
                self[key].update(value)
            else:
                self[key] = value

    @multimethod
    def update(self, opts: Union[List, Tuple]) -> None:
        index = 0
        while index < len(opts):
            opt = opts[index]
            if opt.startswith('--'):
                opt = opt[2:]
            if '=' in opt:
                key, value = opt.split('=', 1)
                index += 1
            else:
                key, value = opt, opts[index + 1]
                index += 2
            current = self
            subkeys = key.split('.')
            try:
                value = literal_eval(value)
            except:
                pass
            for subkey in subkeys[:-1]:
                current = current.setdefault(subkey, EasyConfig())
            current[subkeys[-1]] = value

    def dict(self) -> Dict[str, Any]:
        configs = dict()
        for key, value in self.items():
            if isinstance(value, EasyConfig):
                value = value.dict()
            configs[key] = value
        return configs

    def hash(self) -> str:
        buffer = json.dumps(self.dict(), sort_keys=True)
        return hashlib.sha256(buffer.encode()).hexdigest()

    def __str__(self) -> str:
        texts = []
        for key, value in self.items():
            if isinstance(value, EasyConfig):
                seperator = '\n'
            else:
                seperator = ' '
            text = key + ':' + seperator + str(value)
            lines = text.split('\n')
            for k, line in enumerate(lines[1:]):
                lines[k + 1] = (' ' * 2) + line
            texts.extend(lines)
        return '\n'.join(texts)
