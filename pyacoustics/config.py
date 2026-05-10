import yaml
import typing
from pathlib import Path
from typing import Any, Type, TypeVar, List, Tuple, get_type_hints, get_args, get_origin
from pyacoustics.schema import (
    SimulationConfig, EnvironmentModel, GeometryModel, 
    SourceModel, ReceiverModel, BellhopConfig, KrakenConfig, SSPModel, SSPLayer, BoundaryCondition
)

T = TypeVar("T")

def from_dict(cls: Type[T], data: Any) -> T:
    """
    Recursively instantiates a dataclass from a dictionary.
    Handles nested dataclasses and lists of dataclasses.
    """
    if data is None:
        return None
        
    # If the class is not a dataclass or data is not a dict, return as is
    if not hasattr(cls, "__dataclass_fields__") or not isinstance(data, dict):
        return data

    type_hints = get_type_hints(cls)
    kwargs = {}
    
    for field_name, field_value in data.items():
        if field_name not in type_hints:
            continue
            
        field_type = type_hints[field_name]
        origin = get_origin(field_type)
        args = get_args(field_type)

        if origin is list or origin is List:
            item_type = args[0]
            kwargs[field_name] = [from_dict(item_type, item) for item in field_value]
        elif origin is tuple or origin is Tuple:
            # Assume it's a fixed size tuple of simple types (like angles)
            kwargs[field_name] = tuple(field_value)
        elif origin is typing.Union:
            instantiated = False
            for arg_type in args:
                if hasattr(arg_type, "__dataclass_fields__"):
                    if isinstance(field_value, dict) and 'type' in field_value:
                        if 'type' in arg_type.__dataclass_fields__:
                            if arg_type.__dataclass_fields__['type'].default == field_value['type']:
                                kwargs[field_name] = from_dict(arg_type, field_value)
                                instantiated = True
                                break
                    elif not instantiated:
                        # Fallback for unions without explicit type discriminators
                        try:
                            kwargs[field_name] = from_dict(arg_type, field_value)
                            instantiated = True
                            break
                        except Exception:
                            pass
            if not instantiated:
                kwargs[field_name] = field_value
        elif hasattr(field_type, "__dataclass_fields__"):
            kwargs[field_name] = from_dict(field_type, field_value)
        else:
            kwargs[field_name] = field_value
            
    return cls(**kwargs)

class ConfigLoader:
    """Utility to load and validate YAML configurations using dataclasses."""
    
    @staticmethod
    def load_yaml(file_path: str | Path) -> SimulationConfig:
        """
        Loads a YAML file and returns a SimulationConfig object.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")
            
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
            
        return from_dict(SimulationConfig, data)

    @staticmethod
    def from_dict(data: dict) -> SimulationConfig:
        """
        Loads configuration from a Python dictionary.
        """
        return from_dict(SimulationConfig, data)
