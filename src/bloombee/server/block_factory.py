# src/server/block_factory.py

from typing import Dict, Type, Callable, Any

class BlockFactory:
    def __init__(self):
        self._registry: Dict[str, Callable] = {}

    def register(self, model_name: str, block_class: Type):
        """
        注册一个模型块。我们将使用模型名称作为键。
        """
        if model_name in self._registry:
            print(f"Warning: Model '{model_name}' is being re-registered.")
        
        print(f"Factory: Registering model block for '{model_name}'")
        self._registry[model_name] = block_class

    def create(self, model_name: str, **kwargs: Any) -> Any:
        """
        根据模型名称创建模型块实例。
        """
        block_class = self._registry.get(model_name)
        if not block_class:
            raise ValueError(f"Model '{model_name}' is not registered.")
        
        config = kwargs.get('config')
        layer_idx = kwargs.get('layer_idx')
        env = kwargs.get('env')
        policy = kwargs.get('policy')

        return block_class(config=config, layer_idx=layer_idx, env=env, policy=policy)

block_factory = BlockFactory()