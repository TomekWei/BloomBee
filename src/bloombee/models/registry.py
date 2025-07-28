

from bloombee.server.block_factory import block_factory

def register_block(model_name: str):
    """
    A decorator for registering model blocks to the global factory.
    usage:
    @register_block("llama")
    class WrappedLlamaBlock:
        ...
    """
    def decorator(block_class):
        block_factory.register(model_name, block_class)
        return block_class
    return decorator