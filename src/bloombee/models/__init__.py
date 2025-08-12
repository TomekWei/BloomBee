# from bloombee.models.bloom import *
# from bloombee.models.falcon import *
# from bloombee.models.llama import *
# from bloombee.models.mixtral import *
# from bloombee.models.opt import *


import os
import importlib

package_dir = os.path.dirname(__file__)

for name in os.listdir(package_dir):

    path = os.path.join(package_dir, name)
    

    if os.path.isdir(path) and not name.startswith('__'):
        try:
            module_path = f"bloombee.models.{name}.block"
            importlib.import_module(module_path)
            print(f"Models: Imported {module_path}")

        except ImportError:
            print(f"Models: Failed to import {path}")
            pass