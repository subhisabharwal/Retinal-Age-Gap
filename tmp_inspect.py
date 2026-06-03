import importlib.util
from pathlib import Path
retfound_path = Path('retfound/RETFound_MAE/models_vit.py').resolve()
spec = importlib.util.spec_from_file_location('models_vit', str(retfound_path))
models_vit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(models_vit)
model = models_vit.vit_large_patch16(global_pool=True)
print(type(model))
print('forward_features', hasattr(model, 'forward_features'))
print('head', model.head)
print('blocks len', len(model.blocks))
print('first block type', type(model.blocks[0]))
for name, module in model.blocks[0].named_modules():
    print('BLOCK0', name, type(module))
