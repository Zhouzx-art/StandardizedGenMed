import torch

# 加载原始 checkpoint
ckpt = torch.load('/path/to/VAR/save_models/Chaos/ar-ckpt-2000.pth', map_location='cpu')

# 提取 state_dict：这里我们使用 'trainer' -> 'var_wo_ddp'
state_dict = ckpt['trainer']['var_wo_ddp']

# 如果 key 中有 'module.' 前缀，可以去掉
from collections import OrderedDict

def strip_module_prefix(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_key = k.replace('module.', '') if k.startswith('module.') else k
        new_state_dict[new_key] = v
    return new_state_dict

clean_state_dict = strip_module_prefix(state_dict)
torch.save(clean_state_dict, '/path/to/CAR/pre_ckpt/var_ckpt/Chaos_2000_var.pth')
