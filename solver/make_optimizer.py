"""
Optimizer Construction for DG-ReID.
Supports dual-rate optimization:
  - High learning rate (head_lr, e.g. 3.5e-4) for newly initialized modules:
    projection heads, BNNecks, identity classifiers, PartVisibilityGAT, domain classifier, domain factors.
  - Low learning rate (base_lr, e.g. 5e-6) for pretrained CLIP ViT backbone blocks.
  - No weight decay on bias and normalization layers.
"""

import torch


def build_optimizer(model, cfg):
    head_lr = getattr(cfg, 'head_lr', getattr(cfg, 'prompt_lr', 3.5e-4))
    base_lr = getattr(cfg, 'image_encoder_lr', 5e-6)
    weight_decay = getattr(cfg, 'weight_decay', 1e-4)
    opt_name = getattr(cfg, 'optimizer', 'Adam')

    params = []
    head_param_names = [
        'global_proj', 'global_bn', 'global_classifier',
        'part_gat', 'local_classifier', 'domain_classifier',
        'domain_factors', 'bottleneck', 'classifier'
    ]

    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue

        is_head = any(name in key for name in head_param_names)
        lr = head_lr if is_head else base_lr

        # Exclude normalization and bias terms from weight decay
        wd = 0.0 if ('bn' in key or 'bias' in key or 'ln' in key) else weight_decay

        if 'bias' in key:
            lr = lr * 2.0

        params.append({
            'params': [value],
            'lr': lr,
            'weight_decay': wd,
            'param_name': key
        })

    if opt_name == 'SGD':
        optimizer = torch.optim.SGD(params, momentum=0.9, nesterov=True)
    elif opt_name == 'AdamW':
        optimizer = torch.optim.AdamW(params, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.Adam(params, weight_decay=weight_decay)

    return optimizer


def make_optimizer(args, model):
    return build_optimizer(model, args)
