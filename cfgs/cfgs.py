"""
Config module for CLIP-FGDI reproduction.
Supports:
  - Protocol 2: leave-one-out across {Market, MSMT17, cuhk_sysu, cuhk03}
  - Occluded-Duke: all-4-source -> Occluded-DukeMTMC zero-shot test
  - Debug mode: subsample PIDs for local smoke testing
"""

# Dataset name -> number of training identities
DATASET_CLASSES = {
    "Market": 1502,
    "MSMT17": 4101,
    "cuhk_sysu": 11934,
    "cuhk03": 1467,
}

ALL_SOURCE_DATASETS = ["Market", "MSMT17", "cuhk_sysu", "cuhk03"]


def parse_common_args(parser):
    parser.add_argument('--model', type=str, default='clip')
    parser.add_argument('--backbone', type=str, default='ViT-B-16')
    parser.add_argument('--device_id', type=str, default='0')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device: auto | cuda | mps | cpu')
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--num_instance', type=int, default=4)
    parser.add_argument('--data_path', type=str, default='data', help="dataset root path")
    parser.add_argument('--log_path', type=str, default='logs', help="log path")
    return parser


def parse_train_args(parser, datasets=None, classes=None, combine_all=False):
    if classes is None:
        classes = [1502, 11934, 1816, 1467]
    if datasets is None:
        datasets = ["Market", "cuhk_sysu", "CUHK02", "cuhk03"]
    parser = parse_common_args(parser)
    parser.add_argument('--epochs', type=int, default=60, help='total training epochs')
    parser.add_argument('--stage1_epochs', type=int, default=10, help='Stage 1 frozen-backbone epochs')
    parser.add_argument('--stage2_epochs', type=int, default=30, help='Stage 2 partial unfreezing epochs')
    parser.add_argument('--prompt-epoch', type=int, default=120, help='prompt epoch')
    parser.add_argument('--prompt-domain-epoch', type=int, default=30, help='prompt domain epoch')
    parser.add_argument('--prior-epoch', type=int, default=3, help='prior epoch')
    parser.add_argument('--image-encoder-epoch', type=int, default=60, help='image encoder epoch')
    parser.add_argument('--prompt-lr', default=0.00035, type=float, help='head learning rate')
    parser.add_argument('--head-lr', default=0.00035, type=float, help='head learning rate')
    parser.add_argument('--image-encoder-lr', default=0.000005, type=float, help='backbone learning rate')
    parser.add_argument('--beta', default=0.1, type=float, help='weight for apn-loss')
    parser.add_argument('--omega', default=0.01, type=float, help='weight for domain loss')
    parser.add_argument('--epsilon', default=0.1, type=float, help='weights for gradient backpropagation')
    parser.add_argument('--lamda', default=0.01, type=float, help='learning rate for domain classifier')
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    # Exact BTP report loss weights (Table 2)
    parser.add_argument('--lambda_id', default=1.0, type=float, help='global id loss weight')
    parser.add_argument('--lambda_tri', default=1.0, type=float, help='global triplet loss weight')
    parser.add_argument('--lambda_id_local', default=0.5, type=float, help='local id loss weight')
    parser.add_argument('--lambda_tri_local', default=0.5, type=float, help='local triplet loss weight')
    parser.add_argument('--lambda_occ', default=0.3, type=float, help='visibility loss weight')
    parser.add_argument('--lambda_adv', default=0.1, type=float, help='GRL domain loss weight')
    parser.add_argument('--lambda_ortho', default=0.05, type=float, help='orthogonal domain factor loss weight')
    # Exact resolution: 256x128
    parser.add_argument('--size_train', default=[256, 128], type=list)
    parser.add_argument('--gray_scale', default=0.1, type=float)
    parser.add_argument('--pad', default=10, type=int)
    parser.add_argument('--random_horizontal_flip', default=0.5, type=float)
    parser.add_argument('--color_jitter', default=0.1, type=float)
    parser.add_argument('--aug_mix', default=0.6, type=float)
    parser.add_argument('--random_erasing', default=0.0, type=float)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--checkpoint_period', type=int, default=10, help='Save checkpoint every N epochs')
    parser.add_argument('--resume', type=str, default='', help='Path to checkpoint .pth file to resume training')
    parser.add_argument('--eval_only', action='store_true', help='Only evaluate on testloaders and exit')
    parser.add_argument('--log_period', type=int, default=100)
    parser.add_argument('--eval_period', type=int, default=10)
    parser.add_argument('--train_datasets', type=list, default=datasets)
    parser.add_argument('--classes', type=list, default=classes)
    parser.add_argument('--sampling_method', type=str, default='PK')
    parser.add_argument('--combine_all', type=bool, default=combine_all)
    # --- New Protocol-2 / Occluded-Duke / Ablations args ---
    parser.add_argument('--benchmark', type=str, default='protocol2',
                        choices=['protocol1', 'protocol2', 'occluded_duke'],
                        help='Benchmark: protocol1 | protocol2 | occluded_duke')
    parser.add_argument('--held_out_domain', type=str, default='Market',
                        choices=['Market', 'MSMT17', 'cuhk_sysu', 'cuhk03'],
                        help='For protocol2: which domain to hold out as target')
    parser.add_argument('--debug_subset_ids', type=int, default=0,
                        help='If >0, subsample this many PIDs per source dataset for debug')
    parser.add_argument('--disable_grl', action='store_true', help='Disable Gradient Reversal Layer')
    parser.add_argument('--disable_part_branch', action='store_true', help='Disable PartVisibilityGAT branch')
    parser.add_argument('--disable_vis_gating', action='store_true', help='Disable visibility gating in PartVisibilityGAT')
    parser.add_argument('--eval_feat', type=str, default='combined', choices=['combined', 'global', 'local'],
                        help='Evaluation feature type: combined | global | local')
    parser.add_argument('--occlusion_severity', type=str, default='clean',
                        choices=['clean', 'mild', 'moderate', 'severe'],
                        help='Target query occlusion severity: clean | mild | moderate | severe')
    parser.add_argument('--no_sysu', action='store_true', help='Exclude cuhk_sysu from training datasets')
    return parser


def parse_test_args(parser, datasets=None):
    if datasets is None:
        datasets = ["PRID", "GRID", "VIPeR", "iLIDS"]
    parser = parse_common_args(parser)
    parser.add_argument('--size_test', default=[256, 128], type=list)
    parser.add_argument('--test_datasets', type=list, default=datasets)
    parser.add_argument('--test_batch_size', type=int, default=256)
    parser.add_argument('--eval_feat', type=str, default='combined', choices=['combined', 'global', 'local'])
    parser.add_argument('--occlusion_severity', type=str, default='clean',
                        choices=['clean', 'mild', 'moderate', 'severe'])
    return parser


# ============================================================
# Legacy Protocol 1 (original CLIP-FGDI paper, kept for compat)
# ============================================================

def protocol_1(parser, parsertest):
    return parse_train_args(parser, ["Market", "cuhk_sysu", "CUHK02", "cuhk03"], [1502, 11934, 1816, 1467], True), \
           parse_test_args(parsertest, ["PRID", "GRID", "VIPeR", "iLIDS"]), \
           'protocol_1'


# ============================================================
# Protocol 2: leave-one-out (config-driven, no code change)
# ============================================================

def protocol_2(parser, parsertest, held_out_domain="Market", no_sysu=False):
    """
    Leave-one-out split: train on 3 source datasets, test on held_out_domain.
    The held_out_domain is specified via --held_out_domain arg.
    """
    source_datasets = [d for d in ALL_SOURCE_DATASETS if d != held_out_domain]
    if no_sysu and "cuhk_sysu" in source_datasets:
        source_datasets.remove("cuhk_sysu")
    source_classes = [DATASET_CLASSES[d] for d in source_datasets]
    test_datasets = [held_out_domain]

    return parse_train_args(parser, source_datasets, source_classes, False), \
           parse_test_args(parsertest, test_datasets), \
           f'protocol_2_{held_out_domain}'


# ============================================================
# Occluded-Duke: all-4-source -> OccludedDuke zero-shot test
# ============================================================

def protocol_occluded_duke(parser, parsertest):
    """
    Train on all 4 source datasets, test zero-shot on Occluded-DukeMTMC.
    """
    source_datasets = list(ALL_SOURCE_DATASETS)
    source_classes = [DATASET_CLASSES[d] for d in source_datasets]
    test_datasets = ["OccludedDuke"]

    return parse_train_args(parser, source_datasets, source_classes, False), \
           parse_test_args(parsertest, test_datasets), \
           'protocol_occluded_duke'


# ============================================================
# Legacy Protocol 2 variants (kept for backward compatibility)
# ============================================================

def protocol_2_C3(parser, parsertest):
    return parse_train_args(parser, ["Market", "MSMT17", "cuhk_sysu"], [1502, 4101, 11934], False), \
           parse_test_args(parsertest, ["cuhk03"]), \
           'protocol_2_C3'


def protocol_3_C3(parser, parsertest):
    return parse_train_args(parser, ["Market", "MSMT17", "cuhk_sysu"], [1502, 4101, 11934], True), \
           parse_test_args(parsertest, ["cuhk03"]), \
           'protocol_2_C3'


def protocol_2_MS(parser, parsertest):
    return parse_train_args(parser, ["cuhk03", "cuhk_sysu", "Market"], [1467, 11934, 1502], False), \
           parse_test_args(parsertest, ["MSMT17"]), \
           'protocol_2_MS'


def protocol_3_MS(parser, parsertest):
    return parse_train_args(parser, ["cuhk03", "cuhk_sysu", "Market"], [1467, 11934, 1502], True), \
           parse_test_args(parsertest, ["MSMT17"]), \
           'protocol_3_MS'


def protocol_2_M(parser, parsertest):
    return parse_train_args(parser, ["cuhk03", "cuhk_sysu", "MSMT17"], [1467, 11934, 4101], False), \
           parse_test_args(parsertest, ["Market"]), \
           'protocol_2_M'


def protocol_3_M(parser, parsertest):
    return parse_train_args(parser, ["cuhk03", "cuhk_sysu", "MSMT17"], [1467, 11934, 4101], True), \
           parse_test_args(parsertest, ["Market"]), \
           'protocol_3_M'
