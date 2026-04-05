import __init__
import os, argparse, yaml, numpy as np, torch
from torch import multiprocessing as mp
from examples.classification.train import main as train
from examples.classification.pretrain import main as pretrain
from openpoints.utils import EasyConfig, dist_utils, find_free_port, generate_exp_directory, resume_exp_directory, parse_config_path


def _normalize_runtime_cfg(cfg):
    runtime_cfg = cfg.get('runtime', EasyConfig())
    if not isinstance(runtime_cfg, EasyConfig):
        normalized = EasyConfig()
        normalized.update(runtime_cfg)
        runtime_cfg = normalized
    runtime_cfg.device = str(runtime_cfg.get('device', 'auto')).lower()
    runtime_cfg.gpu_id = int(runtime_cfg.get('gpu_id', 0))
    if runtime_cfg.device not in {'auto', 'cpu', 'gpu', 'cuda'}:
        raise ValueError(
            f"Unsupported runtime.device={runtime_cfg.device!r}. Use one of: auto, cpu, gpu."
        )
    cfg.runtime = runtime_cfg
    return runtime_cfg


def resolve_runtime_device(cfg):
    runtime_cfg = _normalize_runtime_cfg(cfg)
    requested_device = runtime_cfg.device

    if requested_device == 'cpu':
        return torch.device('cpu')

    if requested_device in {'gpu', 'cuda'} and not torch.cuda.is_available():
        raise RuntimeError(
            'runtime.device is set to GPU, but torch.cuda.is_available() is False. '
            'Switch runtime.device to cpu or run in an environment with visible NVIDIA devices.'
        )

    if requested_device in {'gpu', 'cuda', 'auto'} and torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        if runtime_cfg.gpu_id < 0 or runtime_cfg.gpu_id >= gpu_count:
            raise RuntimeError(
                f'runtime.gpu_id={runtime_cfg.gpu_id} is out of range for {gpu_count} visible GPU(s).'
            )
        return torch.device(f'cuda:{runtime_cfg.gpu_id}')

    return torch.device('cpu')


if __name__ == "__main__":
    parser = argparse.ArgumentParser('S3DIS scene segmentation training')
    parser.add_argument('--cfg', type=str, required=True, help='config file')
    parser.add_argument('--profile', action='store_true', default=False, help='set to True to profile speed')
    args, opts = parser.parse_known_args()
    cfg = EasyConfig()
    cfg.load(args.cfg, recursive=True)
    cfg.update(opts)
    cfg.resolve_references()
    resolved_device = resolve_runtime_device(cfg)
    cfg.device = str(resolved_device)
    cfg.device_type = resolved_device.type
    cfg.device_id = 0 if resolved_device.index is None else resolved_device.index
    cfg.use_gpu = cfg.device_type == 'cuda'
    if not cfg.use_gpu:
        cfg.dist_backend = 'gloo'
    model_name = str(cfg.get('model', {}).get('NAME', ''))
    if model_name in {'BaseSeg', 'BasePartSeg'}:
        raise ValueError(
            f'Config `{args.cfg}` defines a segmentation model ({model_name}). '
            'Use `examples/segmentation/main.py` for semantic segmentation or '
            '`examples/shapenetpart/main.py` for part segmentation.'
        )
    if cfg.seed is None:
        cfg.seed = np.random.randint(1, 10000)

    # init distributed env first, since logger depends on the dist info.
    cfg.rank, cfg.world_size, cfg.distributed, cfg.mp = dist_utils.get_dist_info(cfg)
    cfg.sync_bn = cfg.use_gpu and cfg.world_size > 1

    # init log dir
    cfg.task_name, cfg.exp_name = parse_config_path(args.cfg)
    tags = [
        cfg.task_name,  # task name (the folder of name under ./cfgs
        cfg.mode,
        cfg.exp_name,  # cfg file name
        f'ngpus{cfg.world_size}',
        f'seed{cfg.seed}',
    ]
    opt_list = [] # for checking experiment configs from logging file
    for i, opt in enumerate(opts):
        if 'rank' not in opt and 'dir' not in opt and 'root' not in opt and 'pretrain' not in opt and 'path' not in opt and '/' not in opt:
            opt_list.append(opt)
    cfg.root_dir = os.path.join(cfg.root_dir, cfg.task_name)
    cfg.opts = '-'.join(opt_list)

    if cfg.mode in ['resume', 'val', 'test']:
        resume_exp_directory(cfg, pretrained_path=cfg.pretrained_path)
    else:  # resume from the existing ckpt and reuse the folder.
        generate_exp_directory(cfg, tags, additional_id=os.environ.get('MASTER_PORT', None))
    os.environ["JOB_LOG_DIR"] = cfg.log_dir
    cfg_path = os.path.join(cfg.run_dir, "cfg.yaml")
    with open(cfg_path, 'w') as f:
        yaml.dump(cfg, f, indent=2)
        os.system('cp %s %s' % (args.cfg, cfg.run_dir))
    cfg.cfg_path = cfg_path

    if cfg.mode == 'pretrain':
        main = pretrain
    else:
        main = train

    # multi processing.
    if cfg.mp:
        port = find_free_port()
        cfg.dist_url = f"tcp://localhost:{port}"
        print('using mp spawn for distributed training')
        mp.spawn(main, nprocs=cfg.world_size, args=(cfg, args.profile))
    else:
        main(0, cfg, profile=args.profile)
