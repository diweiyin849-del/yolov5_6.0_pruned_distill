# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Feature Distillation Training Script
基于官方 yolo26s-visdrone-4output-map0.398.py 改写，只支持特征蒸馏 (Feature-based Distillation)

使用方法:
    python train_featdistill.py --weights student.pt --t_weights teacher.pt --data dataset.yaml --distill-layers model.17.cv2 model.20.cv2 model.23.cv2 --distill-feat-weight 0.05
"""

import argparse
import math
import os
import random
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import yaml
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import SGD, AdamW, lr_scheduler
from tqdm import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

import val
from models.experimental import attempt_load
from models.yolo import Model
from models.featdistill import FeatureDistillManager, FeatureDistillLoss
from utils.loss import ComputeLoss
from utils.autoanchor import check_anchors
from utils.autobatch import check_train_batch_size
from utils.callbacks import Callbacks
from utils.datasets import create_dataloader
from utils.downloads import attempt_download
from utils.general import (LOGGER, check_dataset, check_file, check_img_size, check_suffix, check_yaml,
                           colorstr, get_latest_run, increment_path, init_seeds, intersect_dicts,
                           labels_to_class_weights, labels_to_image_weights, methods, one_cycle,
                           print_args, print_mutation, strip_optimizer)
from utils.loggers import Loggers
from utils.loggers.wandb.wandb_utils import check_wandb_resume
from utils.metrics import fitness
from utils.plots import plot_evolve, plot_labels
from utils.torch_utils import EarlyStopping, ModelEMA, de_parallel, select_device, torch_distributed_zero_first

LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))


def patch_upsample(model):
    for m in model.modules():
        if isinstance(m, nn.Upsample) and not hasattr(m, 'recompute_scale_factor'):
            m.recompute_scale_factor = True


def parse_opt():
    parser = argparse.ArgumentParser()
    # ===== 基础参数 =====
    parser.add_argument('--weights', type=str, default='yolov5s.pt', help='student initial weights path')
    parser.add_argument('--cfg', type=str, default='', help='model.yaml path')
    parser.add_argument('--data', type=str, default='data/coco128.yaml', help='dataset.yaml path')
    parser.add_argument('--hyp', type=str, default='data/hyps/hyp.scratch-low.yaml', help='hyperparameters path')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=16, help='total batch size for all GPUs')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='yolo26s-visdrone-4output-map0.398, val image size')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--noval', action='store_true', help='only validate final epoch')
    parser.add_argument('--noautoanchor', action='store_true', help='disable autoanchor check')
    parser.add_argument('--evolve', type=int, nargs='?', const=300, help='evolve hyperparameters for x generations')
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--cache', type=str, nargs='?', const='ram', help='--cache images in "ram" or "disk"')
    parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
    parser.add_argument('--single-cls', action='store_true', help='yolo26s-visdrone-4output-map0.398 multi-class data as single-class')
    parser.add_argument('--adam', action='store_true', help='use torch.optim.Adam() optimizer')
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--workers', type=int, default=8, help='max dataloader workers')
    parser.add_argument('--project', default=ROOT / 'runs/yolo26s-visdrone-4output-map0.398', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--quad', action='store_true', help='quad dataloader')
    parser.add_argument('--linear-lr', action='store_true', help='linear LR')
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing epsilon')
    parser.add_argument('--patience', type=int, default=100, help='EarlyStopping patience')
    parser.add_argument('--freeze', nargs='+', type=int, default=[0], help='Freeze layers')
    parser.add_argument('--save-period', type=int, default=-1, help='Save checkpoint every x epochs')
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter')
    parser.add_argument('--entity', default=None, help='W&B: Entity')
    parser.add_argument('--upload_dataset', nargs='?', const=True, default=False, help='W&B: Upload data')
    parser.add_argument('--bbox_interval', type=int, default=-1, help='W&B: Set bounding-box image logging interval')
    parser.add_argument('--artifact_alias', type=str, default='latest', help='W&B: Version of dataset artifact')

    # ===== 特征蒸馏参数（只用这个） =====
    parser.add_argument('--t_weights', type=str, required=True, help='teacher model weights path')
    parser.add_argument('--distill-layers', nargs='+',
                        default=['model.17.cv3', 'model.20.cv3', 'model.23.cv3'],
                        help='layers for feature distillation')
    parser.add_argument('--distill-feat-weight', type=float, default=0.5,
                        help='feature distillation loss weight')
    parser.add_argument('--distill-feat-loss', type=str, default='mse',
                        choices=['mse', 'l1', 'smooth_l1', 'cosine'],
                        help='feature distillation loss type')

    return parser.parse_args()


def train(hyp, opt, device, callbacks):
    save_dir, epochs, batch_size, weights, single_cls, evolve, data, cfg, resume, noval, nosave, workers, freeze = \
        Path(opt.save_dir), opt.epochs, opt.batch_size, opt.weights, opt.single_cls, opt.evolve, opt.data, opt.cfg, \
        opt.resume, opt.noval, opt.nosave, opt.workers, opt.freeze

    w = save_dir / 'weights'
    (w.parent if evolve else w).mkdir(parents=True, exist_ok=True)
    last, best = w / 'last.pt', w / 'best.pt'

    if isinstance(hyp, str):
        with open(hyp, errors='ignore') as f:
            hyp = yaml.safe_load(f)
    LOGGER.info(colorstr('hyperparameters: ') + ', '.join(f'{k}={v}' for k, v in hyp.items()))

    if not evolve:
        with open(save_dir / 'hyp.yaml', 'w') as f:
            yaml.safe_dump(hyp, f, sort_keys=False)
        with open(save_dir / 'opt.yaml', 'w') as f:
            yaml.safe_dump(vars(opt), f, sort_keys=False)

    # Loggers
    data_dict = None
    if RANK in [-1, 0]:
        loggers = Loggers(save_dir, weights, opt, hyp, LOGGER)
        if loggers.wandb:
            data_dict = loggers.wandb.data_dict
            if resume:
                weights, epochs, hyp = opt.weights, opt.epochs, opt.hyp
        for k in methods(loggers):
            callbacks.register_action(k, callback=getattr(loggers, k))

    plots = not evolve
    cuda = device.type != 'cpu'
    init_seeds(1 + RANK)
    with torch_distributed_zero_first(LOCAL_RANK):
        data_dict = data_dict or check_dataset(data)
    train_path, val_path = data_dict['yolo26s-visdrone-4output-map0.398'], data_dict['val']
    nc = 1 if single_cls else int(data_dict['nc'])
    names = ['item'] if single_cls and len(data_dict['names']) != 1 else data_dict['names']
    assert len(names) == nc, f'{len(names)} names found for nc={nc} dataset in {data}'
    is_coco = isinstance(val_path, str) and val_path.endswith('coco/val2017.txt')

    # ===== 加载学生模型 =====
    check_suffix(weights, '.pt')
    pretrained = weights.endswith('.pt')
    if pretrained:
        with torch_distributed_zero_first(LOCAL_RANK):
            weights = attempt_download(weights)
        ckpt = torch.load(weights, map_location=device, weights_only=False)
        if 'model' in ckpt:
            model = ckpt['model'].float().to(device)
            LOGGER.info(f'Loaded student model from {weights}')
        else:
            model = ckpt.float().to(device)
        model.nc = nc
        model.names = names
        model.hyp = hyp
        patch_upsample(model)
    else:
        model = Model(cfg, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)

    # ===== 加载教师模型 =====
    LOGGER.info(f'Loading teacher model from {opt.t_weights}')
    t_ckpt = torch.load(opt.t_weights, map_location=device, weights_only=False)
    teacher_model = t_ckpt.get("model", t_ckpt).float().to(device)
    teacher_model.eval()
    for p in teacher_model.parameters():
        p.requires_grad = False
    patch_upsample(teacher_model)
    LOGGER.info(f'Teacher model loaded: {sum(p.numel() for p in teacher_model.parameters()):,} parameters')

    # ===== 特征蒸馏初始化 =====
    adapter_channels = []
    for layer_name in opt.distill_layers:
        try:
            t_layer = dict(teacher_model.named_modules())[layer_name]
            s_layer = dict(model.named_modules())[layer_name]
            if hasattr(t_layer, 'conv') and hasattr(s_layer, 'conv'):
                t_ch = t_layer.conv.out_channels
                s_ch = s_layer.conv.out_channels
            elif hasattr(t_layer, 'cv2') and hasattr(s_layer, 'cv2'):
                t_ch = t_layer.cv2.conv.out_channels
                s_ch = s_layer.cv2.conv.out_channels
            else:
                t_ch = getattr(t_layer, 'out_channels', 256)
                s_ch = getattr(s_layer, 'out_channels', 128)
            adapter_channels.append((s_ch, t_ch))
        except Exception as e:
            LOGGER.warning(f'Cannot infer channels for {layer_name}: {e}, using default')
            adapter_channels.append((128, 256))
    LOGGER.info(f'Feature distill adapter channels: {adapter_channels}')

    distill_manager = FeatureDistillManager(
        teacher_model=teacher_model,
        student_model=model,
        distill_layers=opt.distill_layers,
        adapter_channels=adapter_channels
    )
    distill_manager = distill_manager.to(device)
    distill_loss_fn = FeatureDistillLoss(
        loss_type=opt.distill_feat_loss,
        normalize=False,
        temperature=1.0
    )
    LOGGER.info(f'Feature distillation enabled. Layers: {opt.distill_layers}')

    # Freeze
    freeze = [f'model.{x}.' for x in (freeze if len(freeze) > 1 else range(freeze[0]))]
    for k, v in model.named_parameters():
        v.requires_grad = True
        if any(x in k for x in freeze):
            LOGGER.info(f'freezing {k}')
            v.requires_grad = False

    # Image size
    gs = max(int(model.stride.max()), 32)
    imgsz = check_img_size(opt.imgsz, gs, floor=gs * 2)

    # Batch size
    if RANK == -1 and batch_size == -1:
        batch_size = check_train_batch_size(model, imgsz)
        loggers.on_params_update({"batch_size": batch_size})

    # Optimizer
    nbs = 64
    accumulate = max(round(nbs / batch_size), 1)
    hyp['weight_decay'] *= batch_size * accumulate / nbs
    LOGGER.info(f"Scaled weight_decay = {hyp['weight_decay']}")

    g0, g1, g2 = [], [], []
    for v in model.modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
            g2.append(v.bias)
        if isinstance(v, nn.BatchNorm2d):
            g0.append(v.weight)
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
            g1.append(v.weight)

    if opt.adam:
        optimizer = AdamW(g0, lr=hyp['lr0'])
    else:
        optimizer = SGD(g0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)
    optimizer.add_param_group({'params': g1, 'weight_decay': hyp['weight_decay']})
    optimizer.add_param_group({'params': g2})

    if opt.linear_lr:
        lf = lambda x: (1 - x / (epochs - 1)) * (1.0 - hyp['lrf']) + hyp['lrf']
    else:
        lf = one_cycle(1, hyp['lrf'], epochs)
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    ema = ModelEMA(model) if RANK in [-1, 0] else None

    start_epoch, best_fitness = 0, 0.0
    if pretrained and 'optimizer' in ckpt and ckpt['optimizer'] is not None:
        try:
            optimizer.load_state_dict(ckpt['optimizer'])
            best_fitness = ckpt.get('best_fitness', 0.0)
            start_epoch = ckpt.get('epoch', -1) + 1
        except:
            LOGGER.warning("Could not load optimizer state")

    if cuda and RANK == -1 and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    if opt.sync_bn and cuda and RANK != -1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)

    # Dataloader
    train_loader, dataset = create_dataloader(
        train_path, imgsz, batch_size // WORLD_SIZE, gs, single_cls,
        hyp=hyp, augment=True, cache=opt.cache, rect=opt.rect, rank=LOCAL_RANK,
        workers=workers, image_weights=opt.image_weights, quad=opt.quad,
        prefix=colorstr('yolo26s-visdrone-4output-map0.398: '), shuffle=True
    )
    mlc = int(np.concatenate(dataset.labels, 0)[:, 0].max())
    nb = len(train_loader)
    assert mlc < nc, f'Label class {mlc} exceeds nc={nc}'

    if RANK in [-1, 0]:
        val_loader = create_dataloader(
            val_path, imgsz, batch_size // WORLD_SIZE * 2, gs, single_cls,
            hyp=hyp, cache=None if noval else opt.cache, rect=True, rank=-1,
            workers=workers, pad=0.5, prefix=colorstr('val: ')
        )[0]
        if not resume:
            labels = np.concatenate(dataset.labels, 0)
            if plots:
                plot_labels(labels, names, save_dir)
            if not opt.noautoanchor:
                try:
                    check_anchors(dataset, model=model, thr=hyp['anchor_t'], imgsz=imgsz)
                except:
                    LOGGER.warning("Anchor check failed, skipping")
            model.float()
        callbacks.run('on_pretrain_routine_end')

    if cuda and RANK != -1:
        model = DDP(model, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK)

    # Model attributes
    nl = de_parallel(model).model[-1].nl
    hyp['box'] *= 3 / nl
    hyp['cls'] *= nc / 80 * 3 / nl
    hyp['obj'] *= (imgsz / 640) ** 2 * 3 / nl
    hyp['label_smoothing'] = opt.label_smoothing
    model.nc = nc
    model.hyp = hyp
    model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc
    model.names = names

    # Start training
    t0 = time.time()
    nw = max(round(hyp['warmup_epochs'] * nb), 1000)
    last_opt_step = -1
    maps = np.zeros(nc)
    results = (0, 0, 0, 0, 0, 0, 0)
    scheduler.last_epoch = start_epoch - 1
    scaler = amp.GradScaler(enabled=True)
    stopper = EarlyStopping(patience=opt.patience)
    compute_loss = ComputeLoss(model)
    LOGGER.info(f'Starting training for {epochs} epochs...')

    for epoch in range(start_epoch, epochs):
        model.train()
        if opt.image_weights:
            cw = model.class_weights.cpu().numpy() * (1 - maps) ** 2 / nc
            iw = labels_to_image_weights(dataset.labels, nc=nc, class_weights=cw)
            dataset.indices = random.choices(range(dataset.n), weights=iw, k=dataset.n)

        mloss = torch.zeros(3, device=device)
        if RANK != -1:
            train_loader.sampler.set_epoch(epoch)
        pbar = enumerate(train_loader)
        LOGGER.info(('\n' + '%10s' * 7) % ('Epoch', 'gpu_mem', 'box', 'obj', 'cls', 'labels', 'img_size'))
        if RANK in [-1, 0]:
            pbar = tqdm(pbar, total=nb, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
        optimizer.zero_grad()

        for i, (imgs, targets, paths, _) in pbar:
            ni = i + nb * epoch
            imgs = imgs.to(device, non_blocking=True).float() / 255

            if ni <= nw:
                xi = [0, nw]
                accumulate = max(1, np.interp(ni, xi, [1, nbs / batch_size]).round())
                for j, x in enumerate(optimizer.param_groups):
                    x['lr'] = np.interp(ni, xi, [hyp['warmup_bias_lr'] if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, xi, [hyp['warmup_momentum'], hyp['momentum']])

            if opt.multi_scale:
                sz = random.randrange(imgsz * 0.5, imgsz * 1.5 + gs) // gs * gs
                sf = sz / max(imgs.shape[2:])
                if sf != 1:
                    ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]
                    imgs = nn.functional.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)

            with torch.amp.autocast('cuda', enabled=True):
                pred = model(imgs)
                loss, loss_items = compute_loss(pred, targets.to(device))
                total_loss = loss

                # ===== 特征蒸馏（只有这个） =====
                with torch.no_grad():
                    _ = teacher_model(imgs)
                feat_loss = distill_manager.get_distill_loss(distill_loss_fn)
                dist_feat_weight = hyp.get('dist_feat', 0.5)
                total_loss = total_loss + dist_feat_weight * feat_loss

                if RANK != -1:
                    total_loss *= WORLD_SIZE
                if opt.quad:
                    total_loss *= 4.

            scaler.scale(total_loss).backward()

            if ni - last_opt_step >= accumulate:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                if ema:
                    ema.update(model)
                last_opt_step = ni

            if RANK in [-1, 0]:
                mloss = (mloss * i + loss_items) / (i + 1)
                mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'
                pbar.set_description(('%10s' * 2 + '%10.4g' * 5) % (
                    f'{epoch}/{epochs - 1}', mem, *mloss, targets.shape[0], imgs.shape[-1]))
                callbacks.run('on_train_batch_end', ni, model, imgs, targets, paths, plots, opt.sync_bn)

        lr = [x['lr'] for x in optimizer.param_groups]
        scheduler.step()

        if RANK in [-1, 0]:
            callbacks.run('on_train_epoch_end', epoch=epoch)
            ema.update_attr(model, include=['yaml', 'nc', 'hyp', 'names', 'stride', 'class_weights'])
            final_epoch = (epoch + 1 == epochs) or stopper.possible_stop
            if not noval or final_epoch:
                results, maps, _ = val.run(
                    data_dict,
                    batch_size=batch_size // WORLD_SIZE * 2,
                    imgsz=imgsz,
                    model=ema.ema,
                    single_cls=single_cls,
                    dataloader=val_loader,
                    save_dir=save_dir,
                    plots=False,
                    callbacks=callbacks,
                    compute_loss=compute_loss
                )

            fi = fitness(np.array(results).reshape(1, -1))
            if fi > best_fitness:
                best_fitness = fi
            log_vals = list(mloss) + list(results) + lr
            callbacks.run('on_fit_epoch_end', log_vals, epoch, best_fitness, fi)

            if (not nosave) or (final_epoch and not evolve):
                # ✅ 只保存 state_dict，彻底绕过 hook 序列化问题
                model_to_save = de_parallel(model)

                # 保存权重
                torch.save(model_to_save.state_dict(), last)
                if best_fitness == fi:
                    torch.save(model_to_save.state_dict(), best)
                if (epoch > 0) and (opt.save_period > 0) and (epoch % opt.save_period == 0):
                    torch.save(model_to_save.state_dict(), w / f'epoch{epoch}.pt')

                callbacks.run('on_model_save', last, epoch, final_epoch, best_fitness, fi)

            if RANK == -1 and stopper(epoch=epoch, fitness=fi):
                break

    if RANK in [-1, 0]:
        LOGGER.info(f'\n{epoch - start_epoch + 1} epochs completed in {(time.time() - t0) / 3600:.3f} hours.')
        LOGGER.info(f"Best model saved at {best}")
        # 不进行最终验证，训练过程中每10轮已验证并保存最佳模型
        callbacks.run('on_train_end', last, best, plots, epoch, results)
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}")

    torch.cuda.empty_cache()
    return results


def main(opt, callbacks=Callbacks()):
    if RANK in [-1, 0]:
        print_args(FILE.stem, opt)

    if opt.resume and not check_wandb_resume(opt) and not opt.evolve:
        ckpt = opt.resume if isinstance(opt.resume, str) else get_latest_run()
        assert os.path.isfile(ckpt), 'ERROR: --resume checkpoint does not exist'
        with open(Path(ckpt).parent.parent / 'opt.yaml', errors='ignore') as f:
            opt = argparse.Namespace(**yaml.safe_load(f))
        opt.cfg, opt.weights, opt.resume = '', ckpt, True
        LOGGER.info(f'Resuming training from {ckpt}')
    else:
        opt.data, opt.cfg, opt.hyp, opt.weights, opt.project = \
            check_file(opt.data), check_yaml(opt.cfg), check_yaml(opt.hyp), str(opt.weights), str(opt.project)
        assert len(opt.cfg) or len(opt.weights), 'either --cfg or --weights must be specified'
        if opt.evolve:
            opt.project = str(ROOT / 'runs/evolve')
            opt.exist_ok, opt.resume = opt.resume, False
        opt.save_dir = str(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))

    device = select_device(opt.device, batch_size=opt.batch_size)
    if LOCAL_RANK != -1:
        assert torch.cuda.device_count() > LOCAL_RANK, 'insufficient CUDA devices for DDP command'
        assert opt.batch_size % WORLD_SIZE == 0, '--batch-size must be multiple of CUDA device count'
        assert not opt.image_weights, '--image-weights argument is not compatible with DDP training'
        assert not opt.evolve, '--evolve argument is not compatible with DDP training'
        torch.cuda.set_device(LOCAL_RANK)
        device = torch.device('cuda', LOCAL_RANK)
        dist.init_process_group(backend="nccl" if dist.is_nccl_available() else "gloo")

    if not opt.evolve:
        train(opt.hyp, opt, device, callbacks)
        if WORLD_SIZE > 1 and RANK == 0:
            LOGGER.info('Destroying process group... ')
            dist.destroy_process_group()
    else:
        pass


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)