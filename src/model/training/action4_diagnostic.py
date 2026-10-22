"""GPU fixed-ROI component diagnostic adapted from the preserved R3 CPU helper.

No formal gate is produced. Model, sampling, loss and decoding reuse the existing
implementation. FP32 preserves the historical helper arithmetic; AMP is checked
separately before formal training.
"""
from __future__ import annotations
import copy,json,math,resource,sys,time,os
from pathlib import Path
from data.action4_paths import checked_root, contained_path


def diagnose(args):
    if args.roi_count != 32 or not 1 <= args.updates <= 500:
        raise ValueError("This protocol requires 32 distinct valid ROIs and at most 500 updates")
    if args.max_seconds <= 0 or args.max_memory_gib <= 0:
        raise ValueError("Positive resource limits required")
    root = checked_root(args.root)
    import numpy as np
    import torch
    from PIL import Image, ImageDraw, ImageFont
    from data.action4_adapter import DataAdapter
    from data.action4_dataset import Action4Dataset, is_ignored, prepare_rois, build_corner_targets
    from model.action4_network import Action4Net
    from model.action4_decode import decode_corners
    from model.action4_losses import corner_heatmap_loss
    from model.training.action4_trainer import build_optimizer, learning_rate_factor
    from model.training.action4_runtime import (read_config, run_directories, seed_all, sha256_file,
                                               write_json_new, save_checkpoint_atomic, capture_rng,
                                               environment_identity, input_path, select_device)
    torch.set_num_threads(2)
    seed = 20261018
    seed_all(seed, deterministic=True)
    config_path = Path(args.config)
    config_path = config_path if config_path.is_absolute() else root / config_path
    base_config = read_config(config_path)
    if ({k:base_config['network'].get(k) for k in ('decoder_channels','head_channels','freeze_bn')} != {'decoder_channels':256,'head_channels':64,'freeze_bn':True}
            or set(base_config['network']) - {'decoder_channels','head_channels','freeze_bn','corner_probability_bias'}):
        raise ValueError('Expected the actual formal ResNet50/256 decoder configuration')
    if base_config['preprocess']['roi_size'] != 256:
        raise ValueError('The diagnostic preserves formal256-pixel ROIs')
    manifest = Path(args.manifest)
    manifest = manifest if manifest.is_absolute() else root / manifest
    dataset = Action4Dataset(root, manifest, allow_diagnostic=True)
    eligible = [(i,j) for i,row in enumerate(dataset.records)
                for j,instance in enumerate(row['instances']) if not is_ignored(instance)]
    if len(eligible) < args.roi_count:
        raise ValueError(f"Need 32 distinct valid ROIs; found {len(eligible)}")
    positions = np.linspace(0, len(eligible)-1, args.roi_count).round().astype(int).tolist()
    assert len(set(positions)) == args.roi_count
    items, instance_indices, selected = [], [], []
    for position in positions:
        row_index, instance_index = eligible[position]
        item = dataset[row_index]
        items.append(item)
        instance_indices.append(instance_index)
        selected.append({'sample_id':item['record']['sample_id'],
                         'instance_id':item['record']['instances'][instance_index]['instance_id'],
                         'sample_index':row_index,'instance_index':instance_index,
                         'image':item['record']['image'],'scene_family':item['record']['scene_family'],
                         'status':item['record'].get('status')})
    device = select_device(args.device)
    allowed = {'GPU-1648afa2-3e25-b1c7-b149-38b07306eb8f','GPU-8c93aadc-2a0a-665c-46c3-dbd3f063e0d8',
               'GPU-b1ae5be6-1dba-7343-e233-59df41402f33','GPU-a68477c3-1c2a-d557-58f5-405d80e5383e'}
    if device.type != 'cuda' or torch.cuda.device_count() != 1:
        raise ValueError('Diagnostic requires one explicitly assigned CUDA device')
    gpu_uuid = 'GPU-'+str(torch.cuda.get_device_properties(device).uuid)
    if gpu_uuid not in allowed or os.environ.get('CUDA_VISIBLE_DEVICES') != gpu_uuid:
        raise ValueError('CUDA physical identity is not the assigned allowed UUID')
    torch.cuda.reset_peak_memory_stats(device)
    config = copy.deepcopy(base_config)
    config['runtime'].update({'device': str(device), 'require_cuda': True, 'workers': 0,
                              'cpu_threads': 2, 'amp': False, 'seed': seed})
    config['stages']['D0'].update({'roi_count': args.roi_count, 'global_batch': args.roi_count, 'micro_batch': 1})
    config['video']['frame_manifest'] = None
    model_dir, output_dir = run_directories(root, args.run_id)
    dependencies = ['src/data/action4_adapter.py', 'src/data/action4_dataset.py',
                    'src/model/action4_network.py', 'src/model/action4_losses.py',
                    'src/model/action4_decode.py', 'src/model/training/action4_trainer.py',
                    'src/model/training/action4_runtime.py']
    snapshots = output_dir / 'source_snapshot'
    snapshots.mkdir()
    hashes = {}
    for relative in dependencies:
        source = root / relative
        data = source.read_bytes()
        target = snapshots / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as handle:
            handle.write(data)
        hashes[relative] = sha256_file(source)
    helper = Path(__file__).resolve()
    with (snapshots / helper.name).open('xb') as handle:
        handle.write(helper.read_bytes())
    protocol = {'schema_version': 'action4.gpu_fixed_roi_diagnostic.v1',
                'claim_scope': 'component_diagnostic', 'formal_D0_gate_eligible': False,
                'root': str(root), 'manifest': str(manifest.resolve()),
                'manifest_sha256': sha256_file(manifest), 'selected': selected,
                'config': config, 'base_config': base_config,
                'architecture_changed': False, 'pretrained_backbone': True,
                'fixed_GT_ROI_inputs': True, 'full_image_detection_trained': False,
                'augmentation': None, 'backbone_frozen_updates': 0,
                'global_roi_batch': args.roi_count, 'micro_batch': 1, 'accumulation': args.roi_count, 'gpu_uuid': gpu_uuid, 'precision': 'fp32',
                'schedule_total_updates': 500, 'schedule_warmup_updates': 0,
                'requested_updates': args.updates, 'probe_only': args.probe_only,
                'wall_limit_seconds': args.max_seconds, 'process_memory_limit_gib': args.max_memory_gib,
                'timing_scope': '32 cachedGT-ROI forward/backward passes plus clippedAdamW update and CUDA synchronization; excludes data decode/eval',
                'source_sha256': hashes, 'helper_sha256': sha256_file(helper),
                'environment': environment_identity(torch)}
    write_json_new(output_dir / 'protocol.json', protocol)
    images, records = [i['image'] for i in items], [i['record'] for i in items]
    boxes = [record['instances'][gi]['bbox_full_xyxy'] for record, gi in zip(records, instance_indices)]
    rois, affines = prepare_rois(images, boxes, list(range(args.roi_count)), roi_size=256,
                                expansion=config['preprocess']['roi_expansion'], device=device)
    targets, valid = build_corner_targets(records, list(range(args.roi_count)), instance_indices, affines,
                                          roi_size=256, heatmap_size=64, device=device)
    if not bool(valid.all()):
        raise ValueError('This fixedGT-ROI diagnostic requires32 valid corner targets')
    down = np.array([[.25, 0, -.375], [0, .25, -.375], [0, 0, 1.]])
    truth = torch.tensor(np.asarray([DataAdapter.transform_points(record['instances'][gi]['corners_px'], down @ affine)
                                   for record, gi, affine in zip(records, instance_indices, affines)]), dtype=torch.float32, device=device)
    ideal = decode_corners(targets, from_logits=False)
    zero = decode_corners(torch.zeros_like(targets), from_logits=False)
    norm = lambda points: torch.linalg.vector_norm(points - truth, dim=-1)
    ideal_errors, zero_errors = norm(ideal), norm(zero)
    ideal_p95 = float(torch.quantile(ideal_errors, .95))
    zero_p95 = float(torch.quantile(zero_errors, .95))
    if ideal_p95 > 1:
        raise ValueError('Ideal distance heatmap decode failed its1px check before model training')
    model = Action4Net(weights=None, **config['network'])
    initialization = model.load_backbone_weights(root / config['initialization']['backbone_weights'],
                                                  config['initialization']['sha256'])
    model = model.to(device)
    if not all(parameter.requires_grad for parameter in model.backbone.parameters()):
        raise ValueError('D0 diagnostic must keep the entire backbone trainable')
    optimizer = build_optimizer(model, config['stages']['D0'])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: learning_rate_factor(step, 500, 0))
    first_backbone = next(model.backbone.parameters())
    starting_backbone = first_backbone.detach().clone()
    starting_bn = model.backbone.bn1.running_mean.clone()
    history, gradient_seen = [], False

    def evaluate(step):
        model.eval()
        maps = []
        with torch.inference_mode():
            for index in range(args.roi_count):
                maps.append(model(None, roi_images=rois[index:index + 1])['corner_logits'])
        logits = torch.cat(maps)
        prediction = decode_corners(logits)
        errors = norm(prediction)
        row = {'step': step, 'epe_mean_heatmap_px': float(errors.mean()),
               'epe_p95_heatmap_px': float(torch.quantile(errors, .95)),
               'valid_corners': int(valid.sum()), 'ideal_heatmap_p95_px': ideal_p95,
               'zero_heatmap_p95_px': zero_p95,
               'improvement_over_zero_p95': 1 - float(torch.quantile(errors, .95)) / zero_p95,
               'corner_peak_scores': logits.sigmoid().flatten(2).amax(-1).tolist(),
               'corners_hm': prediction.tolist(), 'errors_heatmap_px': errors.tolist(),
               'argmax_xy': torch.stack((logits.flatten(2).argmax(-1)%64, logits.flatten(2).argmax(-1)//64), -1).tolist(),
               'corner_spread_xy': (prediction.amax(1)-prediction.amin(1)).tolist()}
        np.savez_compressed(output_dir / f'heatmaps_step_{step:04d}.npz', logits=logits.cpu().numpy(), targets=targets.cpu().numpy(), valid=valid.cpu().numpy())
        history.append(row)
        with (output_dir / 'epe_history.jsonl').open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(row, allow_nan=False) + '\n')
        return prediction.detach().cpu()

    before = evaluate(0)
    write_json_new(output_dir / 'corner_reference.json', {'selected': selected, 'corners_hm': truth.tolist(),
                   'ideal_decoded_hm': ideal.tolist(), 'zero_decoded_hm': zero.tolist(),
                   'orig_to_roi': [a.tolist() for a in affines],
                   'scope': 'synthetic reference used only for this fixedGT-ROI training/diagnostic'})
    updates, stopped_reason, durations, losses = 0, None, [], []
    started = time.perf_counter()
    for step in range(args.updates):
        if time.perf_counter() - started > args.max_seconds:
            stopped_reason = 'wall_time_limit'
            break
        model.train()
        optimizer.zero_grad(set_to_none=True)
        used_lrs = {g['group_name']: g['lr'] for g in optimizer.param_groups}
        torch.cuda.synchronize(device)
        update_start = time.perf_counter()
        accumulated = 0.
        for index in range(args.roi_count):
            logits = model(None, roi_images=rois[index:index + 1])['corner_logits']
            loss = corner_heatmap_loss(logits, targets[index:index + 1], valid[index:index + 1],
                                       torch.tensor([0], dtype=torch.long, device=device), 1)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError('Nonfinite ROI loss')
            (loss / args.roi_count).backward()
            accumulated += float(loss.detach()) / args.roi_count
        if not all(parameter.grad is None or bool(torch.isfinite(parameter.grad).all()) for parameter in model.parameters()):
            raise FloatingPointError('Nonfinite gradient; no optimizer update')
        gradient_seen |= first_backbone.grad is not None and bool(first_backbone.grad.abs().sum() > 0)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        if not bool(torch.isfinite(grad_norm)):
            raise FloatingPointError('Nonfinite gradient norm')
        optimizer.step()
        scheduler.step()
        updates += 1
        torch.cuda.synchronize(device)
        seconds = time.perf_counter() - update_start
        peak_gib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2
        durations.append(seconds)
        losses.append(accumulated)
        row = {'step': updates, 'loss': accumulated, 'seconds': seconds, 'peak_process_rss_gib': peak_gib, 'cuda_peak_allocated_bytes': torch.cuda.max_memory_allocated(device), 'cuda_peak_reserved_bytes': torch.cuda.max_memory_reserved(device),
               'gradient_norm_before_clip': float(grad_norm), 'lr_used': used_lrs,
               'claim_scope': 'component_diagnostic'}
        with (output_dir / 'train_updates.jsonl').open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(row, allow_nan=False) + '\n')
        if updates % 50 == 0 or updates == args.updates:
            print(json.dumps(row), flush=True)
        if peak_gib >= args.max_memory_gib:
            stopped_reason = 'process_memory_limit'
            break
        if updates % 50 == 0 or updates == args.updates:
            evaluate(updates)
    if history[-1]['step'] != updates:
        after = evaluate(updates)
    else:
        after = torch.tensor(history[-1]['corners_hm'])
    changed = not torch.equal(starting_backbone, first_backbone.detach())
    bn_unchanged = torch.equal(starting_bn, model.backbone.bn1.running_mean)
    if not bn_unchanged:
        raise RuntimeError('Backbone runningBN stats changed despite frozen-stat protocol')
    # Keep geometry-derived GT visibly separate from model outputs. Pages
    # remain readable; a 32-row strip made failed predictions look like labels.
    mean = torch.tensor([.485, .456, .406])[None, :, None, None]
    std = torch.tensor([.229, .224, .225])[None, :, None, None]
    decoded_rgb = ((rois.cpu() * std + mean).clamp(0, 1) * 255).round().byte()
    font = ImageFont.load_default(size=14)
    edges = ((0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7))
    annotation_dir = contained_path(root, output_dir / 'visualizations')
    annotation_dir.mkdir(parents=True, exist_ok=False)
    def comparison(indices, destination):
        canvas = Image.new('RGB', (1100,72+len(indices)*312), (246,246,246))
        drawer = ImageDraw.Draw(canvas)
        drawer.text((12,8),'CHECK ONLY: green = synthetic GT; red = model predictions, NOT training labels',font=font,fill=(20,20,20))
        for row_index,i in enumerate(indices):
            source = Image.fromarray(decoded_rgb[i].permute(1,2,0).numpy())
            for panel,points in enumerate((None,truth[i].cpu(),before[i],after[i])):
                tile=source.copy();draw=ImageDraw.Draw(tile)
                if points is not None:
                    xy=points.cpu().numpy()*4+1.5
                    color=(0,255,80) if panel==1 else (255,30,30)
                    for u,v in edges:draw.line((tuple(xy[u]),tuple(xy[v])),fill=color,width=2)
                    for corner_id,point in enumerate(xy):draw.text(tuple(point),str(corner_id),fill=color,font=font,stroke_width=1,stroke_fill=(0,0,0))
                x,y=12+panel*272,72+row_index*312
                canvas.paste(tile,(x,y))
                drawer.text((x,y-24),('RAW ROI','SYNTHETIC GT - full body','INITIAL PREDICTION',f'PREDICTION - step {updates}')[panel],font=font,fill=(20,20,20))
            drawer.text((12,72+row_index*312+260),selected[i]['sample_id']+' / instance '+str(selected[i]['instance_id']),font=font,fill=(20,20,20))
        canvas.save(destination)
    diagnostic_pages=[]
    for first in range(0,args.roi_count,8):
        page=annotation_dir/f'roi_gt_prediction_page_{first//8+1:02d}.png'
        comparison(list(range(first,min(first+8,args.roi_count))),page)
        diagnostic_pages.append(str(page.relative_to(root)))
    annotation_image=annotation_dir/'roi_gt_prediction_overview.png'
    comparison([0,8,16,24],annotation_image)
    final_checkpoint = None
    if not args.probe_only:
        final_checkpoint = model_dir / 'component_final.pt'
        checkpoint = {'schema_version': 'action4.checkpoint.v1', 'stage': 'D0', 'run_id': args.run_id,
                      'claim_scope': 'component_diagnostic', 'config': config,
                      'identity': {'helper': protocol['helper_sha256'], 'source': hashes,
                                   'manifest': protocol['manifest_sha256'], 'selected': selected},
                      'initialization': {'mode': 'imagenet_backbone', **initialization},
                      'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                      'scheduler': scheduler.state_dict(), 'scaler': {}, 'global_step': updates,
                      'epoch_boundary': False, 'next_epoch': 0, 'rng_by_rank': [capture_rng()],
                      'resume_scope': 'not compatible with formal trainer exact-resume identity; component checkpoint only'}
        save_checkpoint_atomic(final_checkpoint, checkpoint)
    result = {'status': 'bounded_component_complete' if stopped_reason is None else 'bounded_component_stopped',
              'claim_scope': 'component_diagnostic', 'formal_D0_gate_pass': False,
              'architecture': model.configuration, 'initialization': initialization,
              'updates': updates, 'requested_updates': args.updates, 'stop_reason': stopped_reason,
              'annotation_image': str(annotation_image.relative_to(root)), 'diagnostic_pages': diagnostic_pages,
              'roi_count': args.roi_count, 'gpu_uuid': gpu_uuid, 'cuda_peak_allocated_bytes': torch.cuda.max_memory_allocated(device), 'cuda_peak_reserved_bytes': torch.cuda.max_memory_reserved(device), 'parameters': sum(p.numel() for p in model.parameters()),
              'first_update_seconds': durations[0] if durations else None,
              'second_update_seconds': durations[1] if len(durations) > 1 else None,
              'median_update_seconds_after_first': float(np.median(durations[1:])) if len(durations) > 1 else None,
              'peak_process_rss_gib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2,
              'loss_first': losses[0] if losses else None, 'loss_last': losses[-1] if losses else None,
              'before': history[0], 'after': history[-1],
              'finite_nonzero_backbone_gradient': bool(gradient_seen), 'backbone_parameter_changed': changed,
              'backbone_running_stats_unchanged': bn_unchanged,
              'checkpoint': str(final_checkpoint) if final_checkpoint else None,
              'checkpoint_sha256': sha256_file(final_checkpoint) if final_checkpoint else None,
              'source_changed_during_run': {name: sha256_file(root / name) != digest for name, digest in hashes.items()},
              'limitations': ['32 fixed syntheticGT crops only', 'pilot data remains unaccepted',
                              'no full-image detector training', 'no eval/test generalization claim',
                              'not a successful real-video annotation result']}
    write_json_new(output_dir / 'completion.json', result)
    print(json.dumps({'completion': str(output_dir / 'completion.json'), 'updates': updates,
                      'second_update_seconds': result['second_update_seconds'],
                      'peak_process_rss_gib': result['peak_process_rss_gib'],
                      'final_p95_heatmap_px': history[-1]['epe_p95_heatmap_px']}), flush=True)

    return result
