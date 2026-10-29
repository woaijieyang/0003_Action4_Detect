"""Close coverage1000 delivery with actual published reads and bound reports. No training."""
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone, timedelta
import ast,copy,hashlib,json,os,sqlite3,time
import numpy as np
import torch
from data.action4_paths import PROJECT_ROOT as R,contained_path
from data.action4_occlusion500_similarity import write
from data.action4_stage2_data import Stage2Units
from data.action4_phrase2_dataset import surface_source
from data.action4_dataset import prepare_full_images

O=R/'outputs/coverage1000_v001'
def load(p):return json.loads(p.read_text())
def main():
 assert Path.cwd().resolve()==R
 torch.set_num_threads(1)
 done=load(O/'MERGE_COMPLETE.json');complete=load(R/'data/phrase2/DATASET_COMPLETE.json')
 assert done['status']=='FORMAL6500_DATASET_USABLE' and complete['status']=='FORMAL_DATASET_USABLE'
 assert complete['splits']=={'train':5200,'eval':650,'test':650}
 baseline=load(O/'BASELINE_IDENTITY.json');audit=load(O/'FINAL_AUDIT.json');proof=load(O/'SIMILARITY_COMPLETE.json')
 assert complete['similarity_dataset_identity']==proof['dataset_identity']==audit['global_identity']
 cfg=load(R/'configs/data/action4_phrase2_consumer_v001.json')
 assert cfg['training_enabled'] is False
 source=surface_source();live=[];samples=[];allids=set();old_bytes=0;new_bytes=0
 for split,n in complete['splits'].items():
  manifest=R/'data/phrase2'/split/'annotations.jsonl';blob=manifest.read_bytes()
  assert hashlib.sha256(blob).hexdigest()==complete['manifest_sha256'][split]
  prior=(R/'data/phrase2/releases/base5500_v001'/(split+'_annotations.jsonl')).read_bytes()
  assert hashlib.sha256(prior).hexdigest()==baseline['manifests'][split]['sha256'] and blob.startswith(prior)
  records=[json.loads(x) for x in blob.splitlines()]
  assert len(records)==n and len({r['sample_id'] for r in records})==n
  for row in records:
   assert row['split']==split and len(row['instances'])==2 and row['sample_id'] not in allids
   allids.add(row['sample_id'])
   assert [x['instance_id'] for x in row['instances']]==[1,2]
   for obj in row['instances']:
    assert np.asarray(obj['corners_2d']).shape==(8,2) if 'corners_2d' in obj else True
  base={'mode':'integration','seed':2026102304,'dataset_root':'data/phrase2','surface_source':source,
        'roi_jitter':{'center_fraction':0.,'scale_range':[1.,1.]}}
  a=Stage2Units(dict(base,role='A0'),split);b=Stage2Units(dict(base,role='B0'),split)
  assert len(a)==2*n and len(b)==n
  chosen=[next(i for i,r in enumerate(records) if r.get('batch_id')!='coverage1000_v001'),next(i for i,r in enumerate(records) if r.get('batch_id')=='coverage1000_v001')]
  for i in chosen:
   item=b[(i,0,True,1)]['item'];rgb,_=prepare_full_images([item['image']],size=(1024,768),device='cpu')
   assert tuple(rgb.shape)==(1,3,768,1024) and torch.isfinite(rgb).all()
   for j in range(2):
    u=a[(2*i+j,0,True,1)]
    assert tuple(u['roi'].shape)==(3,448,448) and tuple(u['surface_masks'].shape)==(2,112,112)
    assert u['geometry']['corners'].shape==(8,2) and torch.isfinite(u['geometry']['corners']).all()
    assert len(u['surface_sparse']['xyz'])>0 and torch.isfinite(u['surface_sparse']['xyz']).all()
    samples.append({'split':split,'sample_id':records[i]['sample_id'],'batch_id':records[i].get('batch_id','base5000'),
      'instance_id':j+1,'native_size':[records[i]['width'],records[i]['height']],'A0_roi':list(u['roi'].shape),'B0_batch':list(rgb.shape),
      'sparse_points':len(u['surface_sparse']['xyz']),'hidden_points':int((~u['geometry']['visible']).sum())})
  live.append({'split':split,'B0_images':len(b),'A0_target_units':len(a),'objects':2*n})
 # Stat identities were sealed at startup and rechecked by merge. No repeated full RGB hashing.
 oldfiles=[json.loads(x) for x in (O/'OLD5500_FILE_IDENTITIES.jsonl').read_text().splitlines()]
 assert len(oldfiles)==38500
 old_bytes=sum(x['bytes'] for x in oldfiles);new_bytes=audit['bytes']
 for role,entry in baseline['models'].items():
  s=contained_path(R,entry['path'],exists=True).stat()
  assert (s.st_ino,s.st_size,s.st_mtime_ns)==(entry['inode'],entry['bytes'],entry['mtime_ns'])
 counters=Counter();rendered=0;workers=0
 for directory in (O/'workers').iterdir():
  if directory.is_symlink() or not directory.is_dir():continue
  p=directory/'COMPLETE.json'
  if not p.exists():raise ValueError('Generation worker has not completed: '+directory.name)
  d=load(p);workers+=1;rendered+=d['rendered'];counters.update(d['rejections'])
 db=sqlite3.connect(str(O/'slots.sqlite3'));assert db.execute('select status,count(*) from slots group by status').fetchall()==[('accepted',1000)]
 failures=dict(db.execute('select reason,count from failures'));assert rendered==1000+sum(failures.values())
 preview=load(O/'runs/preview_v001/COMPLETE.json');bulk=load(O/'runs/bulk_v001/COMPLETE.json')
 seconds=preview['elapsed_seconds']+bulk['elapsed_seconds'];dist=load(O/'DISTRIBUTION_REPORT.json')
 assert sum(x['targets'] for x in dist['new1000']['observation_direction_x_size_x_occlusion'])==2000
 now=time.time();stamp=datetime.fromtimestamp(now,timezone(timedelta(hours=8))).isoformat(timespec='seconds')
 write(O/'MONITOR_CLOSED.json',{'status':'DELETED','automation_id':'action4-coverage1000','delete_status':'deleted','tool_confirmation':'Deleted automation in the app.','time_unix':now})
 validation={'status':'PASS_FORMAL6500_PUBLISHED_CONSUMPTION','splits':live,'actual_published_samples':samples,
  'manifest_sha256':complete['manifest_sha256'],'old5500_manifest_exact_byte_prefix_preserved':True,
  'all13000_B0_targets_verified_in':'outputs/coverage1000_v001/FORMAL_ROOT_INTEGRATION_VALIDATION.json',
  'old5500_files_unchanged_verified_in':'outputs/coverage1000_v001/MERGE_COMPLETE.json',
  'default_model_steps':{r:e['step'] for r,e in baseline['models'].items()},'no_model_or_optimizer_constructed':True,'time_unix':now}
 write(O/'FINAL_DELIVERY_VERIFICATION.json',validation)
 report={'status':'DELIVERED_FORMAL6500_NO_TRAINING','completion':str((R/'data/phrase2/DATASET_COMPLETE.json').relative_to(R)),
  'images':6500,'targets':13000,'corners':104000,'new_images':1000,'groups':audit['groups'],'distributions':dist,
  'storage':{'old5500_artifact_bytes':old_bytes,'new1000_artifact_bytes':new_bytes,'total6500_artifact_bytes':old_bytes+new_bytes,
   'artifact_files':45500,'method':'Sum formal referenced RGB/mask/geometry sizes once; staging/worker hardlinks are not counted twice. Excludes manifests, logs, assets and diagnostic outputs.'},
  'performance':{'preview_seconds':preview['elapsed_seconds'],'preview_images_per_minute':128*60/preview['elapsed_seconds'],
   'bulk_seconds':bulk['elapsed_seconds'],'bulk_images_per_minute':872*60/bulk['elapsed_seconds'],
   'combined_generation_seconds':seconds,'combined_images_per_minute':1000*60/seconds,
   'task_elapsed_including_asset_repair_preflight_and_visual_reviews_seconds':now-baseline['time_unix'],
   'rendered_candidates':rendered,'accepted':1000,'formal_RGB_acceptance_rate':1000/rendered,
   'actual_mask_rejections':failures,'cheap_generation_rejections':dict(counters),'cheap_rejections_total':sum(counters.values()),
   'completed_persistent_worker_sessions':workers,'max_concurrent_renderers':32,'renderers_per_GPU':8,'used_GPU':[4,5,6,7],
   'logical_CPU_budget':72,'allocation':'64 render +6 host quality +2 controller, sequential bounded postprocessing',
   'interpretation':'Configured capacity is not measured sustained utilization. Hard joint constraints produced a geometry-bound tail; stage timings exclude earlier asset/preflight and visual waiting. Initial feasibility sweeps are separately retained, not counted as rendered images.'},
  'similarity':proof,'published_interface':validation,'visual_review':'outputs/coverage1000_v001/ROOT_FINAL_VISUAL_REVIEW.json',
  'asset_repair':'outputs/coverage1000_v001/arm_repair_v001/ROOT_ARM_VISUAL_REVIEW.json',
  'mechanical_clearance':'outputs/coverage1000_v001/arm_repair_v001/INTERNAL_MECHANICAL_AUDIT.json',
  'limitations':['Three simplified environments; no claim of photorealistic human anatomy or measured real-video model accuracy.',
   'Near-duplicate result is bound to frozen exact/geometry/foreground similarity policy, not all-pairs pixel comparison or proof of full distribution coverage.',
   'Accepted continuous parameters are physically coupled and rejection-conditioned, not independent uniform distributions after acceptance.',
   'Merged6500 retains the original5500 distribution, including its concentration at medium sizes. High occlusion remains concentrated in a subset of legal canonical viewing directions; the joint table is reported without claiming six-face balance.',
   'Geometry-rejection tail remains costly. The separate broader-prior test had no successes and was not adopted. Extra tail helpers never dispatched because the original frozen worker completed the final slot.'],
  'training_started':False,'default_checkpoints_changed':False,'time_unix':now}
 write(O/'DELIVERY_REPORT.json',report)
 # New immutable source snapshot corrects inherited pre-release metadata; previous source freezes remain intact.
 oldfreeze=O/'SOURCE_FREEZE_MERGED_v001.json';freeze=copy.deepcopy(load(oldfreeze))
 for p in list(freeze['files']):
  path=contained_path(R,p,exists=True)
  if p=='src/data/action4_coverage1000_merge.py':freeze['files'][p]=hashlib.sha256(path.read_bytes()).hexdigest()
  elif hashlib.sha256(path.read_bytes()).hexdigest()!=freeze['files'][p]:raise ValueError('Unexpected frozen source drift: '+p)
 this='src/data/action4_coverage1000_delivery.py';freeze['files'][this]=hashlib.sha256((R/this).read_bytes()).hexdigest()
 freeze.update(parent_freeze=str(oldfreeze.relative_to(R)),formal_data_published=True,time_unix=now,
   source_change='Post-run release metadata closure and actual published consumer verification only; frozen sampler/renderer/labels unchanged.')
 frozen=O/'SOURCE_FREEZE_MERGED_v002.json';write(frozen,freeze)
 current=load(R/'configs/execution/action4_active.json');authority=load(R/current['active_ref']['path']);assert authority['status']=='PHRASE2_6500_DELIVERED'
 revision='r'+str(int(authority['revision'][1:])+1).zfill(3)
 authority.update(revision=revision,parent_authority=current['active_ref']['path'],current_formal_split_counts=complete['splits'],
   current5500_release_still_usable=False,gates_remaining=[],live_generation_processes=0,monitor_automation_id=None,
   monitor_closed='outputs/coverage1000_v001/MONITOR_CLOSED.json',source_freeze=str(frozen.relative_to(R)),
   delivery_report='outputs/coverage1000_v001/DELIVERY_REPORT.json',final_verification='outputs/coverage1000_v001/FINAL_DELIVERY_VERIFICATION.json')
 authpath=R/f'configs/execution/action4_execution_{revision}.json';authpath.open('x').write(json.dumps(authority,ensure_ascii=False,indent=2))
 assert load(R/'configs/execution/action4_active.json')==current
 write(R/'configs/execution/action4_active.json',{'schema':'action4.active_execution_pointer.v1','active_ref':{'path':str(authpath.relative_to(R)),'sha256':hashlib.sha256(authpath.read_bytes()).hexdigest()}})
 for name in ['PIPELINE_STATUS.json','POSTPROCESS_STATUS.json','CURRENT_WORK.json']:
  p=O/name;backup=O/'source_snapshots/pre_delivery_status';backup.mkdir(parents=True,exist_ok=True)
  if p.exists() and not(backup/name).exists():(backup/name).write_bytes(p.read_bytes())
  write(p,{'status':'FORMAL6500_DELIVERED','authority':str(authpath.relative_to(R)),'completion':'data/phrase2/DATASET_COMPLETE.json',
    'counts':{'accepted':1000},'formal_images':6500,'splits':complete['splits'],'remaining_gates':[],'training_started':False,'monitor_active':False,'time_unix':now})
 # One current-plan/conclusion pair is maintained. Preserve all historical evidence.
 new=dist['new1000'];merged=dist['merged6500'];old=dist['old5500']
 section=f'''\n\n## coverage1000最终交付：6500张正式数据（{stamp}）\n\n本批已完成，权威`{authpath.relative_to(R)}`；原5500保留，新1000已合入`data/phrase2`。train/eval/test=5200/650/650，13000完整机身目标、104000固定ID角点。`DATASET_COMPLETE.json`、`GEOMETRY_ADMISSION.json`、消费配置与三split清单身份一致。本轮未训练、未改默认A0 step14000或B0 step3220，不再启动生成或已结束的控制器；小时监控已删除。旧段落中r083–r086的“当前/尚未发布”均为当时执行记录，当前状态以本段和active指针为准。\n\nA–E各200图，每组160/20/20、三场景140/40/20；总场景700/200/100。A/B/C/E指定小目标四区间各50，左右各100；B/D/E大小比三档70/70/60；C/D/E边缘间隔三档60/80/60。B/C/D指定高遮挡各100、三档30/40/30；E全部200的“小的那台”高遮挡三档60/80/60。逐槽使用最终bbox及visible/amodal实际面积验收，无静默降配额。完整各组区间、左右大小平衡及方向×大小×遮挡表见`outputs/coverage1000_v001/FINAL_AUDIT.json`和`DISTRIBUTION_REPORT.json`。\n\n| 实际指标及计数单位 | 原5500 | 新1000 | 合并6500 |\n|---|---:|---:|---:|\n| 宽度<8%的目标（个） | {old['small_lt_8_percent_instances']} | {new['small_lt_8_percent_instances']} | {merged['small_lt_8_percent_instances']} |\n| 大小比≥1.5的图（张） | {old['ratio_ge_1p5_images']} | {new['ratio_ge_1p5_images']} | {merged['ratio_ge_1p5_images']} |\n| 二维边缘及中心条件同时靠近（张） | {old['near_pair_images']} | {new['near_pair_images']} | {merged['near_pair_images']} |\n| 存在小目标且面积遮挡≥40%（张） | {old['small_and_high_images']} | {new['small_and_high_images']} | {merged['small_and_high_images']} |\n| 四条件组合且小机本身高遮挡（张） | {old['four_conditions_images']} | {new['four_conditions_images']} | {merged['four_conditions_images']} |\n\n小目标+高遮挡另按严格w<8%统计目标个数：原{old['small_and_occlusion_ge_40_instances']}个、新{new['small_and_occlusion_ge_40_instances']}个；不与上表的图像张数混用。新2000目标的光轴Z>=0.8/1.0m及1.2–1.4m实际数量为816/397/145，均超过100/30/10覆盖目标。新Z范围{new['optical_Z_m']['min']:.5f}–{new['optical_Z_m']['max']:.5f}m，同一CAD原点欧氏距离{new['euclidean_CAD_origin_distance_m']['min']:.5f}–{new['euclidean_CAD_origin_distance_m']['max']:.5f}m，不混称。左右宽度相关系数原{old['left_right_width_correlation']:.3f}、新{new['left_right_width_correlation']:.3f}、合并{merged['left_right_width_correlation']:.3f}，确实降低强耦合，但合并集仍保留旧分布。\n\n128正式预览16页与终态64幅代表图8页已实际逐页查看整图、原尺寸、固定ID/完整框、真实A0/B0输入及mask；通过128直接复用、不重复生成。全部1000逐帧通过工程/几何/同帧身份/标签检查，代表性人工式视觉审核不冒充逐张人工审完1000。新上臂54项内部机械净空检查通过；最后困难槽位冷重放角点最大误差0.000162px以内，不重渲染RGB。全批表面重投影最大{audit['max_surface_reprojection_px']:.8f}px、最少{audit['sparse_surface_points_min']}个有效稀疏点；不把微米级数值检查当真实识别成绩。\n\n全6500按冻结定义去重9.54秒，复用原5500特征及原对照证明，仅算1000新特征；56106索引候选对中55793旧旧对复用、313新相关对均未通过几何近似门，精确重复和近重复0，因此未剔除或补渲染。没有做全量两两RGB或SSIM；不能把此结果当作分布覆盖充分或通用感知唯一的证明。\n\n发布前两次确认无依赖旧清单的活跃训练/eval消费者，未停止他人进程。旧5500三个manifest原始字节完整保留为新清单前缀，38500旧数据文件inode/大小/mtime不变，默认模型身份不变；旧发布证明与轻量清单在`data/phrase2/releases/base5500_v001`，未复制整套旧RGB。新7000文件用无覆盖硬链接合入，不改旧split或sample_id。完成标志仅在正式路径RGB/ROI/mask/XYZ读取和全13000 B0完整框目标构造通过后发布。正式有完成标志的入口又实际读取了每split的新旧样本：A0为10400/1300/1300目标单位，B0为5200/650/650图；12个目标的RGB/完整角点/mask/XYZ/region链确实进入实际item，无模型或optimizer。\n\n新1000数据文件{new_bytes:,}字节，合并{old_bytes+new_bytes:,}字节（不重复计算staging硬链接，不含日志/资产）。正式渲染候选{rendered}、接受1000，RGB接受率{1000/rendered:.2%}；实际mask拒绝{json.dumps(failures,ensure_ascii=False)}。128渲染+host 211.16秒/36.37张每分钟；余872为3931.76秒/13.31张每分钟，两个渲染阶段合计{seconds/60:.2f}分钟/{1000*60/seconds:.2f}张每分钟。含资产修复、可行性、审核等待至交付总耗时{(now-baseline['time_unix'])/3600:.2f}小时，不能用纯生成时间冒充整任务耗时。32并发常驻渲染、GPU4–7各8、64渲染+6质检+2控制=72逻辑CPU预算；GPU2–3其他任务、GPU1评估与GPU0均未占用。困难组合尾部廉价几何拒绝仍多，细项见DELIVERY_REPORT，不把进程/显存数量冒充持续满负载。\n\n已有三背景、手指/皮肤仍简化，高遮挡观察方向存在物理偏向；现6500不是原视频精确分布，也尚未证明模型泛化提高。无近重复、数据覆盖、工程正确、实际训练精度分别报告。完整机器交付与限制：`outputs/coverage1000_v001/DELIVERY_REPORT.json`、`FINAL_DELIVERY_VERIFICATION.json`、`ROOT_FINAL_VISUAL_REVIEW.json`；执行源及参数见新SOURCE_FREEZE_MERGED_v002与原冻结配置。当前各阶段均完成，本任务结束后不自动训练。\n'''
 for path in [R/'docs/action4_phrase2_data.md',R/'docs/stages/2026-10-19-1-plan-Action4角点复现增补.md',R/'docs/stages/2026-10-19-1-conclusion-Action4角点复现增补.md']:
  text=path.read_text();assert '## coverage1000最终交付：6500张正式数据' not in text
  banner='> 当前数据发布：phrase2正式6500张（5200/650/650），coverage1000已完成；本文较早状态为历史执行记录，详见末尾最终交付段。\n\n'
  path.write_text(banner+text+section)
 print(json.dumps({'status':validation['status'],'authority':str(authpath.relative_to(R)),'storage':report['storage'],
  'rendered_candidates':rendered,'actual_mask_rejections':failures,'cheap_rejections_total':sum(counters.values()),
  'formal_counts':live,'total_task_hours':(now-baseline['time_unix'])/3600},indent=2))
if __name__=='__main__':main()
