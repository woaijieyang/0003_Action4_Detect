"""Delete retired phrase2 intermediates under a sealed allowlist; never rewrite formal data."""
from pathlib import Path
from collections import Counter
import argparse,hashlib,json,os,re,sqlite3,stat,subprocess,time
from data.action4_paths import PROJECT_ROOT as R,contained_path
from data.action4_occlusion500_similarity import write
OUT=R/'outputs/phrase2_cleanup_v001'
BATCHES=['phrase2_generation_v001','phrase2_occlusion500_v001','coverage1000_v001']
FORMAL=R/'data/phrase2'

def walk(root):
 if not root.exists():return
 if root.is_symlink() or not root.resolve().is_relative_to(R):raise ValueError('Unsafe root '+str(root))
 for parent,ds,fs in os.walk(root,followlinks=False):
  for d in list(ds):
   p=Path(parent)/d
   if p.is_symlink():ds.remove(d)
  for f in fs:
   p=Path(parent)/f
   if not p.is_symlink() and p.is_file():yield p

def live_gate():
 p=subprocess.run(['pgrep','-af',r'(data.action4|model.training.action4|eval.action4|/data/zy/vln/a-test/src/)'],capture_output=True,text=True)
 if p.returncode not in [0,1]:raise RuntimeError('Cannot inspect active project consumers')
 hits=[]
 for line in p.stdout.splitlines():
  _,_,cmd=line.partition(' ')
  if 'action4_phrase2_cleanup' in cmd or cmd.startswith(('bash -c','sh -c')):continue
  if 'data.action4' in cmd or 'model.training.action4' in cmd or 'eval.action4' in cmd or '--python /data/zy/vln/a-test/src/' in cmd:hits.append(line)
 if hits:raise RuntimeError('Active project process blocks scoped cleanup: '+str(hits))

def statrow(path):
 s=path.stat();return {'path':str(path.relative_to(R)),'device':s.st_dev,'inode':s.st_ino,'bytes':s.st_size,'mtime_ns':s.st_mtime_ns,'links':s.st_nlink,'blocks':s.st_blocks*512}

def snapshots():
 commit=json.loads((FORMAL/'DATASET_COMPLETE.json').read_text());assert commit['status']=='FORMAL_DATASET_USABLE' and commit['images']==6500
 manifests={};formal={};refs=set()
 def collect(x):
  if isinstance(x,str) and x.startswith('outputs/'):
   p=R/x
   if p.exists() and p.is_file() and not p.is_symlink():refs.add(p)
  elif isinstance(x,dict):
   for v in x.values():collect(v)
  elif isinstance(x,list):
   for v in x:collect(v)
 for split,n in commit['splits'].items():
  p=FORMAL/split/'annotations.jsonl';blob=p.read_bytes();assert hashlib.sha256(blob).hexdigest()==commit['manifest_sha256'][split]
  rows=[json.loads(x) for x in blob.splitlines()];assert len(rows)==n
  manifests[split]={'sha256':commit['manifest_sha256'][split],'identity':statrow(p)}
  for row in rows:
   collect(row);paths=[row['image']]
   for obj in row['instances']:paths +=[obj['mask_visible'],obj['mask_amodal_expanded'],obj['surface_supervision']['path']]
   for name in paths:
    path=contained_path(R,name,exists=True);assert path.is_relative_to(FORMAL/split) and not path.is_symlink()
    formal[(path.stat().st_dev,path.stat().st_ino)]=statrow(path)
 assert len(formal)==45500
 models={name:statrow(R/'models/default'/name) for name in ['A0.pt','B0.pt']}
 return commit,manifests,formal,refs,models

def prepare():
 if Path.cwd().resolve()!=R:raise ValueError('Project directory required')
 OUT.mkdir(exist_ok=True);live_gate();commit,manifests,formal,refs,models=snapshots();protected=set(refs)
 # Keep every file directly named by a reviewed gate, and all frozen output sources.
 def protect(x):
  if isinstance(x,str) and x.startswith('outputs/'):
   p=R/x
   if p.is_file() and not p.is_symlink():protected.add(p)
  elif isinstance(x,dict):
   for k,v in x.items():
    if k.startswith('outputs/'):protect(k)
    protect(v)
  elif isinstance(x,list):
   for v in x:protect(v)
 for batch in BATCHES:
  root=R/'outputs'/batch
  con=sqlite3.connect(f'file:{root}/slots.sqlite3?mode=ro',uri=True)
  expected={'phrase2_generation_v001':5000,'phrase2_occlusion500_v001':500,'coverage1000_v001':1000}[batch]
  assert con.execute('select status,count(*) from slots group by status').fetchall()==[('accepted',expected)];con.close()
  for p in walk(root):
   if p.suffix=='.json' and (p.name.startswith(('ROOT_','SOURCE_FREEZE')) or p.name in ['PUBLICATION_GATE.json','DATASET_COMPLETE.json']):protect(json.loads(p.read_text()))
 # Final representative packets: preserve two examples per group and all scenes.
 bundle=json.loads((R/'outputs/coverage1000_v001/final_review_v001/FINAL_REVIEW_BUNDLE.json').read_text())
 chosen=[]
 for group in 'ABCDE':
  rows=[r for r in bundle['rows'] if r['slot']['group']==group]
  for row in rows[:2]:chosen.append(row)
 scenes={r['slot']['scene_id'] for r in chosen}
 for scene in ['table_full_garment','cloth_lift10_full_garment','wall']:
  if scene not in scenes:chosen.append(next(r for r in bundle['rows'] if r['slot']['scene_id']==scene));scenes.add(scene)
 selected_roots=[R/r['inspection_root'] for r in chosen]
 for root in selected_roots:protected.update(walk(root))
 # Keep complete final preview packets from the previous two published releases.
 for name in ['outputs/phrase2_generation_v001/preview_review_final_v002','outputs/phrase2_occlusion500_v001/preview_review_final500']:
  protected.update(walk(R/name))
 # Cache descriptors and small source identity records remain useful for future extensions.
 candidates=[];seen=set()
 def add(p,reason):
  if p in seen or p in protected:return
  s=statrow(p)
  if (s['device'],s['inode']) in formal:
   dst=R/formal[(s['device'],s['inode'])]['path'];assert os.path.samefile(p,dst) and p!=dst
   s['formal_hardlink_preserved']=str(dst.relative_to(R))
  elif p.is_relative_to(FORMAL) and not any(part in ['.staging','.staging_batches'] for part in p.parts):raise ValueError('Formal artifact cannot be a cleanup candidate')
  s['reason']=reason;candidates.append(s);seen.add(p)
 for batch in BATCHES:
  root=R/'outputs'/batch
  for p in walk(root):
   rel=p.relative_to(root);ext=p.suffix.lower()
   if p in protected or (rel.parts and rel.parts[0] in ['similarity','assets','source_snapshots']):continue
   if p.name in ['FRAME_SNAPSHOT.json','CANDIDATE.json']:
    if p not in refs and rel.parts[0]=='workers':add(p,'rejected_or_superseded_frame_metadata')
    continue
   if ext in ['.log','.tmp','.temp','.exr'] or p.name.endswith(('.sqlite3-wal','.sqlite3-shm')):add(p,'completed_run_log_or_temporary')
   elif ext in ['.png','.jpg','.jpeg','.webp','.npy','.npz']:
    # Asset meshes/caches outside explicit scratch are not image process files.
    if ext in ['.npy','.npz'] and rel.parts[0] not in ['workers','formal128_review_v001','final_review_v001']:continue
    if ext in ['.jpg','.jpeg','.png'] and ('contact' in p.name.lower() or re.match(r'(REVIEW|FINAL)_\d+\.',p.name)):continue
    add(p,'redundant_render_mask_or_preview')
 staging=[FORMAL/'.staging_batches'/x for x in ['coverage1000_v001','occlusion500_v001']]+[FORMAL/s/'.staging' for s in ['train','eval','test']]
 for root in staging:
  for p in walk(root):
   if p.name in ['annotations.jsonl','README.md','RETIRED_AFTER_PUBLICATION.json']:continue
   add(p,'published_candidate_hardlink' if (p.stat().st_dev,p.stat().st_ino) in formal else 'rejected_or_retired_unpublished_candidate')
 # Record allocation once per inode, only if every existing hardlink will be removed.
 by_inode={}
 for x in candidates:
  v=by_inode.setdefault((x['device'],x['inode']),{'deleted_links':0,'links':x['links'],'blocks':x['blocks']});v['deleted_links']+=1
 estimate=sum(x['blocks'] for x in by_inode.values() if x['deleted_links']==x['links'])
 path=OUT/'DELETE_MANIFEST.jsonl';path.write_text(''.join(json.dumps(x)+'\n' for x in candidates))
 baseline={'formal_completion':commit,'manifests':manifests,'formal_files':list(formal.values()),'models':models,'protected_references':[str(p.relative_to(R)) for p in sorted(refs)],'representative_packets':[r['sample_id'] for r in chosen]}
 write(OUT/'PROTECTED_BASELINE.json',baseline)
 plan={'status':'READY_DELETE_SCOPED_INTERMEDIATES','scope':[f'outputs/{b}' for b in BATCHES]+[str(p.relative_to(R)) for p in staging],
  'files':len(candidates),'logical_bytes':sum(x['bytes'] for x in candidates),'estimated_reclaimable_allocated_bytes':estimate,'reasons':dict(Counter(x['reason'] for x in candidates)),
  'delete_manifest_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'formal_files_protected':45500,'provenance_files_protected':len(refs),
  'no_training_or_weights_change':True,'created_unix':time.time()}
 write(OUT/'PLAN.json',plan);return plan

def apply():
 live_gate();plan=json.loads((OUT/'PLAN.json').read_text());manifest=OUT/'DELETE_MANIFEST.jsonl'
 assert hashlib.sha256(manifest.read_bytes()).hexdigest()==plan['delete_manifest_sha256']
 baseline=json.loads((OUT/'PROTECTED_BASELINE.json').read_text());assert json.loads((FORMAL/'DATASET_COMPLETE.json').read_text())==baseline['formal_completion']
 targets=[json.loads(x) for x in manifest.read_text().splitlines()];allowed=[R/x for x in plan['scope']]
 for row in targets:
  p=R/row['path'];assert any(p.is_relative_to(root) for root in allowed)
  assert not p.is_symlink() and p.resolve().is_relative_to(R)
  s=p.stat();assert(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns)==(row['device'],row['inode'],row['bytes'],row['mtime_ns'])
  if 'formal_hardlink_preserved' in row:assert os.path.samefile(p,R/row['formal_hardlink_preserved'])
 before=os.statvfs(R).f_bavail*os.statvfs(R).f_frsize;deleted=0
 with (OUT/'DELETED.jsonl').open('x') as log:
  for row in targets:
   p=R/row['path'];p.unlink();deleted+=1
   log.write(json.dumps({'path':row['path'],'bytes':row['bytes'],'reason':row['reason']})+'\n')
  log.flush();os.fsync(log.fileno())
 for root in allowed:
  for parent,ds,fs in os.walk(root,topdown=False,followlinks=False):
   p=Path(parent)
   if p!=root and not p.is_symlink() and not any(p.iterdir()):p.rmdir()
 # Provenance and public artifacts are unchanged; hardlink count itself may decrease.
 for entry in baseline['formal_files']+list(baseline['models'].values()):
  p=contained_path(R,entry['path'],exists=True);s=p.stat();assert(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns)==(entry['device'],entry['inode'],entry['bytes'],entry['mtime_ns'])
 for split,entry in baseline['manifests'].items():assert hashlib.sha256((FORMAL/split/'annotations.jsonl').read_bytes()).hexdigest()==entry['sha256']
 for name in baseline['protected_references']:assert contained_path(R,name,exists=True).is_file()
 after=os.statvfs(R).f_bavail*os.statvfs(R).f_frsize
 result={'status':'DELETED_SCOPED_INTERMEDIATES_FORMAL_DATA_PRESERVED','files_deleted':deleted,'logical_bytes_unlinked':plan['logical_bytes'],
  'estimated_reclaimable_allocated_bytes':plan['estimated_reclaimable_allocated_bytes'],'filesystem_free_bytes_delta_observed':after-before,
  'formal_images':6500,'formal_artifacts_preserved':45500,'formal_manifests_unchanged':True,'default_checkpoints_unchanged':True,
  'provenance_files_preserved':len(baseline['protected_references']),'time_unix':time.time()}
 write(OUT/'COMPLETE.json',result);return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--apply',action='store_true');a=p.parse_args();print(json.dumps(apply() if a.apply else prepare(),indent=2))
