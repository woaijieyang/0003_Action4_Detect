"""Invocation-local parallel PNG admission; final source-byte recheck is never cached."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from data import action4_adapter as module
class ParallelPngAdmission:
 def __init__(self,root,paths,workers=12):
  self.root=Path(root).resolve();self.paths=[Path(p).resolve(strict=True) for p in paths]
  if len(set(self.paths))!=len(self.paths) or any(not p.is_relative_to(self.root) for p in self.paths):raise ValueError('Unique project PNG paths required')
  if type(workers) is not int or not 1<=workers<=24:raise ValueError('Bounded worker count required')
  self.workers=workers;self.records={};self.calls={};self.old_size=module._validated_png_size;self.old_sha=module._sha256;self.restore_size=self.old_size;self.restore_sha=self.old_sha
 @staticmethod
 def stat(p):
  x=p.stat();return x.st_dev,x.st_ino,x.st_size,x.st_mtime_ns,x.st_ctime_ns
 def check_one(self,p):
  before=self.stat(p);size=self.old_size(p);sha=self.old_sha(p)
  if self.stat(p)!=before:raise ValueError('PNG changed during admission')
  return p,(before,size,sha)
 def checked(self,p):
  p=Path(p).resolve(strict=True)
  if p in self.records and self.stat(p)!=self.records[p][0]:raise ValueError('PNG changed after admission')
  return p
 def size(self,p):
  p=self.checked(p);return self.records[p][1] if p in self.records else self.old_size(p)
 def sha(self,p):
  p=self.checked(p)
  if p not in self.records:return self.old_sha(p)
  n=self.calls.get(p,0);self.calls[p]=n+1
  return self.records[p][2] if n==0 else self.old_sha(p)
 def __enter__(self):
  with ThreadPoolExecutor(max_workers=self.workers) as pool:self.records=dict(pool.map(self.check_one,self.paths))
  module._validated_png_size=self.size;module._sha256=self.sha;return self
 def __exit__(self,*args):
  module._validated_png_size=self.restore_size;module._sha256=self.restore_sha
