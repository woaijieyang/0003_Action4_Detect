"""Exclusive GPU3 quality-job entry; single-frame labels only."""
from pathlib import Path
import argparse,hashlib,json,os,sys,time,unittest
from data import action4_execution_authority as authority
from data.action4_quality_device import QUALITY_GPU_UUID,gpu_identity
R=Path('/data/zy/vln/a-test')
def main(argv=None):
 p=argparse.ArgumentParser(allow_abbrev=False);p.add_argument('--config',required=True);p.add_argument('--out',required=True);a=p.parse_args(argv)
 if Path.cwd().resolve()!=R:raise ValueError('Exact project root required')
 refs=authority.spec_references(R,{'kind':'quality','argv':[str(R/'.venv/bin/python'),'scripts/action4_quality.py','--config',a.config,'--out',a.out]})
 cfg=authority.readref(R,authority.ref(R,a.config));out=authority.bounded(R,a.out)
 if out==R/'outputs' or not out.is_relative_to(R/'outputs') or out.exists():raise ValueError('Fresh quality outputs required')
 if os.environ.get('CUDA_VISIBLE_DEVICES')!=QUALITY_GPU_UUID:raise ValueError('Only GPU3 for quality')
 gpu_identity(QUALITY_GPU_UUID)
 start=time.perf_counter()
 if cfg['mode']=='smoke':
  out.mkdir(parents=True)
  os.environ['ACTION4_QUALITY_TEST_MANIFESTS']=json.dumps(cfg['fixture_manifests'])
  sys.path.insert(0,str(R))
  suite=unittest.defaultTestLoader.loadTestsFromNames(['tests.test_action4_quality_math','tests.test_action4_quality_real'])
  with (out/'tests.log').open('x') as f:result=unittest.TextTestRunner(stream=f,verbosity=2).run(suite)
  receipts=[]
  for module,klass in [('tests.test_action4_quality_math','CUDAKernelTests'),('tests.test_action4_quality_real','RealRecordTests')]:
   cls=getattr(sys.modules[module],klass);aid=getattr(cls,'aid',None)
   if aid is not None:receipts.append(aid.receipt())
  report={'status':'PASS_GPU3_QUALITY_COMPONENT' if result.wasSuccessful() else 'FAILED_GPU3_QUALITY_COMPONENT','tests_run':result.testsRun,'errors':len(result.errors),'failures':len(result.failures),'actual_gpu_receipts':receipts,'elapsed_seconds':time.perf_counter()-start,'training_accepted':False,'source_refs':refs}
  (out/'REPORT.json').write_text(json.dumps(report,indent=2)+chr(10))
  if not result.wasSuccessful():return 1
 else:
  from eval.action4_quality_stream import run
  stream_result=run(cfg,out)
  if stream_result.get("status")!="PASS_POINTS_MASKS_DATAFLOW_ROOT_VISUAL_PENDING":
   for reference in refs:
    if authority.ref(R,reference["path"])!=reference:raise ValueError("Quality authority changed during failed/incomplete run")
   return 1
 for ref in refs:
  if authority.ref(R,ref['path'])!=ref:raise ValueError('Quality authority/input changed')
 return 0
