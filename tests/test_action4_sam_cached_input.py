import importlib.util
import json
import sys
import threading
import time
import unittest
from pathlib import Path
import subprocess
from types import SimpleNamespace
from model.action4_sam_cached_input import _make_init_wrapper,_validate_loader,_resource_order,install_pinned_cached_input
from model.action4_sam_native_compat import make_init_state_bridge

ROOT=Path("/data/zy/vln/a-test")
RESOURCE=ROOT/"outputs/action4_sam_fullvideo_bootstrap_v001/inputs/frames"
OUT=ROOT/"outputs/action4_sam_cached_input_bridge_v001"

class Reader:
    def __init__(self):self.calls=0
    def as_video_frames(self):
        self.calls+=1;return self,2464,3248
    def __len__(self):return 3
    def __getitem__(self,i):return ("already_normalized",i)

def setup(original_body=None):
    native_calls=[];reader=Reader()
    def loader(resource_path,image_size,offload_video_to_cpu,img_mean=(.5,.5,.5),
               img_std=(.5,.5,.5),async_loading_frames=False,video_loader_type="cv2"):
        native_calls.append(str(resource_path));return "original",2464,3248
    ns={"load_resource_as_video_frames":loader}
    def original(resource_path,offload_video_to_cpu=False,async_loading_frames=False):
        if original_body:return original_body(ns,resource_path,offload_video_to_cpu,async_loading_frames)
        return ns["load_resource_as_video_frames"](resource_path,1008,offload_video_to_cpu,
                  async_loading_frames=async_loading_frames)
    compatible=make_init_state_bridge(original)
    wrapper,usage=_make_init_wrapper(compatible,ns,loader,reader,RESOURCE)
    return wrapper,usage,ns,loader,reader,native_calls

class CachedBoundary(unittest.TestCase):
    def test_lazy_import_does_not_import_torch_or_SAM(self):
        program="import sys;sys.path.insert(0,'/data/zy/vln/a-test/src');import model.action4_sam_cached_input;assert 'torch' not in sys.modules;assert not any(k.startswith('sam3') for k in sys.modules)"
        result=subprocess.run([sys.executable,"-B","-I","-c",program],cwd=ROOT,capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
    def test_cache_returns_original_geometry_order_no_double_normalization(self):
        w,u,ns,loader,reader,calls=setup()
        result=w(str(RESOURCE),offload_video_to_cpu=True,offload_state_to_cpu=False)
        self.assertIs(result[0],reader);self.assertEqual(result[1:],(2464,3248))
        self.assertEqual([result[0][i] for i in range(len(result[0]))],
                         [("already_normalized",0),("already_normalized",1),("already_normalized",2)])
        self.assertEqual(calls,[]);self.assertEqual(reader.calls,1)
        self.assertIs(ns["load_resource_as_video_frames"],loader)
        self.assertTrue(u["last_call"]["module_loader_restored"])
    def test_default_other_path_unchanged(self):
        w,u,ns,loader,reader,calls=setup()
        result=w(str(RESOURCE.parent/"other"),offload_video_to_cpu=False,offload_state_to_cpu=False)
        self.assertEqual(result,("original",2464,3248));self.assertEqual(reader.calls,0)
        self.assertEqual(len(calls),1);self.assertEqual(u["default_calls"],1)
        self.assertIs(ns["load_resource_as_video_frames"],loader)
    def test_mismatched_loader_config_fail_closed(self):
        valid=dict(image_size=1008,offload_video_to_cpu=True,img_mean=(.5,.5,.5),
                   img_std=(.5,.5,.5),async_loading_frames=False,video_loader_type="cv2")
        for key,value in [("image_size",512),("image_size",True),("offload_video_to_cpu",False),
                          ("img_mean",(0.,0.,0.)),("img_std",(1.,1.,1.)),
                          ("async_loading_frames",True),("video_loader_type","torchcodec")]:
            with self.subTest(key=key,value=value):
                bad=dict(valid,**{key:value})
                with self.assertRaises(ValueError):_validate_loader(bad)
        w,u,ns,loader,_,_=setup()
        for kw in [dict(offload_video_to_cpu=False),dict(offload_video_to_cpu=True,async_loading_frames=True),
                   dict(offload_video_to_cpu=True,offload_state_to_cpu=True)]:
            with self.assertRaises(ValueError):w(str(RESOURCE),**kw)
            self.assertIs(ns["load_resource_as_video_frames"],loader)
    def test_exception_restores_global_loader(self):
        def fail(ns,path,offload,async_):
            ns["load_resource_as_video_frames"](path,1008,offload)
            raise RuntimeError("simulated native-init failure")
        w,u,ns,loader,reader,_=setup(fail)
        with self.assertRaisesRegex(RuntimeError,"simulated"):w(str(RESOURCE),offload_video_to_cpu=True)
        self.assertEqual(reader.calls,1);self.assertIs(ns["load_resource_as_video_frames"],loader)
        self.assertFalse(u["last_call"]["init_returned"])
    def test_other_thread_same_resource_uses_original(self):
        other=[];errors=[]
        def original(ns,path,offload,async_):
            def concurrent():
                try:other.append(ns["load_resource_as_video_frames"](path,1008,True))
                except BaseException as e:errors.append(e)
            t=threading.Thread(target=concurrent);t.start();t.join(2)
            self.assertFalse(t.is_alive())
            return ns["load_resource_as_video_frames"](path,1008,offload)
        w,u,ns,loader,reader,calls=setup(original)
        result=w(str(RESOURCE),offload_video_to_cpu=True)
        self.assertFalse(errors);self.assertEqual(other,[("original",2464,3248)])
        self.assertIs(result[0],reader);self.assertEqual(len(calls),1)
        self.assertIs(ns["load_resource_as_video_frames"],loader)
    def test_concurrent_own_init_calls_are_serialized(self):
        active=0;maximum=0;results=[];errors=[];guard=threading.Lock()
        def original(ns,path,offload,async_):
            nonlocal active,maximum
            with guard:active+=1;maximum=max(maximum,active)
            time.sleep(.04)
            result=ns["load_resource_as_video_frames"](path,1008,offload)
            with guard:active-=1
            return result
        w,u,ns,loader,reader,_=setup(original)
        def call():
            try:results.append(w(str(RESOURCE),offload_video_to_cpu=True))
            except BaseException as e:errors.append(e)
        threads=[threading.Thread(target=call) for _ in range(2)]
        for t in threads:t.start()
        for t in threads:t.join(3);self.assertFalse(t.is_alive())
        self.assertFalse(errors);self.assertEqual(maximum,1);self.assertEqual(len(results),2)
        self.assertEqual(reader.calls,2);self.assertIs(ns["load_resource_as_video_frames"],loader)
    def test_no_boundary_hit_and_foreign_swap_rejected(self):
        w,u,ns,loader,_,_=setup(lambda ns,p,o,a:None)
        with self.assertRaisesRegex(RuntimeError,"did not consume"):w(str(RESOURCE),offload_video_to_cpu=True)
        self.assertIs(ns["load_resource_as_video_frames"],loader)
        ns["load_resource_as_video_frames"]=lambda *a,**k:None
        with self.assertRaisesRegex(RuntimeError,"already changed"):w(str(RESOURCE),offload_video_to_cpu=True)
    def test_namespace_interference_is_detected_and_restored(self):
        def original(ns,path,offload,async_):
            result=ns["load_resource_as_video_frames"](path,1008,offload)
            ns["load_resource_as_video_frames"]=lambda *a,**k:None
            return result
        w,u,ns,loader,_,_=setup(original)
        with self.assertRaisesRegex(RuntimeError,"during owned"):w(str(RESOURCE),offload_video_to_cpu=True)
        self.assertIs(ns["load_resource_as_video_frames"],loader)
        self.assertTrue(u["last_call"]["namespace_interference_detected"])

    def test_bad_spec_or_native_class_cannot_install(self):
        with self.assertRaises(ValueError):
            install_pinned_cached_input(SimpleNamespace(model=object()),cache_spec={},
                official_root=RESOURCE,expected_source_sha256="0"*64,snapshot_manifest=RESOURCE)
        with self.assertRaises(ValueError):
            install_pinned_cached_input(SimpleNamespace(model=object()),
                cache_spec={"resource_path":str(RESOURCE),"manifest":{},"contract":{}},
                official_root=ROOT/"outputs/action4_sam_joint_full2850_v001/source/upstream/sam3",
                expected_source_sha256="65e1cea27ee243dad228fb05967a8be92c90ef304665ada76930b899aa034d24",
                snapshot_manifest=ROOT/"outputs/action4_sam_joint_full2850_v001/source/SNAPSHOT_MANIFEST.json")

class ActualCacheCPU(unittest.TestCase):
    @unittest.skipUnless(__import__("os").environ.get("ACTION4_REAL_CACHE_BRIDGE")=="1","explicit real cached tensor CPU test")
    def test_real_reader_native_nestedtensor_and_copies(self):
        import hashlib,numpy as np,torch
        from data.action4_sam_input_cache import SamInputTensorCache,make_contract
        manifest=ROOT/"outputs/action4_sam_input_cache_full2850_v001/cache/CACHE.json"
        expected="5a23840731f7b801703cec1bb7105cc3d16830a68540f59e6d2bd1a921d2ad29"
        self.assertEqual(hashlib.sha256(manifest.read_bytes()).hexdigest(),expected)
        payload=json.loads(manifest.read_text());contract=payload["contract"]
        independently_bound=make_contract(contract["official_source_root"],contract["region"]["path"],range(2850))
        self.assertEqual(contract,independently_bound)
        reader=SamInputTensorCache(manifest,expected_sha256=expected,expected_contract=independently_bound)
        self.assertEqual(reader.as_video_frames()[1:],(2464,3248));self.assertEqual(len(reader),2850)
        _resource_order(RESOURCE,independently_bound)
        source=ROOT/contract["official_source_root"]/"sam3/model/data_misc.py"
        spec=importlib.util.spec_from_file_location("action4_cache_test_native_data_misc",source)
        module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
        nested=module.NestedTensor(reader,None)
        self.assertEqual(len(nested),2850)
        # Execute only the pinned pure traversal helper, with any Tensor.to
        # blocked: this demonstrates the native NestedTensor/reader stays lazy.
        import ast,dataclasses,collections.abc
        from unittest.mock import patch
        tracking=ROOT/contract["official_source_root"]/"sam3/model/sam3_multiplex_tracking.py"
        tree=ast.parse(tracking.read_text())
        node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="recursive_to")
        env={"torch":torch,"np":np,"Mapping":collections.abc.Mapping,"Sequence":collections.abc.Sequence,
             "is_dataclass":dataclasses.is_dataclass,"fields":dataclasses.fields}
        exec(compile(ast.Module(body=[node],type_ignores=[]),str(tracking),"exec"),env)
        with patch.object(torch.Tensor,"to",side_effect=AssertionError("unexpected eager tensor move")):
            self.assertIs(env["recursive_to"](nested,"cuda",non_blocking=True),nested)

        self.assertFalse(torch.cuda.is_initialized())
        for index in [0,159,2849]:
            value=nested[index]
            expected_array=np.load(manifest.parent/f"frame_{index:06d}.npy",allow_pickle=False)
            self.assertEqual(value.dtype,torch.float16);self.assertEqual(tuple(value.shape),(3,1008,1008))
            self.assertTrue(np.array_equal(value.numpy(),expected_array))
            value.zero_();fresh=nested[index]
            self.assertTrue(np.array_equal(fresh.numpy(),expected_array))
        self.assertFalse(torch.cuda.is_initialized())

if __name__=="__main__":unittest.main()
