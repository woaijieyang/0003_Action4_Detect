"""Explicit RGB denoiser selection; never fall back to an unavailable enum."""
import copy
import json
from types import SimpleNamespace
import unittest
from data import action4_synthesis as synth

class DynamicCycles:
    def __init__(self, allowed, ignore_assignment=False):
        self.allowed=allowed;self._denoiser='OPENIMAGEDENOISE';self.use_denoising=False;self.ignore=ignore_assignment
    @property
    def denoiser(self):return self._denoiser
    @denoiser.setter
    def denoiser(self,value):
        if value not in self.allowed:raise TypeError('enum not found')
        if not self.ignore:self._denoiser=value

class DenoiserContracts(unittest.TestCase):
    def test_default_oidn_preserved_for_cpu_and_gpu(self):
        for device in ('CPU','CUDA','OPTIX'):
            self.assertEqual(synth._denoiser_name({'device':device}),'OPENIMAGEDENOISE')
            scene=SimpleNamespace(cycles=DynamicCycles(['OPENIMAGEDENOISE']))
            self.assertEqual(synth._set_denoiser(scene,{'device':device}),'OPENIMAGEDENOISE')
            self.assertTrue(scene.cycles.use_denoising)
    def test_optix_requires_gpu_backend(self):
        for device in ('CPU',None,'HIP'):
            with self.assertRaisesRegex(ValueError,'requires'):synth._denoiser_name({'device':device,'denoiser':'OPTIX'})
        for device in ('CUDA','OPTIX'):
            scene=SimpleNamespace(cycles=DynamicCycles(['OPTIX']))
            self.assertEqual(synth._set_denoiser(scene,{'device':device,'denoiser':'OPTIX'}),'OPTIX')
    def test_unknown_and_null_names_rejected(self):
        for value in ('AUTO','OIDN','optix',None,17):
            with self.assertRaises(ValueError):synth._denoiser_name({'device':'CUDA','denoiser':value})
    def test_unavailable_dynamic_enum_fails_without_fallback(self):
        scene=SimpleNamespace(cycles=DynamicCycles(['OPENIMAGEDENOISE']))
        with self.assertRaisesRegex(RuntimeError,'unavailable'):synth._set_denoiser(scene,{'device':'CUDA','denoiser':'OPTIX'})
        self.assertEqual(scene.cycles.denoiser,'OPENIMAGEDENOISE');self.assertFalse(scene.cycles.use_denoising)
    def test_silent_assignment_mismatch_rejected(self):
        scene=SimpleNamespace(cycles=DynamicCycles(['OPENIMAGEDENOISE','OPTIX'],ignore_assignment=True))
        with self.assertRaisesRegex(RuntimeError,'differs'):synth._set_denoiser(scene,{'device':'CUDA','denoiser':'OPTIX'})
        self.assertFalse(scene.cycles.use_denoising)
    def test_recipe_validation_and_no_mutation(self):
        recipe=json.loads((synth.ROOT/'configs/data/action4_server_native64_preflight_v009_cuda.json').read_text())
        for name in ('OPENIMAGEDENOISE','OPTIX'):
            candidate=copy.deepcopy(recipe);candidate['render']['denoiser']=name;before=copy.deepcopy(candidate)
            synth.validate_recipe(candidate);self.assertEqual(candidate,before)
        recipe['render']['denoiser']='OPTIX';recipe['render']['device']='CPU'
        with self.assertRaisesRegex(ValueError,'requires'):synth.validate_recipe(recipe)

if __name__=='__main__':unittest.main()