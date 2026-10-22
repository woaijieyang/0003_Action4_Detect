"""Amodal crop invariants; renderer integration is verified separately on GPU."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import numpy as np
from PIL import Image
from data import action4_amodal as am
from data import action4_synthesis as synth


class AmodalContracts(unittest.TestCase):
    def setUp(self):
        self.points = np.array([[x,y,z] for x in (30.,50.) for y in (35.,65.) for z in (1.,2.)])

    def test_legacy_default_and_explicit_modes(self):
        self.assertEqual(am.validate_settings({}), 'full')
        self.assertEqual(am.validate_settings({'amodal_mode':'crop','amodal_margin_px':16}), 'crop')
        for mode in ('auto',None,17):
            with self.assertRaises(ValueError):am.validate_settings({'amodal_mode':mode})
        for margin in (15,17,16.,True):
            with self.assertRaises(ValueError):am.validate_settings({'amodal_margin_px':margin})

    def test_integer_expansion_and_clamping(self):
        self.assertEqual(am.crop_bounds(self.points,100,100),(14,19,66,81))
        p=self.points.copy();p[:,0]-=29
        self.assertEqual(am.crop_bounds(p,100,100),(0,19,37,81))

    def test_corner_shape_depth_canvas_and_dimensions(self):
        for points in (self.points[:7],np.full((8,3),np.nan),self.points*np.array([1,1,-1]),self.points+np.array([100,0,0])):
            with self.assertRaises(ValueError):am.crop_bounds(points,100,100)
        for dims in ((0,10),(-1,10),(1.5,10),(True,10)):
            with self.assertRaises(ValueError):am.crop_bounds(self.points,*dims)

    def test_all_parts_coverage_including_detached_child(self):
        b=am.crop_bounds(self.points,100,100)
        am.validate_geometry_coverage(self.points,b,100,100)
        # A disconnected child is wholly outside crop, so a border-only test misses it.
        with self.assertRaisesRegex(ValueError,'safe crop'):
            am.validate_geometry_coverage(np.vstack([self.points,[85,50,1]]),b,100,100)
        with self.assertRaisesRegex(ValueError,'Expanded canvas'):
            am.validate_geometry_coverage([[101,50,1]],b,100,100)
        with self.assertRaisesRegex(ValueError,'behind'):
            am.validate_geometry_coverage([[40,50,-1]],b,100,100)
        with self.assertRaises(ValueError):am.validate_geometry_coverage([],b,100,100)

    def test_bottom_left_to_top_left_exact_padding(self):
        crop=np.zeros((5,6),bool);crop[1,2]=True;crop[3,4]=True
        full=am.pad_mask(crop,(3,2,9,7),12,11)
        expected=np.zeros((11,12),bool);expected[5,5]=True;expected[7,7]=True
        np.testing.assert_array_equal(full,expected)

    def test_rendered_dimensions_and_boundary_rejected(self):
        with self.assertRaises(ValueError):am.pad_mask(np.zeros((3,3),bool),(0,0,4,4),10,10)
        crop=np.zeros((4,4),bool);crop[0,2]=True
        with self.assertRaisesRegex(ValueError,'boundary'):am.pad_mask(crop,(1,1,5,5),10,10)
        with self.assertRaises(ValueError):am.pad_mask(np.zeros((4,4),np.uint8),(1,1,5,5),10,10)
        with self.assertRaises(ValueError):am.pad_mask(np.zeros((4,4),bool),(-1,1,3,5),10,10)

    def test_empty_mask_and_outside_original_frame_are_retained(self):
        empty=am.pad_mask(np.zeros((4,4),bool),(1,1,5,5),30,30)
        self.assertEqual(int(empty.sum()),0)
        crop=np.zeros((4,4),bool);crop[1:3,1:3]=True
        full=am.pad_mask(crop,(1,1,5,5),30,30)
        self.assertEqual(int(full.sum()),4)
        self.assertEqual(int(full[10:20,10:20].sum()),0)

    def test_gray8_png_lossless_roundtrip_and_no_overwrite(self):
        with tempfile.TemporaryDirectory(dir=synth.ROOT/'outputs',prefix='amodal_test_') as folder:
            path=Path(folder)/'mask.png';a=np.zeros((19,31),bool);a[3:8,2:19]=True;a[15,27]=True
            am.write_binary_png(path,a)
            with Image.open(path) as image:
                self.assertEqual(image.mode,'L');self.assertEqual(image.size,(31,19));raw=np.asarray(image)
            np.testing.assert_array_equal(raw,a.astype('uint8')*255)
            with self.assertRaises(FileExistsError):am.write_binary_png(path,a)
            with self.assertRaises(ValueError):am.write_binary_png(Path(folder)/'bad.png',a.astype(float))

    def test_finally_restores_all_border_and_output_nodes(self):
        render=SimpleNamespace(use_border=False,use_crop_to_border=False,border_min_x=.2,border_max_x=.8,border_min_y=.3,border_max_y=.9)
        before=vars(render).copy();scene=SimpleNamespace(render=render)
        nodes=[SimpleNamespace(base_path='original'+str(i),file_slots=[SimpleNamespace(path='slot'+str(i))]) for i in range(2)]
        with self.assertRaisesRegex(RuntimeError,'render failed'):
            with am.temporary_border(scene,nodes,'new',(14,19,66,81),100,100):
                self.assertTrue(render.use_border);self.assertEqual(nodes[0].base_path,'new');raise RuntimeError('render failed')
        self.assertEqual(vars(render),before)
        for i,n in enumerate(nodes):self.assertEqual((n.base_path,n.file_slots[0].path),('original'+str(i),'slot'+str(i)))

    def test_float32_border_preserves_requested_integer_edges(self):
        for size in (615,2464,7392,9744):
            for edge in (0,1,size//3,size-1,size):
                actual=int(float(np.float32(min(1.,(edge+.1)/size)))*size)
                self.assertEqual(actual,edge)

    def test_helper_identity_is_frozen_for_both_modes(self):
        recipe=json.loads((synth.ROOT/'configs/data/action4_synthesis_r3_v004.json').read_text())
        for mode in ('full','crop'):
            recipe['render']['amodal_mode']=mode
            identities=synth._asset_identities(recipe)
            self.assertEqual(identities['src/data/action4_amodal.py'],synth._hash(synth.ROOT/'src/data/action4_amodal.py'))

    def test_recipe_rejects_invalid_mode(self):
        recipe=json.loads((synth.ROOT/'configs/data/action4_synthesis_r3_v004.json').read_text());recipe['render']['amodal_mode']='bad'
        with self.assertRaisesRegex(ValueError,'amodal_mode'):synth.validate_recipe(recipe)


if __name__=='__main__':unittest.main()