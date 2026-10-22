"""Blender CPU regression tests; run with Blender's bundled Python."""
import unittest
import numpy as np
from data.action4_template_collision import Geometry, check_pair, point_in_closed, triangle_contact

V = np.array([[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
              [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]], float)
F = np.array([[0,2,1],[0,3,2],[4,5,6],[4,6,7],[0,1,5],[0,5,4],
              [1,2,6],[1,6,5],[2,3,7],[2,7,6],[3,0,4],[3,4,7]])

def box(scale=1., shift=(0,0,0)):
    return Geometry.create(V*scale+shift,F)

class CollisionTests(unittest.TestCase):
    def test_separated_boxes(self):
        self.assertEqual(check_pair(box(),box(shift=(3,0,0)))['status'],'CLEAR_AABB')

    def test_intersecting_boxes(self):
        self.assertEqual(check_pair(box(),box(shift=(1.5,.1,.2)))['status'],'PROHIBITED_SURFACE_CONTACT')

    def test_containment_without_surface_hits(self):
        self.assertEqual(check_pair(box(),box(.1))['status'],'PROHIBITED_CONTAINMENT')
        self.assertEqual(check_pair(box(.1),box())['status'],'PROHIBITED_CONTAINMENT')

    def test_disconnected_component_inside(self):
        source=Geometry.create(np.concatenate([V*.1+[3,0,0],V*.1]),np.concatenate([F,F+8]))
        result=check_pair(source,box())
        self.assertEqual(result['status'],'PROHIBITED_CONTAINMENT')
        self.assertEqual(len(source.representatives),2)

    def test_open_container_never_zero_hits_pass(self):
        open_box=Geometry.create(V,F[:-2])
        result=check_pair(box(.1),open_box)
        self.assertEqual(result['status'],'UNKNOWN_CONTAINMENT')

    def test_sheet_inside_closed_solid(self):
        sheet=Geometry.create(np.array([[-.1,-.1,0],[.1,-.1,0],[0,.1,0]]),[[0,1,2]])
        self.assertEqual(check_pair(sheet,box())['status'],'PROHIBITED_CONTAINMENT')

    def test_coplanar_disjoint_triangles_with_overlapping_aabb(self):
        a=np.array([[0,0,0],[1,0,0],[0,1,0]],float)
        b=np.array([[1,1,0],[.6,1,0],[1,.6,0]],float)
        self.assertFalse(triangle_contact(a,b))
        self.assertTrue(triangle_contact(a,a+np.array([.01,.01,0])))

    def test_ray_inside_outside_boundary(self):
        self.assertTrue(point_in_closed(np.zeros(3),V,F))
        self.assertFalse(point_in_closed(np.array([3,0,0]),V,F))
        self.assertIsNone(point_in_closed(np.array([1,0,0]),V,F))

    def test_hollow_closed_ring_does_not_fill_hole(self):
        vertices=[]
        for i in range(24):
            u=2*np.pi*i/24
            for j in range(12):
                v=2*np.pi*j/12
                vertices.append([(2+.4*np.cos(v))*np.cos(u),(2+.4*np.cos(v))*np.sin(u),.4*np.sin(v)])
        faces=[]
        for i in range(24):
            for j in range(12):
                a=i*12+j;b=((i+1)%24)*12+j;c=((i+1)%24)*12+(j+1)%12;d=i*12+(j+1)%12
                faces.extend([[a,b,c],[a,c,d]])
        ring=Geometry.create(vertices,faces)
        self.assertTrue(ring.closed)
        self.assertEqual(check_pair(box(.1),ring)['status'],'CLEAR_SURFACE_AND_CONTAINMENT')

    def test_actual_curve_and_modifier_evaluation(self):
        import bpy
        from data.action4_modular_assembly import world_geometry
        from data.action4_template_collision import mesh_signature, verify_evaluation_supported
        curve=bpy.data.curves.new('collision_test_curve','CURVE')
        curve.dimensions='3D';curve.bevel_depth=.05;curve.bevel_resolution=2
        spline=curve.splines.new('POLY');spline.points.add(1)
        spline.points[0].co=(0,0,0,1);spline.points[1].co=(0,0,1,1)
        obj=bpy.data.objects.new('collision_test_curve',curve);bpy.context.scene.collection.objects.link(obj)
        bpy.context.view_layer.update()
        vertices,faces=world_geometry(obj)
        self.assertGreater(len(faces),0)
        self.assertIsNone(mesh_signature(obj))
        self.assertGreater(Geometry.create(vertices,faces).boundary_edges,0)
        bpy.data.objects.remove(obj,do_unlink=True)
        bpy.ops.mesh.primitive_cube_add()
        obj=bpy.context.object
        modifier=obj.modifiers.new('bevel','BEVEL');modifier.width=.1;modifier.segments=2
        bpy.context.view_layer.update()
        vertices,faces=world_geometry(obj)
        self.assertGreater(len(faces),12)
        self.assertIsNone(mesh_signature(obj))
        verify_evaluation_supported(obj)
        modifier.show_render=False
        with self.assertRaises(ValueError):verify_evaluation_supported(obj)
        bpy.data.objects.remove(obj,do_unlink=True)

    def _active_patch_checker(self, patch):
        # Exercise the real per-sample patch path without loading full templates.
        from types import SimpleNamespace
        from data.action4_template_collision import TemplateCollisionChecker
        checker=TemplateCollisionChecker.__new__(TemplateCollisionChecker)
        checker.runtime=SimpleNamespace(scene=None,patches={'left':patch},verify=lambda scene:None)
        checker.excluded_baseline_hidden=[]
        checker.cache={};checker.groups={}
        return checker

    def test_active_patch_render_only_modifier_fails_closed(self):
        import bpy
        bpy.ops.mesh.primitive_plane_add()
        patch=bpy.context.object
        try:
            modifier=patch.modifiers.new('render_only_thickness','SOLIDIFY')
            modifier.thickness=10.
            modifier.show_render=True
            modifier.show_viewport=False
            bpy.context.view_layer.update()
            report=self._active_patch_checker(patch).check()
            self.assertFalse(report['accepted'])
            self.assertEqual(report['status'],'REJECT_UNRESOLVED')
            self.assertIn('Render/viewport modifier visibility differs',str(report['unknown']))
        finally:
            bpy.data.objects.remove(patch,do_unlink=True)

    def test_active_patch_curve_type_fails_closed(self):
        import bpy
        curve=bpy.data.curves.new('collision_patch_wrong_type','CURVE')
        curve.dimensions='2D';curve.fill_mode='BOTH'
        spline=curve.splines.new('POLY');spline.points.add(3)
        for point,coordinate in zip(spline.points,[(0,0,0,1),(1,0,0,1),(1,1,0,1),(0,1,0,1)]):
            point.co=coordinate
        spline.use_cyclic_u=True
        patch=bpy.data.objects.new('collision_patch_wrong_type',curve)
        bpy.context.scene.collection.objects.link(patch)
        try:
            bpy.context.view_layer.update()
            report=self._active_patch_checker(patch).check()
            self.assertFalse(report['accepted'])
            self.assertEqual(report['status'],'REJECT_UNRESOLVED')
            self.assertIn('Occluder must remain a mesh sheet',str(report['unknown']))
        finally:
            bpy.data.objects.remove(patch,do_unlink=True)

    def test_nonfinite_and_degenerate_fail(self):
        with self.assertRaises(ValueError):Geometry.create(V*np.nan,F)
        with self.assertRaises(ValueError):Geometry.create(np.zeros((3,3)),[[0,1,2]])

if __name__ == '__main__':
    unittest.main()

