import unittest
import numpy as np
from data.action4_phrase2_visibility import canonical_visibility

class VisibilityTests(unittest.TestCase):
    def setUp(self):
        self.points=np.array([[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],[-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]],float)
    def result(self,t,ray=None):
        T=np.eye(4);T[:3,3]=t
        return canonical_visibility(self.points,T,[True]*8 if ray is None else ray,[True]*8)
    def test_aligned_face(self):
        self.assertEqual(self.result([0,0,5])['visible'],[True]*4+[False]*4)
    def test_three_faces_seven_corners_not_arbitrary_six_cap(self):
        result=self.result([3,3,5]);self.assertEqual(sum(result['visible']),7)
        self.assertFalse(result['visible'][6])
    def test_external_occlusion_and_complete_occlusion(self):
        ray=[True]*8;ray[0]=False
        self.assertEqual(sum(self.result([3,3,5],ray)['visible']),6)
        self.assertEqual(sum(self.result([3,3,5],[False]*8)['visible']),0)
    def test_no_coordinate_or_id_mutation(self):
        before=self.points.copy();self.result([3,3,5]);np.testing.assert_array_equal(before,self.points)
    def test_invalid_camera_inside_rejected(self):
        with self.assertRaises(ValueError):self.result([0,0,0])

if __name__=='__main__':unittest.main()
