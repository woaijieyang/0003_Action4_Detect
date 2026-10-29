import importlib,importlib.util,unittest,numpy as np
class ComponentsTests(unittest.TestCase):
 def fn(self):
  self.assertIsNotNone(importlib.util.find_spec('model.action4_body_support_components'),'body-window component selector missing')
  return importlib.import_module('model.action4_body_support_components').coherent_extent
 def test_far_component_does_not_pull_extent(self):
  m=np.zeros((500,500),bool);m[100:160,100:180]=True;m[350:400,350:400]=True
  b,p=self.fn()(m,[95,95,185,165]);self.assertEqual(b,[100,100,180,160]);self.assertEqual(p['retained_components'],1)
 def test_near_disconnected_visible_part_preserved(self):
  m=np.zeros((250,250),bool);m[100:140,100:140]=True;m[100:110,145:150]=True
  b,p=self.fn()(m,[95,95,155,145]);self.assertEqual(b,[100,100,150,140]);self.assertEqual(p['retained_components'],2)
 def test_larger_remote_component_cannot_win_by_size(self):
  m=np.zeros((500,500),bool);m[100:150,100:150]=True;m[250:450,250:450]=True
  b,p=self.fn()(m,[90,90,160,160]);self.assertEqual(b,[100,100,150,150])
 def test_raw_mask_unchanged_and_no_support_returns_none(self):
  m=np.zeros((200,200),bool);m[100:150,100:150]=True;before=m.copy()
  b,p=self.fn()(m,[0,0,20,20]);self.assertIsNone(b);self.assertTrue(np.array_equal(m,before))
 def test_connected_attachment_is_explicitly_not_separated(self):
  m=np.zeros((200,250),bool);m[80:140,80:140]=True;m[100:110,140:230]=True
  b,p=self.fn()(m,[75,75,145,145]);self.assertEqual(b[2],230);self.assertFalse(p['semantic_body_certified'])
 def test_invalid_input_fails(self):
  f=self.fn()
  with self.assertRaises(ValueError):f(np.zeros((2,2),int),[0,0,1,1])
  with self.assertRaises(ValueError):f(np.zeros((2,2),bool),[0,0,0,1])
if __name__=='__main__':unittest.main()
