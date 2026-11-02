import unittest
from types import SimpleNamespace as NS
import numpy as np
import cv2
from scripts.first_box import endpoint_geometry, locate_first_boxes, draw_first_boxes

C=dict(width=1280,height=720,fx=600.,fy=600.,ppx=640.,ppy=360.,model='distortion.none',coeffs=[0.]*5)
P=np.array([[[580.,340.,.9],[700.,340.,.9]],[[580.,400.,.9],[700.,400.,.9]]])
B=np.array([[570.,330.,710.,350.],[570.,390.,710.,410.]])
def grid(indices=(1,0),unassigned=()):
 return NS(cells=[NS(row=0,col=0,indices=indices,count=len(indices),bounds=(570,330,710,410))],unassigned_indices=unassigned)

class FirstBoxTests(unittest.TestCase):
 def test_plane_and_width(self):
  r=endpoint_geometry(P[0,:,:2],C,(720,1280),dict(normal=[0,0,1],offset_mm=150))
  np.testing.assert_allclose(r['midpoint_camera_mm'],[0,-5,150]); self.assertAlmostEqual(r['width_mm'],30)
 def test_no_plane(self):
  r=endpoint_geometry(P[0,:,:2],C,(720,1280)); self.assertEqual(r['status'],'rays_only'); self.assertIsNone(r['midpoint_camera_mm'])
 def test_brown_roundtrip(self):
  coeff=[.1,-.03,.001,-.002,.01]; K=np.array([[600.,0,640],[0,600,360],[0,0,1.]])
  xyz=np.array([[-.2,-.1,1.],[.3,.2,1.]])
  pixels=cv2.projectPoints(xyz,np.zeros(3),np.zeros(3),K,np.array(coeff))[0].reshape(2,2)
  r=endpoint_geometry(pixels,dict(C,model='brown_conrady',coeffs=coeff),(720,1280))
  np.testing.assert_allclose(r['rays_camera'],xyz,atol=1e-8)
 def test_bad_geometry(self):
  for plane in [dict(normal=[0,0,1],offset_mm=-150),dict(normal=[0,0,0],offset_mm=1)]:
   with self.assertRaises(ValueError): endpoint_geometry(P[0,:,:2],C,(720,1280),plane)
  with self.assertRaises(ValueError): endpoint_geometry(P[0,:,:2],C,(360,640))
  with self.assertRaises(ValueError): endpoint_geometry(P[0,:,:2],dict(C,model='modified_brown_conrady'),(720,1280))
 def test_select_top_and_not_next_valid(self):
  r=locate_first_boxes(grid(),B,P,C,(720,1280))[0];self.assertEqual(r['detection_index'],0)
  p=P.copy();p[0,0,2]=.1
  r=locate_first_boxes(grid(),B,p,C,(720,1280))[0];self.assertEqual(r['status'],'invalid_first_endpoints');self.assertEqual(r['detection_index'],0)
 def test_incomplete_and_empty(self):
  self.assertEqual(locate_first_boxes(grid(indices=(1,),unassigned=(0,)),B,P,C,(720,1280))[0]['selection_status'],'incomplete_observation')
  self.assertEqual(locate_first_boxes(grid(),B,P,C,(720,1280),lost_count=1)[0]['temporarily_lost_tracks'],1)
  self.assertEqual(locate_first_boxes(grid(()),B,P,C,(720,1280))[0]['status'],'empty')
 def test_direction_and_overlay(self):
  r=locate_first_boxes(grid(),B,P,C,(720,1280),angle_deg=180)
  self.assertEqual(r[0]['detection_index'],1)
  canvas=np.zeros((872,1280,3),np.uint8);draw_first_boxes(canvas,r,152);self.assertTrue(canvas[552,640].any())
  cv2.imwrite('/tmp/d405_firstbox/preview.png',canvas)

if __name__=='__main__': unittest.main()
