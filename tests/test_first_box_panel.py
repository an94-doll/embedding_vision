import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import unittest
import time
import numpy as np
from unittest.mock import patch
from first_box_panel import append_first_box_panel, PANEL_WIDTH
from view_d405_yolo import Controls, FirstBoxReadout, annotated_camera_frame, HEADER_HEIGHT

ROW=dict(row=0,col=0,detected_count=4,status='rays_only',raw_endpoints_px=[[120.1,200.2],[240.3,201.4]],corrected_endpoints_px=[[119.9,200.],[240.,201.]],midpoint_camera_mm=None)
class PanelTests(unittest.TestCase):
 def test_no_image_resize_or_coordinate_offset(self):
  base=np.full((856,1280,3),71,np.uint8)
  panel,page=append_first_box_panel(base,[ROW])
  self.assertEqual(panel.shape,(856,1700,3));np.testing.assert_array_equal(panel[:,:1280],base)
  self.assertEqual(page,0)
 def test_pages_and_missing_positions(self):
  rows=[dict(ROW,row=i,status='invalid_first_endpoints',raw_endpoints_px=None) for i in range(20)]
  for row in rows: row.pop('raw_endpoints_px')
  canvas=np.zeros((856,1280,3),np.uint8)
  _,page=append_first_box_panel(canvas,rows,99);self.assertEqual(page,3)
  _,page=append_first_box_panel(canvas,[],99,state='STALE');self.assertEqual(page,0)
 def test_recording_omits_sidebar_and_header(self):
  raw=np.full((720,1280,3),55,np.uint8)
  base=np.zeros((856,1280,3),np.uint8);base[HEADER_HEIGHT:]=raw
  panel,_=append_first_box_panel(base,[ROW])
  np.testing.assert_array_equal(annotated_camera_frame(panel,{'image':raw}),raw)
 def test_page_controls_keep_recording_contract(self):
  c=Controls(.25,.5,.7)
  c.key(ord('n'));self.assertEqual(c.panel_page,1)
  c.key(ord('p'));c.key(ord('p'));self.assertEqual(c.panel_page,0)
  c.key(ord(' '),{'active':True,'mode':'annotated'});self.assertFalse(c.paused)
 def test_readout_fresh_frame_confirmation_and_disable(self):
  c=Controls(.25,.5,.7);readout=FirstBoxReadout()
  image=np.zeros((720,1280,3),np.uint8)
  detections=[dict(bbox=[100,170,250,210],keypoints=[[120,200,.9],[240,200,.9]],confidence=.9)]
  intr=dict(width=1280,height=720,fx=600,fy=600,ppx=640,ppy=360,coeffs=[0]*5,model='none')
  packet=dict(image=image,sequence=1,received_at=1.)
  basket=np.array([[[0,0,.9],[1279,0,.9],[1279,719,.9],[0,719,.9]]])
  rows=readout.update(packet,detections,intr,c,basket)
  self.assertEqual(len(rows),15);self.assertEqual(sum(r['detected_count'] for r in rows),0)
  for _ in range(4):
   rows=readout.update(packet,detections,intr,c,basket)
   self.assertEqual(sum(r['detected_count'] for r in rows),0)
  for seq in (2,3,4):
   rows=readout.update(dict(packet,sequence=seq,received_at=1.+seq*.1),detections,intr,c,basket)
  self.assertEqual(len(rows),15);self.assertEqual(rows[0]['status'],'rays_only')
  self.assertEqual(rows[0]['detected_count'],1)
  c.enabled=False;self.assertEqual(readout.update(packet,detections,intr,c),[])

if __name__=='__main__':unittest.main()
