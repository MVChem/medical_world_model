"""Tests for scientific validity and independent input paths."""
import json
import unittest
import numpy as np
import torch
from common import DEFAULT_RUN,read_rows
from direction import parse
from features import align_hidden,downsample
from heads import DenseHead,GroundingHead,box_iou,segmentation_loss
from train import objective

torch.set_num_threads(4)

class ContractTests(unittest.TestCase):
    def test_spatial_and_channel_sampling(self):
        # A marked spatial grid must preserve both axes, not alias flattened rows.
        for side,width in [(8,1024),(16,2560),(24,4096),(16,5120),(16,5376)]:
            marker=torch.arange(side*side).float()[:,None].expand(-1,width)
            y=align_hidden(marker)
            self.assertEqual(tuple(y.shape),(64,1024))
            grid=y[:,0].reshape(8,8)
            self.assertTrue((grid[:,1:]>grid[:,:-1]).all())
            self.assertTrue((grid[1:]>grid[:-1]).all())
        with self.assertRaises(ValueError):align_hidden(torch.zeros(32,1024))

    def test_decoder_paths_and_gradients(self):
        # Both inputs must affect a trainable head; feature variants accept no pixels.
        for task in ['segmentation','sr']:
            for variant in ['image','vjepa','vjepa_adapter']:
                torch.manual_seed(7)
                m=DenseHead(task,variant)
                h=torch.randn(1,64,1024,requires_grad=True)
                x=torch.randn(1,576,768,requires_grad=True) if variant!='image' else torch.rand(
                    1,1,128 if task=='sr' else 256,128 if task=='sr' else 256,requires_grad=True)
                y=m(x,h)
                self.assertEqual(tuple(y.shape),(1,1,512,512) if task=='sr' else (1,3,256,256))
                y.square().mean().backward()
                self.assertGreater(float(h.grad.abs().sum()),0)
                self.assertGreater(float(x.grad.abs().sum()),0)
                self.assertFalse(hasattr(m,'stem') if variant!='image' else False)

    def test_grounding_geometry(self):
        a=torch.tensor([[0.,0.,1.,1.],[0.,0.,.5,.5],[0.,0.,.25,.25]])
        b=torch.tensor([[0.,0.,1.,1.],[.5,.5,1.,1.],[0.,0.,.5,.5]])
        torch.testing.assert_close(box_iou(a,b),torch.tensor([1.,0.,.25]))
        m=GroundingHead(29)
        y=m(torch.rand(2,1,256,256),torch.rand(2,64,1024),torch.tensor([0,1]))
        self.assertTrue(((y>=0)&(y<=1)).all())
        self.assertTrue((y[:,2:]>y[:,:2]).all())

    def test_padding_does_not_change_segmentation_loss(self):
        logits=torch.randn(1,3,16,16);target=torch.rand(1,3,16,16);mask=torch.zeros(1,1,16,16)
        mask[:,:,4:12,4:12]=1
        expected=segmentation_loss(logits,target,mask)
        altered=logits.clone();altered[:,:,:4]=100
        torch.testing.assert_close(expected,segmentation_loss(altered,target,mask))

    def test_missing_direction_is_not_stable(self):
        self.assertIsNone(parse('no change'))
        self.assertIsNone(parse('[]'))
        self.assertIsNone(parse(json.dumps([['stable']*3]*5)))
        self.assertEqual(parse(json.dumps([['unknown']*3]*6))[0][0],'unknown')

    def test_microbatch_preserves_sample_weights(self):
        mask=torch.zeros(2,1,16,16);mask[0,:,:8,:8]=1;mask[1]=1
        for task,c in [('segmentation',3),('sr',1)]:
            p=torch.randn(2,c,16,16);t=torch.rand(2,c,16,16)
            full=objective(task,p,t,mask)
            small=sum(objective(task,p[i:i+1],t[i:i+1],mask[i:i+1]) for i in range(2))/2
            torch.testing.assert_close(full,small)

    @unittest.skipUnless((DEFAULT_RUN/'data/manifest.json').exists(),'cohort not prepared')
    def test_frozen_patient_splits_and_exact_lr(self):
        data=DEFAULT_RUN/'data';rows=read_rows(data/'observations.jsonl')
        sets=[{r['subject_id'] for r in rows if r['split']==s} for s in ['train','validate','test','human_test']]
        for a in range(len(sets)):
            for b in range(a):self.assertFalse(sets[a]&sets[b])
        x=np.load(data/'images.npy',mmap_mode='r');lr=np.load(data/'lr_images.npy',mmap_mode='r')
        ids=np.linspace(0,len(rows)-1,17).astype(int)
        expected=(downsample(torch.from_numpy(np.array(x[ids]))[:,None].float()/255)[:,0]*255).round().byte().numpy()
        np.testing.assert_array_equal(lr[ids],expected)
        # Annotation folder image-left must be scored against anatomical right lung.
        masks=np.load(data/'human_masks.npy',mmap_mode='r')
        xx=np.arange(256)[None,None,None,:]
        cx=(masks*xx).sum((2,3))/masks.sum((2,3))
        self.assertTrue((cx[:,0]<cx[:,1]).all())

if __name__=='__main__':unittest.main()
