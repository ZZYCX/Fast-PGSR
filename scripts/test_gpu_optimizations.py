"""Numerical/device regression for GPU optimizations; no rasterizer extension needed."""
import ast
import copy
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
from torch import nn

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from utils.training_cache import pixel_grid,sample_valid_indices,nearest_view_indices
from utils.graphics_utils import getWorld2View2,getProjectionMatrix,fov2focal,patch_offsets
from utils.general_utils import PILtoTorch


def load_path(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


class VideoRegression(unittest.TestCase):
    def test_failed_cuda_attempt_does_not_contaminate_cpu_frames(self):
        decoder=load_path('decoder',ROOT/'scripts/preprocess/extract_frames.py')
        with tempfile.TemporaryDirectory() as temp:
            output=Path(temp)/'frames';calls=[]
            def run(command):
                calls.append(command)
                pattern=next(p for p in command if 'frame_%06d.jpg' in p)
                if '-hwaccel' in command:
                    Path(pattern.replace('%06d','000999')).write_bytes(b'failed GPU fragment')
                    return types.SimpleNamespace(returncode=1)
                Path(pattern.replace('%06d','000001')).write_bytes(b'CPU frame')
                return types.SimpleNamespace(returncode=0)
            with patch.object(subprocess,'run',run):
                self.assertEqual(decoder.extract('soybean video.mp4',output),1)
            self.assertEqual(len(calls),2)
            self.assertIn('cuda',calls[0]);self.assertNotIn('-hwaccel',calls[1])
            self.assertEqual([p.name for p in output.iterdir()],['frame_000001.jpg'])

    def test_cuda_success_does_not_decode_twice(self):
        decoder=load_path('decoder',ROOT/'scripts/preprocess/extract_frames.py')
        with tempfile.TemporaryDirectory() as temp:
            def run(command):
                pattern=next(p for p in command if 'frame_%06d.jpg' in p)
                Path(pattern.replace('%06d','000001')).touch()
                return types.SimpleNamespace(returncode=0)
            with patch.object(subprocess,'run',side_effect=run) as mock:
                self.assertEqual(decoder.extract('video.mp4',Path(temp)/'frames'),1)
                self.assertEqual(mock.call_count,1)
                self.assertEqual(mock.call_args[0][0][mock.call_args[0][0].index('-hwaccel_device')+1],'0')

    def test_colmap_cuda_capability_check(self):
        converter=load_path('converter',ROOT/'scripts/preprocess/convert.py')
        for text,valid in [('COLMAP 3.9.1 (with CUDA)',True),('COLMAP 3.9.1 (without CUDA)',False)]:
            with patch.object(subprocess,'run',return_value=types.SimpleNamespace(returncode=0,stdout=text)):
                if valid: converter.require_cuda('colmap')
                else:
                    with self.assertRaises(RuntimeError):converter.require_cuda('colmap')


@unittest.skipUnless(torch.cuda.is_available(),'CUDA GPU required for device/numerical checks')
class GPURegression(unittest.TestCase):
    def setUp(self):
        namespace=dict(torch=torch,nn=nn,np=np,Image=Image,copy=copy,PILtoTorch=PILtoTorch,
                       pixel_grid=pixel_grid,getWorld2View2=getWorld2View2,
                       getProjectionMatrix=getProjectionMatrix,fov2focal=fov2focal)
        tree=ast.parse((ROOT/'scene/cameras.py').read_text(encoding='utf-8'))
        nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))]
        exec(compile(ast.Module(body=nodes,type_ignores=[]),'scene/cameras.py','exec'),namespace)
        self.camera_class=namespace['Camera'];self.process_image=namespace['process_image'];self.sample_cam=namespace['sample_cam']
        self.tmp=tempfile.TemporaryDirectory();self.path=Path(self.tmp.name)/'image.png'
        self.array=np.random.RandomState(7).randint(0,256,(18,24,3),dtype=np.uint8)
        Image.fromarray(self.array).save(self.path)
    def tearDown(self):self.tmp.cleanup();pixel_grid.cache_clear();patch_offsets.cache_clear()
    def camera(self,translation=None):
        return self.camera_class(0,np.eye(3),np.array([0.,0.,0.] if translation is None else translation),
                                 1.1,.9,24,18,str(self.path),'image',0,ncc_scale=.5)

    def test_grayscale_gpu_matches_pil_cpu_and_preloads(self):
        camera=self.camera()
        rgb,gray=camera.get_image()
        original=PILtoTorch(Image.open(self.path),(24,18))
        ncc=PILtoTorch(Image.open(self.path),(48,36))
        expected=(.299*ncc[0]+.587*ncc[1]+.114*ncc[2])[None]
        self.assertTrue(rgb.is_cuda and gray.is_cuda)
        torch.testing.assert_close(rgb.cpu(),original,rtol=0,atol=0)
        torch.testing.assert_close(gray.cpu(),expected,rtol=1e-6,atol=1e-7)
        self.assertEqual(camera.get_image()[0].data_ptr(),rgb.data_ptr())

    def test_rgba_and_cpu_memory_path(self):
        rgba=np.dstack([self.array,np.full((18,24),128,np.uint8)])
        Image.fromarray(rgba).save(self.path)
        cpu=self.process_image(str(self.path),(12,9),.5,'cpu')
        gpu=self.process_image(str(self.path),(12,9),.5,'cuda')
        for expected,actual in zip(cpu,gpu):
            self.assertTrue(actual.is_cuda)
            torch.testing.assert_close(actual.cpu(),expected,rtol=1e-6,atol=1e-7)

    def test_rays_and_intrinsics_cache_preserve_formula(self):
        camera=self.camera()
        for scale in (1.,2.,.5):
            width,height=int(24/scale),int(18/scale)
            x,y=torch.meshgrid(torch.arange(width),torch.arange(height),indexing='xy')
            old=torch.stack([(x-camera.Cx/scale)/camera.Fx*scale,
                             (y-camera.Cy/scale)/camera.Fy*scale,torch.ones_like(x)],-1).float()
            torch.testing.assert_close(camera.get_rays(scale).cpu(),old,rtol=1e-6,atol=1e-7)
            k=camera.get_k(scale);inv=camera.get_inv_k(scale)
            self.assertTrue(k.is_cuda and inv.is_cuda)
            self.assertEqual(k.data_ptr(),camera.get_k(scale).data_ptr())
            torch.testing.assert_close(k@inv,torch.eye(3,device='cuda'),rtol=1e-6,atol=1e-6)
        grid=pixel_grid(18,24,camera.world_view_transform.device)
        self.assertTrue(grid.is_cuda)
        self.assertEqual(grid.data_ptr(),pixel_grid(18,24,camera.world_view_transform.device).data_ptr())
        self.assertEqual(pixel_grid.cache_info().maxsize,4)

    def test_virtual_camera_pose_cache_updates(self):
        left=self.camera();right=self.camera([2.,1.,.5])
        mixed=self.sample_cam(left,right)
        torch.testing.assert_close(mixed.translation_tensor.cpu(),torch.tensor(mixed.T,dtype=torch.float32))
        torch.testing.assert_close(mixed.rotation_tensor.cpu(),torch.tensor(mixed.R,dtype=torch.float32))
        torch.testing.assert_close(left.translation_tensor.cpu(),torch.zeros(3))

    def test_neighbor_order_matches_original_numpy_lexsort(self):
        centers=torch.tensor([[0.,0.,0.],[1.,0.,0.],[-1.,0.,0.],[0.,1.,0.],[0.,0.,2.]],device='cuda')
        rays=torch.tensor([[0.,0.,1.],[0.,0.,1.],[.2,0.,.98],[0.,0.,1.],[0.,0.,-1.]],device='cuda')
        rays=torch.nn.functional.normalize(rays,dim=-1)
        distance=torch.norm(centers[:,None]-centers[None],dim=-1).cpu().numpy()
        angle=(torch.arccos(torch.sum(rays[:,None]*rays[None],dim=-1))*180/3.14159).cpu().numpy()
        expected=[]
        for i in range(len(centers)):
            order=np.lexsort((angle[i],distance[i]))
            mask=(angle[i,order]<90)&(distance[i,order]>.01)&(distance[i,order]<2.)
            expected.append(order[mask][:3].tolist())
        self.assertEqual(nearest_view_indices(centers,rays,90,.01,2.,3),expected)

    def test_ncc_samples_stay_on_gpu_and_have_no_duplicates(self):
        mask=torch.tensor([True,False,True,True,False,True],device='cuda')
        for count in (0,2,10):
            values=sample_valid_indices(mask,count)
            self.assertTrue(values.is_cuda)
            self.assertEqual(len(values),min(count,4))
            self.assertEqual(len(torch.unique(values)),len(values))
            self.assertTrue(mask[values].all())
        self.assertEqual(sample_valid_indices(torch.zeros(6,dtype=torch.bool,device='cuda'),3).numel(),0)


if __name__=='__main__':unittest.main(verbosity=2)
