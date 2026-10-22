"""Oracle-correspondence and outlier geometry tests, not learned accuracy."""
from pathlib import Path
import json
import numpy as np
from data import DataAdapter
from data.action4_dataset import virtual_roi

def main():
    assert Path('src/model/action4_surface_pnp.py').exists(),'Surface-to-eight-corners decoder missing'
    from model.action4_surface_pnp import decode_surface_pose
    root=Path.cwd();record=json.loads((root/'data/train/annotations.jsonl').read_text().splitlines()[0]);obj=record['instances'][0];data=np.load(root/f'data/train/surface_v002_candidate/{record["sample_id"]}_i1.npz');canonical=np.array(record['corners_3d_mm'])/1000;low=canonical.min(0);extent=canonical.max(0)-low
    xyz=np.zeros((3,112*112));xyz[:,data['hit_grid_indices']]=((data['xyz_object_m']-low)/extent).T;mask=np.zeros(112*112);mask[data['hit_grid_indices']]=1;xyz=xyz.reshape(3,112,112);mask=mask.reshape(112,112)
    affine=DataAdapter.crop_affine(virtual_roi(obj['bbox_full_xyxy'],2.25),(112,112));K=affine@np.array(record['camera']['K']);truth=DataAdapter.transform_points(obj['corners_px'],affine)
    result=decode_surface_pose(xyz,mask,K,canonical);assert result['valid'];error=np.linalg.norm(np.array(result['corners_hm'])-truth,axis=1).max();assert error<.01,error
    corrupted=xyz.copy();bad=data['hit_grid_indices'][::5];rng=np.random.default_rng(42);corrupted.reshape(3,-1)[:,bad]=rng.random((3,len(bad)))
    robust=decode_surface_pose(corrupted,mask,K,canonical);assert robust['valid'];robust_error=np.linalg.norm(np.array(robust['corners_hm'])-truth,axis=1).max();assert robust_error<.1,robust_error
    assert not decode_surface_pose(xyz,np.zeros_like(mask),K,canonical)['valid']
    assert not decode_surface_pose(np.ones_like(xyz)*.5,mask,K,canonical)['valid']
    report={'status':'PASS','oracle_max_corner_error_hm_px':float(error),'20percent_outliers_max_error_hm_px':float(robust_error),'empty_mask_and_collapsed_xyz_rejected':True,'scope':'one synthetic object oracle geometry, not model performance'}
    (root/'outputs/action4_gdr_surface_candidate_v001/PNP_TEST.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
if __name__=='__main__':main()
