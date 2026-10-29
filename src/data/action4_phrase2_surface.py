"""Same-frame sparse surface sidecars from frozen CAD and actual visible masks."""
import numpy as np
from pathlib import Path
from data import DataAdapter
from data.action4_paths import PROJECT_ROOT as R
from data.action4_dataset import virtual_roi
from data.action4_surface_intersection import exact_surface_hits

class SurfaceDeriver:
    def __init__(self):
        from mathutils.bvhtree import BVHTree
        mesh=np.load(R/"data/DJI_Action4_construct/r3_v001/mesh_v001.npz")
        self.vertices=mesh["vertices_mm"].astype(float)/1000;self.faces=mesh["faces"]
        self.tree=BVHTree.FromPolygons(self.vertices.tolist(),self.faces.tolist(),all_triangles=True)
    def derive(self,record,destination):
        from mathutils import Vector
        from data.action4_render import _mask_array
        destination=Path(destination);destination.mkdir(exist_ok=True)
        yy,xx=np.meshgrid(np.arange(112),np.arange(112),indexing="ij");grid=np.c_[xx.ravel(),yy.ravel()]
        for obj in record["instances"]:
            affine=DataAdapter.crop_affine(virtual_roi(obj["bbox_full_xyxy"],2.25),(112,112))
            native=DataAdapter.transform_points(grid,np.linalg.inv(affine));pixel=np.rint(native).astype(int);mask=_mask_array(R/obj["mask_visible"])
            inside=(pixel[:,0]>=0)&(pixel[:,1]>=0)&(pixel[:,0]<record["width"])&(pixel[:,1]<record["height"])
            idx=np.where(inside)[0];idx=idx[mask[pixel[idx,1],pixel[idx,0]]];uv=native[idx]
            K=np.array(record["camera"]["K"]);T=np.array(obj["canonical8_geometry"]["object_to_capture_cv_m"]);inverse=np.linalg.inv(T)
            origin=inverse[:3,3];rays=np.c_[uv,np.ones(len(uv))]@np.linalg.inv(K).T@inverse[:3,:3].T;rays/=np.linalg.norm(rays,axis=1)[:,None]
            hit=[];face=[];missing=[]
            for i,ray in enumerate(rays):
                location,_,triangle,_=self.tree.ray_cast(Vector(origin),Vector(ray))
                if location is None:missing.append(i)
                else:hit.append(i);face.append(triangle)
            hit=np.array(hit,dtype=int);face=np.array(face,dtype=int)
            if not len(hit):raise ValueError("Visible target has no CAD surface correspondence")
            exact=exact_surface_hits(uv[hit],K,T,self.vertices[self.faces[face]]);valid=exact["valid"]
            points=exact["xyz_object_m"][valid].astype(np.float32);selected=hit[valid];faces=face[valid]
            if not len(points):raise ValueError("All actual CAD intersections invalid")
            camera=points.astype(float)@T[:3,:3].T+T[:3,3];h=camera@K.T;error=np.linalg.norm(h[:,:2]/h[:,2:]-uv[selected],axis=1)
            if not np.isfinite(points).all() or (camera[:,2]<=0).any() or error.max()>.005:raise ValueError("Surface reprojection failure")
            path=destination/f"{record['sample_id']}_i{obj['instance_id']}.npz"
            with path.open("xb") as file:np.savez_compressed(file,xyz_object_m=points,hit_grid_indices=idx[selected].astype(np.int32),native_pixels=uv[selected],depth_capture_m=camera[:,2].astype(np.float32),face_indices=faces.astype(np.int32),missing_grid_indices=np.setdiff1d(idx,idx[selected]).astype(np.int32),K=K,object_to_capture=T)
            obj["surface_supervision"]={"path":str(path.relative_to(R)),"version":"phrase2_exact_CAD_v001","points":len(points),"max_reprojection_error_px":float(error.max()),"source":"actual visible mask + frozen CAD exact ray/triangle; no model prediction"}
        return record
