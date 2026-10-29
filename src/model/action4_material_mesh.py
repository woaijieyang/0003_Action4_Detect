"""Estimated A0 point -> frozen CAD triangle; no GT, tracking or pose solving.

A passing binding is provisional self-consistency, never independent3D truth.
"""
import json,hashlib
import numpy as np
from data.action4_paths import PROJECT_ROOT as R,contained_path


def closest_triangle_points(point,triangles):
    x=np.asarray(point,float);tri=np.asarray(triangles,float)
    if x.shape!=(3,) or tri.ndim!=3 or tri.shape[1:]!=(3,3) or not np.isfinite(x).all() or not np.isfinite(tri).all():raise ValueError('Finite point/triangle arrays required')
    n=len(tri);a,b,c=tri[:,0],tri[:,1],tri[:,2];ab=b-a;ac=c-a;ap=x-a
    d00=(ab*ab).sum(1);d01=(ab*ac).sum(1);d11=(ac*ac).sum(1);d20=(ap*ab).sum(1);d21=(ap*ac).sum(1);den=d00*d11-d01*d01
    valid=den>np.maximum(1e-30,np.finfo(float).eps*d00*d11*16)
    safe=np.where(valid,den,1);v=(d11*d20-d01*d21)/safe;w=(d00*d21-d01*d20)/safe
    bary=np.stack([1-v-w,v,w],1);plane=np.einsum('ni,nij->nj',bary,tri)
    best=np.full(n,np.inf);points=np.zeros((n,3));weights=np.zeros((n,3))
    inside=valid&(bary.min(1)>=0);dist=((plane-x)**2).sum(1);best[inside]=dist[inside];points[inside]=plane[inside];weights[inside]=bary[inside]
    for u,vv in [(0,1),(1,2),(2,0)]:
        edge=tri[:,vv]-tri[:,u];denom=(edge*edge).sum(1);t=np.clip(((x-tri[:,u])*edge).sum(1)/np.where(denom>0,denom,1),0,1)
        q=tri[:,u]+t[:,None]*edge;dd=((q-x)**2).sum(1);take=dd<best
        bb=np.zeros((n,3));bb[:,u]=1-t;bb[:,vv]=t;best[take]=dd[take];points[take]=q[take];weights[take]=bb[take]
    return points,weights,np.sqrt(best)


class MaterialMeshBinder:
    def __setattr__(self,name,value):
        if getattr(self,'_sealed',False):raise AttributeError('Material mesh/policy sealed; construct a new binding identity')
        object.__setattr__(self,name,value)

    @property
    def binding_provenance(self):return json.loads(self._binding_provenance_json)

    def _seal(self,provenance):
        for name in ('vertices','faces','components','regions','tri','low','high','normals','usable'):
            array=np.ascontiguousarray(getattr(self,name));object.__setattr__(self,name,np.frombuffer(array.tobytes(),dtype=array.dtype).reshape(array.shape))
        text=json.dumps(provenance,sort_keys=True)
        object.__setattr__(self,'_binding_provenance_json',text)
        object.__setattr__(self,'binding_identity',hashlib.sha256(text.encode()).hexdigest())
        object.__setattr__(self,'_buffer_objects',tuple(getattr(self,k) for k in ('vertices','faces','components','regions','tri','low','high','normals','usable')))
        object.__setattr__(self,'_sealed',True)

    def assert_sealed(self):
        if not self._sealed or self.binding_identity!=hashlib.sha256(self._binding_provenance_json.encode()).hexdigest():raise ValueError('Mesh binding seal invalid')
        if any(getattr(self,k).flags.writeable for k in ('vertices','faces','components','regions','tri','low','high','normals','usable')):raise ValueError('Mesh buffers mutable')
        if not isinstance(self.excluded,frozenset):raise ValueError('Mesh exclusion set mutable')
        if any(getattr(self,k) is not v for k,v in zip(('vertices','faces','components','regions','tri','low','high','normals','usable'),self._buffer_objects)):raise ValueError('Mesh buffers replaced after sealing')
        p=self.binding_provenance
        if self.distance!=self.diagonal*p['distance_fraction'] or self.ambiguity!=self.diagonal*p['ambiguity_fraction'] or self.normal_cos!=np.cos(np.deg2rad(p['ambiguity_angle_deg'])) or sorted(self.excluded)!=p['excluded_components']:raise ValueError('Mesh numeric policy changed after sealing')

    def __init__(self,vertices,faces,components,regions,*,mesh_identity,excluded_components=(),distance_fraction=.02,ambiguity_fraction=.002,ambiguity_angle_deg=15):
        self.vertices=np.asarray(vertices,float);self.faces=np.asarray(faces);self.components=np.asarray(components);self.regions=np.asarray(regions,float)
        if self.vertices.ndim!=2 or self.vertices.shape[1]!=3 or not np.isfinite(self.vertices).all():raise ValueError('Finite meter vertices required')
        if self.faces.ndim!=2 or self.faces.shape[1]!=3 or self.faces.dtype.kind not in 'iu' or not len(self.faces) or self.faces.min()<0 or self.faces.max()>=len(self.vertices):raise ValueError('Triangle indices invalid')
        if self.components.shape!=(len(self.faces),) or self.components.dtype.kind not in 'iu':raise ValueError('Component identities invalid')
        if self.regions.ndim!=2 or self.regions.shape[1]!=3 or not np.isfinite(self.regions).all():raise ValueError('Region centers invalid')
        if not isinstance(mesh_identity,str) or not mesh_identity:raise ValueError('Mesh identity required')
        if not 0<distance_fraction<1 or not 0<=ambiguity_fraction<=distance_fraction or not 0<ambiguity_angle_deg<90:raise ValueError('Explicit bounded mesh policy required')
        self.mesh_identity=mesh_identity;self.tri=self.vertices[self.faces];self.low=self.tri.min(1);self.high=self.tri.max(1)
        normal=np.cross(self.tri[:,1]-self.tri[:,0],self.tri[:,2]-self.tri[:,0]);norm=np.linalg.norm(normal,axis=1);self.usable=norm>0;self.normals=normal/np.where(norm>0,norm,1)[:,None]
        self.diagonal=float(np.linalg.norm(self.vertices.max(0)-self.vertices.min(0)))
        if self.diagonal<=0:raise ValueError('Degenerate CAD extent')
        self.distance=self.diagonal*distance_fraction;self.ambiguity=self.diagonal*ambiguity_fraction;self.normal_cos=np.cos(np.deg2rad(ambiguity_angle_deg));self.excluded=frozenset(map(int,excluded_components))
        provenance={'algorithm':'action4.material_mesh_binding.v1','mesh_identity':mesh_identity,
            'regions_array_sha256':hashlib.sha256(self.regions.tobytes()).hexdigest(),'units':'metres',
            'distance_fraction':distance_fraction,'ambiguity_fraction':ambiguity_fraction,'ambiguity_angle_deg':ambiguity_angle_deg,'excluded_components':sorted(self.excluded)}
        provenance['geometry_array_sha256']=hashlib.sha256(self.vertices.tobytes()+self.faces.astype(np.int64).tobytes()+self.components.astype(np.int64).tobytes()).hexdigest()
        self._seal(provenance)


    @classmethod
    def from_policy(cls,path):
        path=contained_path(R,path,exists=True);cfg=json.loads(path.read_text())
        if cfg.get('schema')!='action4.material_mesh_policy.v1' or cfg.get('units_conversion_mm_to_m')!=.001 or cfg.get('normal_facing_requires_positive_dot') is not True:raise ValueError('Explicit metric mesh policy required')
        mesh=contained_path(R,cfg['mesh'],exists=True);centers=contained_path(R,cfg['surface_regions'],exists=True)
        def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
        if sha(mesh)!=cfg['mesh_sha256'] or sha(centers)!=cfg['surface_regions_sha256']:raise ValueError('Frozen CAD/region source changed')
        if sha(contained_path(R,cfg['semantic_source'].split(':')[0],exists=True))!=cfg['semantic_source_sha256']:raise ValueError('Component semantic source changed')
        canonical=contained_path(R,cfg['canonical_manifest'],exists=True)
        if sha(canonical)!=cfg['canonical_manifest_sha256']:raise ValueError('Canonical manifest identity changed')
        region_array=np.asarray(json.loads(centers.read_text())['xyz_object_m'],float)
        if region_array.shape!=(64,3):raise ValueError('Formal region source must contain64 centers')
        with np.load(mesh,allow_pickle=False) as m:
            counts={str(int(k)):int((m['face_components']==k).sum()) for k in np.unique(m['face_components'])}
            if counts!=cfg['component_triangle_counts']:raise ValueError('Component inventory differs')
            corners=np.asarray(json.loads(canonical.read_text())['corners_3d_mm'],float)
            if not np.allclose(corners.min(0),m['vertices_mm'].min(0),atol=1e-9,rtol=0) or not np.allclose(corners.max(0),m['vertices_mm'].max(0),atol=1e-9,rtol=0):raise ValueError('Canonical and mesh bounds differ')
            model=cls(m['vertices_mm']*.001,m['faces'],m['face_components'],region_array,mesh_identity=cfg['mesh_sha256'],excluded_components=cfg['excluded_components'],distance_fraction=cfg['max_surface_distance_fraction_CAD_diagonal'],ambiguity_fraction=cfg['ambiguity_distance_fraction_CAD_diagonal'],ambiguity_angle_deg=cfg['ambiguity_normal_angle_deg'])
        provenance=model.binding_provenance
        provenance.update(policy_sha256=sha(path),regions_sha256=cfg['surface_regions_sha256'],semantic_source_sha256=cfg['semantic_source_sha256'],canonical_manifest_sha256=cfg['canonical_manifest_sha256'],units_conversion_mm_to_m=.001)
        model._seal(provenance)
        return model

    def bind(self,point,region_id,rotation,translation,K,query_uv,*,max_reprojection_px):
        self.assert_sealed()
        x=np.asarray(point,float);rot=np.asarray(rotation,float);t=np.asarray(translation,float);K=np.asarray(K,float);uv=np.asarray(query_uv,float)
        if x.shape!=(3,) or rot.shape!=(3,3) or t.shape!=(3,) or K.shape!=(3,3) or uv.shape!=(2,) or not all(np.isfinite(a).all() for a in [x,rot,t,K,uv]):raise ValueError('Finite explicit geometric inputs required')
        if not np.allclose(rot.T@rot,np.eye(3),atol=1e-5,rtol=0) or not np.isclose(np.linalg.det(rot),1,atol=1e-5):raise ValueError('Rotation is not SO3')
        if K[0,0]<=0 or K[1,1]<=0 or not np.allclose(K[2],[0,0,1],atol=1e-12,rtol=0) or not np.isfinite(max_reprojection_px) or max_reprojection_px<=0:raise ValueError('Valid pinhole K and pixel gate required')
        if type(region_id) is not int or not 1<=region_id<=len(self.regions):raise ValueError('Declared foreground region ID required')
        result={'status':'rejected','reason':None,'mesh_identity':self.mesh_identity,'binding_identity':self.binding_identity,'binding_provenance':json.loads(json.dumps(self.binding_provenance)),'binding_quality':'A0_surface_pose_mesh_self_consistency_only','geometry_accepted':False}
        lower=np.linalg.norm(np.maximum(0,np.maximum(self.low-x,x-self.high)),axis=1);ids=np.flatnonzero((lower<=self.distance+self.ambiguity)&self.usable)
        if not len(ids):return dict(result,reason='outside_surface_distance')
        q,bary,dist=closest_triangle_points(x,self.tri[ids]);order=np.argsort(dist,kind='stable');j=int(order[0]);fid=int(ids[j]);minimum=float(dist[j])
        result.update(surface_distance_m=minimum,CAD_diagonal_m=self.diagonal)
        if minimum>self.distance:return dict(result,reason='outside_surface_distance')
        nearby=ids[dist<=minimum+self.ambiguity]
        if any(int(self.components[k]) in self.excluded for k in nearby):return dict(result,reason='display_or_glass_semantic_exclusion')
        if (self.normals[nearby]@self.normals[fid]<self.normal_cos).any():return dict(result,reason='multiple_face_normal_ambiguity')
        point=q[j];camera_center=-rot.T@t
        if self.normals[fid]@(camera_center-point)<=0:return dict(result,reason='back_facing_surface')
        nearest_region=int(np.argmin(np.sum((self.regions-point)**2,axis=1)))+1
        if nearest_region!=region_id:return dict(result,reason='surface_region_mismatch')
        camera=rot@point+t
        if camera[2]<=0:return dict(result,reason='nonpositive_depth')
        projected=K@camera;projected=projected[:2]/projected[2];error=float(np.linalg.norm(projected-uv));result['reprojection_error_px']=error
        if error>max_reprojection_px:return dict(result,reason='pose_surface_projection_inconsistent')
        return dict(result,status='provisional',reason='estimated_binding_requires_temporal_validation',triangle_id=fid,
            component_id=int(self.components[fid]),barycentric=bary[j].tolist(),X_cad_m=point.tolist(),surface_normal_object=self.normals[fid].tolist(),region_id=region_id)
