"""Versioned arm_geometry runtime reusing frozen complete templates and camera exports."""
from pathlib import Path
import hashlib,json,numpy as np
from data.action4_template_runtime import TemplateRuntime,TEMPLATES,bounded,read,verified_read

def validate_arm_proposal(proposal,metadata,anchor,name):
    from data import action4_parameter_sampling as sampling
    if proposal.get("occlusion_mode")!="arm_geometry" or proposal.get("occluder_proposals")!={}:
        raise ValueError("phrase2 requires arm_geometry, no cloth patches")
    if proposal["background_request"]["template"]!=name or proposal["target_presence"]!=[True,True]:
        raise ValueError("Fixed worker template and two physical actors required")
    if proposal["canonical8_m"]!=metadata["canonical_definition"]["corners_3d_m"] or proposal["corner_ids"]!=list(range(8)):
        raise ValueError("Frozen canonical8 changed")
    if proposal["K_native"]!=metadata["K_native"] or proposal["W_from_capture"]!=metadata["W_from_capture"]:
        raise ValueError("Frozen capture conversion changed")
    if proposal["provenance"]["anchor_ref"]!=metadata["baseline_parameters_ref"]:
        raise ValueError("Installation provenance differs")
    for side in ["left","right"]:
        fixed=anchor["sides"][side]
        for key in ["A_brace_from_CAD","S_skin_relative_to_brace","F_body_from_brace"]:
            if proposal["sides"][side][key]!=fixed[key]:raise ValueError("Mount/skin relation changed")
        pose=proposal["actor_pose_capture_by_side"][side]
        B=sampling.rigid(pose["brace_B_capture"]);C=sampling.rigid(pose["core_C_capture"])
        if not np.allclose(B@np.array(fixed["A_brace_from_CAD"]),C,atol=1e-9,rtol=0):
            raise ValueError("C=B@A_mount violated")
    return True

def extend_arm_once(obj,hand,total_from_junction=.50):
    """Continue existing terminal ring with a fixed upper arm; geometry freezes after init."""
    import bmesh
    from data.action4_arm_endcap_candidate import boundary_components,world_points
    groups,_=boundary_components(obj.data);hg,_=boundary_components(hand.data)
    if len(groups)!=2 or len(hg)!=1:raise ValueError("Expected existing arm junction and terminal rings")
    hp=world_points(hand,sorted(hg[0]))
    distances=[float(np.linalg.norm(world_points(obj,sorted(g))[:,None]-hp[None],axis=-1).min(1).mean()) for g in groups]
    near=int(np.argmin(distances));far=1-near
    if distances[near]>.002:raise ValueError("Source arm junction not continuous")
    # Order the far boundary by actual mesh edges.
    _,edges=boundary_components(obj.data);selected=groups[far];neighbors={i:[] for i in selected}
    for a,b in edges:
        if a in selected and b in selected:neighbors[a].append(b);neighbors[b].append(a)
    if any(len(v)!=2 for v in neighbors.values()):raise ValueError("Non-cyclic arm terminal")
    order=[min(selected)];previous=None
    while True:
        candidates=[v for v in neighbors[order[-1]] if v!=previous]
        nxt=candidates[0]
        if nxt==order[0]:break
        previous=order[-1];order.append(nxt)
        if len(order)>len(selected):raise ValueError("Terminal ring traversal failed")
    vertices=np.array([v.co[:] for v in obj.data.vertices]);start=vertices[sorted(groups[near])].mean(0);end=vertices[order].mean(0)
    m=np.array(obj.matrix_world);world_vec=m[:3,:3]@(end-start);length=float(np.linalg.norm(world_vec))
    extra=total_from_junction-length
    if not .05<extra<.45:raise ValueError("Unexpected base forearm proportion: "+str(length))
    direction=(end-start)/length
    new_vertices=vertices.tolist();faces=[list(f.vertices) for f in obj.data.polygons];previous_ring=order
    # Fixed continuous extension, mild taper and no per-sample mesh changes.
    for fraction in [.25,.5,.75,1.]:
        center=end+direction*extra*fraction
        ring=center+(vertices[order]-end)*(1-.10*fraction)
        ids=list(range(len(new_vertices),len(new_vertices)+len(order)));new_vertices.extend(ring.tolist())
        for i in range(len(order)):faces.append([previous_ring[i],previous_ring[(i+1)%len(order)],ids[(i+1)%len(order)],ids[i]])
        previous_ring=ids
    faces.append(previous_ring[::-1])
    mesh=obj.data.copy();mesh.clear_geometry();mesh.from_pydata(new_vertices,[],faces);mesh.update()
    bm=bmesh.new();bm.from_mesh(mesh);bmesh.ops.recalc_face_normals(bm,faces=list(bm.faces));bm.to_mesh(mesh);bm.free()
    for poly in mesh.polygons:poly.use_smooth=True
    obj.data=mesh
    return {"original_forearm_m":length,"extension_m":extra,"total_m":total_from_junction,"terminal_local":(end+direction*extra).tolist(),"junction_error_m":distances[near]}

class Phrase2Runtime(TemplateRuntime):
    def __init__(self,name):
        import bpy
        from mathutils import Matrix
        from data import action4_modular_render as camera
        self.name=name;new_meta="data/action4_scene_templates/phrase2_v003/"+name+"/template.json"
        previous_meta="data/action4_scene_templates/phrase2_v002/"+name+"/template.json"
        first_meta="data/action4_scene_templates/phrase2_v001/"+name+"/template.json"
        self.metadata_path=new_meta if bounded(new_meta).exists() else previous_meta if bounded(previous_meta).exists() else first_meta if bounded(first_meta).exists() else "data/action4_scene_templates/v007_candidate/"+TEMPLATES[name]+"/template.json"
        self.metadata=read(self.metadata_path);self.anchor=verified_read(self.metadata["baseline_parameters_ref"])
        repaired=bool(self.metadata.get("phrase2_arm_geometry_version"))
        blend=bounded(self.metadata["blend_ref"]["path"])
        if hashlib.sha256(blend.read_bytes()).hexdigest()!=self.metadata["blend_ref"]["sha256"]:raise ValueError("Base scene identity changed")
        bpy.ops.wm.open_mainfile(filepath=str(blend));self.scene=bpy.context.scene;self.scene.render.use_compositing=True
        for image in bpy.data.images:
            if image.source=="FILE" and not image.packed_file and image.filepath:bounded(bpy.path.abspath(image.filepath))
        for lib in bpy.data.libraries:bounded(bpy.path.abspath(lib.filepath))
        self.K=np.array(self.metadata["K_native"]);self.W=np.array(self.metadata["W_from_capture"])
        if not np.allclose(camera.actual_K(self.scene),self.K,atol=1e-6,rtol=0):raise ValueError("Actual K changed")
        if any(o.name.startswith("random_cloth") for o in self.scene.objects):raise ValueError("Legacy floating patches present in source")
        # Original table/garment remain at their verified base proportions. Only blank far wall grows to cover view.
        wall=self.scene.objects.get("modular_preview_wall")
        if wall:
            capture=np.linalg.inv(self.W)@np.array(wall.matrix_world);capture[2,3]=1.85
            wall.matrix_world=Matrix((self.W@capture).tolist());wall.dimensions=(3.6,2.8,.02)
        self.arm_repairs=self.metadata.get("arm_repairs",{}) if repaired else {}
        if not repaired:
            for side in ["left","right"]:
                obj=self.scene.objects[side+"_forearm_continuation"];hand=self.scene.objects[side+"_forearm_natural_hand"]
                self.arm_repairs[side]=extend_arm_once(obj,hand)
        bpy.context.view_layer.update()
        self.world={o.name:np.array(o.matrix_world) for o in self.scene.objects};self.parents={o.name:o.parent.name if o.parent else None for o in self.scene.objects}
        self.hide={o.name:bool(o.hide_render) for o in self.scene.objects};self.pass_indices={o.name:int(o.pass_index) for o in self.scene.objects}
        self.bindings=None
        for line in bounded(self.metadata["parent_manifest_ref"]["path"]).read_bytes().splitlines():
            row=json.loads(line)
            if row["sample_id"]==self.metadata["sample_id"]:self.bindings=row["modular_backend"]["scene_contract"]["actor_bindings"];break
        if self.bindings is None:raise ValueError("Original target role bindings missing")
        self.base_lights={o.name:float(o.data.energy)/self.anchor["environment_controls"]["area_light_gain"] for o in self.scene.objects if o.type=="LIGHT"}
        self.base_exposure=float(self.scene.view_settings.exposure)-self.anchor["environment_controls"]["exposure_offset_ev"]
        self.patches={};self.additional_source_refs=[];self.current=None;self.expected=None
        self.local_actor={};self.arm_axis_B={};self.arm_terminal_B={};self.arm_terminal_points_B={};self.arm_junction_points_B={};self.upperarm_points_B={}
        for side in ["left","right"]:
            group=self.metadata["actor_groups"][side];base=np.array(group["base_brace_capture_m"]);inverse=np.linalg.inv(self.W@base)
            obj=self.scene.objects[side+"_forearm_continuation"];terminal=np.array(self.arm_repairs[side]["terminal_local"]);terminal_world=np.array(obj.matrix_world)@np.r_[terminal,1]
            end=(inverse@terminal_world)[:3];self.arm_terminal_B[side]=end
            if "junction_local" in self.arm_repairs[side]:
                junction=(inverse@np.array(obj.matrix_world)@np.r_[self.arm_repairs[side]["junction_local"],1])[:3]
                self.arm_axis_B[side]=junction/np.linalg.norm(junction)
            else:self.arm_axis_B[side]=end/np.linalg.norm(end)
            vertices=np.array([v.co[:] for v in obj.data.vertices])
            ring=vertices[self.arm_repairs[side]["terminal_vertex_indices"]] if "terminal_vertex_indices" in self.arm_repairs[side] else vertices[-44:]
            local_to_B=inverse@np.array(obj.matrix_world)
            self.arm_terminal_points_B[side]=ring@local_to_B[:3,:3].T+local_to_B[:3,3]
            from data.action4_arm_endcap_candidate import boundary_components
            boundaries,_=boundary_components(obj.data)
            if len(boundaries)!=1:raise ValueError("One exact forearm-upperarm junction required")
            junction_points=vertices[sorted(boundaries[0])]
            self.arm_junction_points_B[side]=junction_points@local_to_B[:3,:3].T+local_to_B[:3,3]
            self.upperarm_points_B[side]=vertices@local_to_B[:3,:3].T+local_to_B[:3,3]
        self.table_plane=None;self.support_solid=None
        table=self.scene.objects.get("modular_environment_v002_table") or self.scene.objects.get("modular_preview_table")
        if table:
            m=np.linalg.inv(self.W)@np.array(table.matrix_world);normal=m[:3,2];normal/=np.linalg.norm(normal)
            point=m[:3,3]
            if np.dot(normal,-point)<0:normal=-normal
            self.table_plane=(normal,point+normal*float(table.dimensions.z)/2)
            vv=np.array([v.co[:] for v in table.data.vertices]) if table.type=="MESH" else np.empty((0,3))
            if len(vv)==8 and all(len(np.unique(vv[:,i]))==2 for i in range(3)):
                self.support_solid=(np.linalg.inv(np.linalg.inv(self.W)@np.array(table.matrix_world)),vv.min(0),vv.max(0))
        for side in ["left","right"]:
            inverse=np.linalg.inv(self.W@np.array(self.metadata["actor_groups"][side]["base_brace_capture_m"]))
            corners=[]
            for name in self.metadata["actor_groups"][side]["objects"]:
                obj=self.scene.objects[name]
                if obj.type not in ("MESH","CURVE") or obj.hide_render or name in self.bindings[side].get("always_hidden",[]):continue
                mm=inverse@np.array(obj.matrix_world)
                from data.action4_modular_assembly import world_geometry
                world_vertices,_=world_geometry(obj)
                corners.extend((world_vertices@inverse[:3,:3].T+inverse[:3,3]).tolist())
            self.local_actor[side]=np.asarray(corners)
        self._geometry_id={n:(len(o.data.vertices),len(o.data.polygons)) for n,o in self.scene.objects.items() if o.type=="MESH"}

    def apply(self,proposal):
        import bpy
        from mathutils import Matrix
        validate_arm_proposal(proposal,self.metadata,self.anchor,self.name)
        expected={name:matrix.copy() for name,matrix in self.world.items()}
        for name in self.world:
            obj=self.scene.objects[name];obj.hide_render=self.hide[name];obj.pass_index=self.pass_indices[name]
        for side in ["left","right"]:
            group=self.metadata["actor_groups"][side];B=np.array(proposal["actor_pose_capture_by_side"][side]["brace_B_capture"])
            delta=self.W@B@np.linalg.inv(self.W@np.array(group["base_brace_capture_m"]))
            for name in group["roots"]:self.scene.objects[name].matrix_world=Matrix((delta@self.world[name]).tolist())
            for name in group["objects"]:expected[name]=delta@self.world[name]
            for name in self.bindings[side].get("always_hidden",[]):self.scene.objects[name].hide_render=True
        for name,value in self.base_lights.items():self.scene.objects[name].data.energy=value*proposal["environment_controls"]["area_light_gain"]
        self.scene.view_settings.exposure=self.base_exposure+proposal["environment_controls"]["exposure_offset_ev"]
        bpy.context.view_layer.update();self.current=proposal;self.expected=expected
        self.expected_flags={o.name:(bool(o.hide_render),int(o.pass_index)) for o in self.scene.objects}
        self.verify(self.scene);return self.targets()

    def verify(self,scene):
        super().verify(scene)
        if any(o.name.startswith("random_cloth") for o in scene.objects):raise ValueError("Legacy patches may not reappear after reset")
        for n,(v,f) in self._geometry_id.items():
            obj=scene.objects[n]
            if len(obj.data.vertices)!=v or len(obj.data.polygons)!=f:raise ValueError("Frozen phrase2 mesh topology changed")
