"""Build opt-in result-time pose-cache synchronization with read-only diagnostics."""
import os,json,shutil,subprocess,time
from pathlib import Path
from data.action4_paths import PROJECT_ROOT as R

def build():
    out=R/"outputs/action4_m3t_pose_cache_v001";out.mkdir(exist_ok=False)
    src=R/"runtime/m3t_pose_cache_src_v001";wrapper=R/"runtime/pym3t_pose_cache_src_v001"
    for old,new in [(R/"runtime/m3t_observed_src_v001",src),(R/"runtime/pym3t_a0prior_src_v001",wrapper)]:
        if not old.resolve().is_relative_to(R) or not new.parent.resolve().is_relative_to(R):
            raise ValueError("Source/output outside project")
        for directory,dirs,files in os.walk(old,followlinks=False):
            if any((Path(directory)/n).is_symlink() for n in dirs+files):
                raise ValueError("Unexpected source symlink")
        shutil.copytree(old,new)
    header=src/"M3T/include/m3t/texture_modality.h"
    text=header.read_text()
    marker="  std::vector<std::vector<float>> observation_audit() const {"
    addition="""  int audit_keyframe_age() const { return keyframe_age_; }
  std::vector<int> audit_keyframe_point_counts() const {
    std::vector<int> counts;
    for (const auto &points : points_keyframes_)
      counts.push_back(static_cast<int>(points.size()));
    return counts;
  }
"""
    if text.count(marker)!=1:raise ValueError("Expected exact parent header")
    header.write_text(text.replace(marker,addition+marker))
    binding=wrapper/"src/pym3t.cpp";text=binding.read_text()
    marker='        .def("observation_audit", &TextureModality::observation_audit)'
    addition='\n        .def("audit_keyframe_age", &TextureModality::audit_keyframe_age)\n        .def("audit_keyframe_point_counts", &TextureModality::audit_keyframe_point_counts)'
    if text.count(marker)!=1:raise ValueError("Expected exact parent binding")
    binding.write_text(text.replace(marker,marker+addition))
    header=src/"M3T/include/m3t/texture_modality.h";text=header.read_text()
    marker="  int audit_keyframe_age() const { return keyframe_age_; }"
    extra="""  bool refresh_result_pose_cache() const { return refresh_result_pose_cache_; }
  void set_refresh_result_pose_cache(bool value) { refresh_result_pose_cache_ = value; }
  float result_pose_cache_before() const { return result_pose_cache_before_; }
  float audit_pose_cache_difference() const {
    const auto actual = color_camera_ptr_->world2camera_pose() * body_ptr_->body2world_pose();
    return (body2camera_pose_.matrix() - actual.matrix()).cwiseAbs().maxCoeff();
  }
"""
    assert text.count(marker)==1;text=text.replace(marker,extra+marker)
    marker="  int keyframe_age_ = 0;"
    assert text.count(marker)==1;text=text.replace(marker,marker+"\n  bool refresh_result_pose_cache_ = false;\n  float result_pose_cache_before_ = 0.0f;")
    header.write_text(text)
    cpp=src/"M3T/src/texture_modality.cpp";text=cpp.read_text()
    marker="""bool TextureModality::CalculateResults(int iteration) {
  if (!IsSetup()) return false;
"""
    replacement=marker+"""
  result_pose_cache_before_ = audit_pose_cache_difference();
  if (refresh_result_pose_cache_) PrecalculatePoseVariables();
"""
    assert text.count(marker)==1;cpp.write_text(text.replace(marker,replacement))
    binding=wrapper/"src/pym3t.cpp";text=binding.read_text()
    marker='        .def("audit_keyframe_point_counts", &TextureModality::audit_keyframe_point_counts)'
    extra='\n        .def_property("refresh_result_pose_cache", &TextureModality::refresh_result_pose_cache, &TextureModality::set_refresh_result_pose_cache)\n        .def_property_readonly("result_pose_cache_before", &TextureModality::result_pose_cache_before)\n        .def("audit_pose_cache_difference", &TextureModality::audit_pose_cache_difference)'
    assert text.count(marker)==1;binding.write_text(text.replace(marker,marker+extra))
    old_config=json.loads((R/"outputs/action4_m3t_a0_prior_v001/CONFIGURE.json").read_text())
    replacements={'runtime/pym3t_a0prior_src_v001':'runtime/pym3t_pose_cache_src_v001',
      '.cache/m3t_native/a0prior_build_v001':'.cache/m3t_native/pose_cache_build_v001',
      'runtime/m3t_a0prior_v001':'runtime/m3t_pose_cache_v001','runtime/m3t_observed_src_v001':'runtime/m3t_pose_cache_src_v001'}
    config=[]
    for arg in old_config:
        for old,new in replacements.items():arg=arg.replace(old,new)
        config.append(arg)
    (out/"CONFIGURE.json").write_text(json.dumps(config,indent=2))
    (out/"PREREGISTERED.json").write_text(json.dumps(dict(parent="runtime/m3t_a0prior_v001",
      change="refresh cached camera/body transform at CalculateResults after current renderer refresh; defaultOFF",
      required=["native getter import", "defaultOFF same-input pose parity", "actual stale cache before results", "ON cache/current pose equality", "same352 original video review"],
      deployment_accepted=False),indent=2))
    prefix=R/"runtime/m3t_native";overlay=R/"runtime/m3t_osmesa"
    env=dict(os.environ,HOME=str(R/".cache/m3t_native/process_home"),TMPDIR=str(R/".cache/m3t_native/tmp"),XDG_CACHE_HOME=str(R/".cache/m3t_native"),XDG_CONFIG_HOME=str(R/".cache/m3t_native/config"),CUDA_VISIBLE_DEVICES="",OMP_NUM_THREADS="3",OPENBLAS_NUM_THREADS="1",PYTHONNOUSERSITE="1",PYTHONDONTWRITEBYTECODE="1",LD_LIBRARY_PATH=str(overlay/"lib")+":"+str(prefix/"lib"),PATH=str(prefix/"bin")+":"+os.environ["PATH"])
    cmake=str(prefix/"bin/cmake");directory=R/".cache/m3t_native/pose_cache_build_v001"
    for name,command in [("CONFIGURE",config),("COMPILE",[cmake,"--build",str(directory),"--target","_pym3t_mod","-j3"]),("INSTALL",[cmake,"--install",str(directory)])]:
        with (out/(name+".log")).open("x") as log:
            process=subprocess.Popen(command,env=env,cwd=R,stdout=log,stderr=subprocess.STDOUT)
            (out/"BUILD_STATUS.json").write_text(json.dumps(dict(stage=name,pid=process.pid,start=time.time())))
            status=process.wait()
        if status:
            (out/"BUILD_FAILED.json").write_text(json.dumps(dict(stage=name,exit_code=status)));raise RuntimeError(name+" failed")
    target=R/"runtime/m3t_pose_cache_v001/pym3t";target.mkdir(exist_ok=True)
    shutil.copyfile(wrapper/"src/pym3t/__init__.py",target/"__init__.py")
    (out/"BUILD_COMPLETE.json").write_text(json.dumps(dict(status="BUILT_NOT_INTERFACE_OR_EFFECT_ACCEPTED",end=time.time(),prefix=str(target.parent))))
