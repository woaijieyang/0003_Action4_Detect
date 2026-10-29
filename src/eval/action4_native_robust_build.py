"""Build one opt-in native Tukey derivative-weight candidate, preserving parent binaries."""
import os,json,shutil,subprocess,time,hashlib
from pathlib import Path
from data.action4_paths import PROJECT_ROOT as R,contained_path

def build():
    out=R/"outputs/action4_m3t_tukey_derivative_v001";out.mkdir(exist_ok=False)
    src=R/"runtime/m3t_tukey_src_v001";wrapper=R/"runtime/pym3t_tukey_src_v001"
    for old,new in [(R/"runtime/m3t_observed_src_v001",src),(R/"runtime/pym3t_a0prior_src_v001",wrapper)]:
        for base,dirs,files in os.walk(old,followlinks=False):
            if any((Path(base)/n).is_symlink() for n in dirs+files):raise ValueError("Unexpected source symlink")
        shutil.copytree(old,new)
    h=src/"M3T/include/m3t/texture_modality.h";text=h.read_text()
    text=text.replace('#include <', '#include <',1)
    marker='  std::vector<std::vector<float>> observation_audit() const {'
    addition='''
  bool tukey_derivative_weights() const { return tukey_derivative_weights_; }
  void set_tukey_derivative_weights(bool enabled) { tukey_derivative_weights_ = enabled; }
  static float TukeyDerivativeWeight(float error, float c, float variance) {
    if (!std::isfinite(error) || error < 0 || !std::isfinite(c) || c <= 0 ||
        !std::isfinite(variance) || variance <= 0)
      throw std::invalid_argument("Finite nonnegative error and positive scale/variance required");
    if (error >= c) return 0.0f;
    const float ratio = error / c;
    const float value = 1.0f - ratio * ratio;
    return value * value / variance;
  }
'''
    assert marker in text;text=text.replace(marker,addition+marker)
    old='''      const float weight = error > std::numeric_limits<float>::min() ?
        (cost/squared_error)/variance_ : 1.0f/variance_;'''
    new='''      const float weight = tukey_derivative_weights_ ? TukeyDerivativeWeight(error,c,variance_) :
        (error > std::numeric_limits<float>::min() ? (cost/squared_error)/variance_ : 1.0f/variance_);'''
    assert old in text;text=text.replace(old,new)
    text=text.replace('  float tukey_norm_constant_ = 20.0f;','  float tukey_norm_constant_ = 20.0f;\n  bool tukey_derivative_weights_ = false;')
    text='#include <stdexcept>\n'+text;h.write_text(text)
    p=src/"M3T/src/texture_modality.cpp";text=p.read_text()
    old='''    float weight = 1.0f / variance_;
    if (error > std::numeric_limits<float>::min())
      weight = (TukeyNorm(error) / squared_error) / variance_;'''
    new='''    float weight = 1.0f / variance_;
    if (tukey_derivative_weights_)
      weight = TukeyDerivativeWeight(error, tukey_norm_constant_, variance_);
    else if (error > std::numeric_limits<float>::min())
      weight = (TukeyNorm(error) / squared_error) / variance_;'''
    assert old in text;text=text.replace(old,new);p.write_text(text)
    p=wrapper/"src/pym3t.cpp";text=p.read_text();marker='        .def("observation_audit", &TextureModality::observation_audit)'
    assert marker in text;text=text.replace(marker,marker+'\n        .def_property("tukey_derivative_weights", &TextureModality::tukey_derivative_weights, &TextureModality::set_tukey_derivative_weights)\n        .def_static("TukeyDerivativeWeight", &TextureModality::TukeyDerivativeWeight)');p.write_text(text)
    old_config=json.loads((R/"outputs/action4_m3t_a0_prior_v001/CONFIGURE.json").read_text())
    replacements={'runtime/pym3t_a0prior_src_v001':'runtime/pym3t_tukey_src_v001',
      '.cache/m3t_native/a0prior_build_v001':'.cache/m3t_native/tukey_build_v001',
      'runtime/m3t_a0prior_v001':'runtime/m3t_tukey_v001','runtime/m3t_observed_src_v001':'runtime/m3t_tukey_src_v001'}
    config=[]
    for arg in old_config:
        for old,new in replacements.items():arg=arg.replace(old,new)
        config.append(arg)
    (out/"CONFIGURE.json").write_text(json.dumps(config,indent=2))
    (out/"PREREGISTERED.json").write_text(json.dumps(dict(parent='runtime/m3t_a0prior_v001',default_enabled=False,
      variant='Tukey rho derivative divided by residual: (1-(e/c)^2)^2/variance within c, zero outside',
      changes_inlier_weights_too=True,scale_px=20,GT_used=False,required=['scalar finite difference','actual zero gradient/Hessian beyond scale','defaultOFF exact parity','same two120frame clips'],
      deployment_accepted=False),indent=2))
    prefix=R/"runtime/m3t_native";overlay=R/"runtime/m3t_osmesa"
    env=dict(os.environ,HOME=str(R/".cache/m3t_native/process_home"),TMPDIR=str(R/".cache/m3t_native/tmp"),XDG_CACHE_HOME=str(R/".cache/m3t_native"),XDG_CONFIG_HOME=str(R/".cache/m3t_native/config"),CUDA_VISIBLE_DEVICES="",OMP_NUM_THREADS="3",OPENBLAS_NUM_THREADS="1",PYTHONNOUSERSITE="1",PYTHONDONTWRITEBYTECODE="1",LD_LIBRARY_PATH=str(overlay/"lib")+":"+str(prefix/"lib"),PATH=str(prefix/"bin")+":"+os.environ["PATH"])
    cmake=str(prefix/"bin/cmake");directory=R/".cache/m3t_native/tukey_build_v001"
    for name,command in [("CONFIGURE",config),("COMPILE",[cmake,"--build",str(directory),"--target","_pym3t_mod","-j3"]),("INSTALL",[cmake,"--install",str(directory)])]:
        with (out/(name+".log")).open("x") as log:
            process=subprocess.Popen(command,env=env,cwd=R,stdout=log,stderr=subprocess.STDOUT)
            (out/"BUILD_STATUS.json").write_text(json.dumps(dict(stage=name,pid=process.pid,start=time.time())))
            status=process.wait()
        if status:
            (out/"BUILD_FAILED.json").write_text(json.dumps(dict(stage=name,exit_code=status)));raise RuntimeError(name+" failed")
    target=R/"runtime/m3t_tukey_v001/pym3t";target.mkdir(exist_ok=True)
    shutil.copyfile(wrapper/"src/pym3t/__init__.py",target/"__init__.py")
    (out/"BUILD_COMPLETE.json").write_text(json.dumps(dict(status="BUILT_NOT_INTERFACE_OR_EFFECT_ACCEPTED",end=time.time(),prefix=str(target.parent))))
