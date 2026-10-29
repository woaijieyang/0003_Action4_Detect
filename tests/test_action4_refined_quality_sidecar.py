"""Tiny CPU contract fixtures; q numbers are fabricated, never model evidence."""
import copy,hashlib,json,unittest,uuid
from pathlib import Path
import numpy as np
from PIL import Image
from data.action4_paths import PROJECT_ROOT as R
from data.action4_sam_roi_inputs import array_digest
from eval.action4_b0_scoring import REFINED_QUALITY_PROTOCOL
from eval.action4_refined_quality_sidecar import quality_rows,make_sidecar,RefinedQualitySidecar

class RefinedSidecarTests(unittest.TestCase):
 def setUp(self):
  self.root=R/"outputs/action4_refined_quality_sidecar_preparation_v001/tests"/uuid.uuid4().hex;self.root.mkdir(parents=True)
  self.rgb=np.arange(36,dtype=np.uint8).reshape(3,4,3);im=self.root/"image.png";Image.fromarray(self.rgb).save(im)
  def bind(path,value):
   path.write_text(json.dumps(value));return {"path":str(path.relative_to(R)),"sha256":hashlib.sha256(path.read_bytes()).hexdigest()}
  self.bind=bind;self.K=np.eye(3)
  self.manifest=[{"sample_id":"fixture","image":str(im.relative_to(R)),"rgb_sha256":hashlib.sha256(im.read_bytes()).hexdigest(),
   "height":3,"width":4,"camera":{"K":self.K.tolist()}}]
  self.base=[{"sample_id":"fixture","instances":[{"grid_index":3,"score":.7,"bbox_full_xyxy":[0.,0.,3.,2.],"center_score":.7,"old_quality_score":.4}]}]
  self.enriched=[dict(self.base[0]["instances"][0],refined_quality_score=.8,refined_quality_protocol=copy.deepcopy(REFINED_QUALITY_PROTOCOL))]
  self.refs={k:bind(self.root/(k+".json"),v) for k,v in {"base_predictions":self.base,"data_manifest":self.manifest,"B0":{"fixture":"not_model"},"source":{"fixture":"not_inference"}}.items()}
  from eval.action4_refined_quality_sidecar import _image
  self.captures=[{"sample_id":"fixture","image":_image(self.manifest[0]),"RGB_pixel_sha256":array_digest(self.rgb),"enriched":self.enriched}]
  self.doc=make_sidecar(self.refs,self.base,self.manifest,self.captures,scope="cpu_fixture",model_config_id="fixture",selected_step=9500)
 def load(self,doc=None):
  ref=self.bind(self.root/(uuid.uuid4().hex+".json"),self.doc if doc is None else doc)
  return RefinedQualitySidecar(ref,bindings=self.refs,model_config_id="fixture",selected_step=9500,expected_scope="cpu_fixture")
 def consume(self,consumer,**kw):
  args={"sample_id":"fixture","rgb":self.rgb,"K":self.K,"base_instances":self.base[0]["instances"],"b0_sha256":self.refs["B0"]["sha256"]};args.update(kw)
  return consumer.consume(**args)
 def test_valid_unadmitted_and_copies_immutable(self):
  c=self.load();a=self.consume(c);self.assertFalse(a["admitted"]);self.assertIsNone(a["threshold"])
  a["qualities"][0]["refined_quality_score"]=0.
  self.assertEqual(self.consume(c)["qualities"][0]["refined_quality_score"],.8)
  with self.assertRaises(TypeError):c._rows["fixture"]="[]"
  with self.assertRaises(AttributeError):c._model="forged"
 def fast_args(self):
  return dict(sample_id="fixture",K=self.K,base_instances=self.base[0]["instances"],b0_sha256=self.refs["B0"]["sha256"])
 def test_fast_path_requires_actual_prior_pixels_and_exact_bindings(self):
  c=self.load();args=self.fast_args()
  self.assertIsNone(c.consume_previously_verified_original(**args))
  normal=self.consume(c)
  self.assertEqual(c.consume_previously_verified_original(**args),normal)
  path,stamp,sha=c._files[0]
  same=RefinedQualitySidecar({"path":str(path.relative_to(R)),"sha256":sha},bindings=self.refs,
      model_config_id="fixture",selected_step=9500,expected_scope="cpu_fixture")
  self.assertEqual(same.consume_previously_verified_original(**args),normal)
  for key,value in [("K",np.eye(3)*2),("sample_id","other"),("base_instances",[]),("b0_sha256","f"*64)]:
   with self.assertRaises(ValueError):same.consume_previously_verified_original(**dict(args,**{key:value}))
  changed=self.rgb.copy();changed[0,0,0]^=1
  with self.assertRaises(ValueError):self.consume(c,rgb=changed)
  self.assertIsNone(c.consume_previously_verified_original(**args))
 def test_fast_path_stale_file_and_other_process_refused(self):
  from unittest.mock import patch
  c=self.load();self.consume(c);args=self.fast_args()
  with patch("eval.action4_refined_quality_sidecar.os.getpid",return_value=c._pid+1):
   with self.assertRaises(ValueError):c.consume_previously_verified_original(**args)
  path=self.root/"image.png";raw=path.read_bytes();path.write_bytes(raw)
  self.assertIsNone(c.consume_previously_verified_original(**args))
  self.consume(c);changed=self.rgb.copy();changed[0,0,0]^=1;Image.fromarray(changed).save(path)
  with self.assertRaises(ValueError):c.consume_previously_verified_original(**args)
 def test_full_base_fields_no_tolerance(self):
  for key,value in (("score",.7000000000000001),("grid_index",4),("bbox_full_xyxy",[0.,0.,3.000000001,2.])):
   rows=copy.deepcopy(self.enriched);rows[0][key]=value
   with self.subTest(key=key),self.assertRaises(ValueError):quality_rows("fixture",self.base[0]["instances"],rows)
  rows=copy.deepcopy(self.enriched);rows[0]["extra"]=1
  with self.assertRaises(ValueError):quality_rows("fixture",self.base[0]["instances"],rows)
 def test_old_quality_missing_protocol_duplicate_and_nonfinite_refused(self):
  for change in ({"refined_quality_score":None},{"refined_quality_score":float("nan")},{"refined_quality_protocol":{}},{"refined_quality_score":2}):
   rows=copy.deepcopy(self.enriched);rows[0].update(change)
   with self.assertRaises(ValueError):quality_rows("fixture",self.base[0]["instances"],rows)
  with self.assertRaises(ValueError):quality_rows("fixture",self.base[0]["instances"],self.base[0]["instances"])
  with self.assertRaises(ValueError):quality_rows("fixture",self.base[0]["instances"]*2,self.enriched*2)
 def test_current_RGB_K_model_sample_and_base_refused(self):
  c=self.load();rgb=self.rgb.copy();rgb[0,0,0]^=1
  for kw in ({"rgb":rgb},{"K":np.eye(3)*2},{"sample_id":"other"},{"b0_sha256":"0"*64},{"base_instances":[]}):
   with self.subTest(kw=list(kw)),self.assertRaises(ValueError):self.consume(c,**kw)
 def test_missing_conflicting_resealed_domain_and_identity_refused(self):
  for mutation in ("missing","grid","sample","K","protocol","model","threshold"):
   doc=copy.deepcopy(self.doc)
   if mutation=="missing":doc["rows"][0]["qualities"]=[]
   if mutation=="grid":doc["rows"][0]["qualities"][0]["grid_index"]=2
   if mutation=="sample":doc["rows"][0]["image"]["sample_id"]="foreign"
   if mutation=="K":doc["rows"][0]["image"]["K"][0][0]=2.
   if mutation=="protocol":doc["score_protocol"]={}
   if mutation=="model":doc["model_config_id"]="foreign"
   if mutation=="threshold":doc["threshold"]=.5
   with self.subTest(mutation=mutation),self.assertRaises(ValueError):self.load(doc)
 def test_source_and_sidecar_changes_refused(self):
  c=self.load();path=R/self.refs["base_predictions"]["path"];path.write_text(path.read_text()+" ")
  with self.assertRaises(ValueError):self.consume(c)
 def test_authenticated_JSONL_manifest_consumed_without_rewriting_source(self):
  from eval.action4_refined_quality_sidecar import _read_manifest
  path=self.root/"annotations.jsonl";raw="".join(json.dumps(r)+"\n" for r in self.manifest);path.write_text(raw)
  ref={"path":str(path.relative_to(R)),"sha256":hashlib.sha256(path.read_bytes()).hexdigest()}
  self.assertEqual(_read_manifest(ref),self.manifest)
  self.refs["data_manifest"]=ref;self.doc["bindings"]=copy.deepcopy(self.refs)
  self.assertFalse(self.consume(self.load())["admitted"]);self.assertEqual(path.read_text(),raw)
  path.write_text(raw+" ")
  with self.assertRaises(ValueError):_read_manifest(ref)

 def test_actual_predict_boundary_requests_refined_field_without_GT(self):
  from unittest.mock import patch
  from eval.action4_refined_quality_sidecar import capture_one,_image
  with patch("eval.action4_refined_quality_sidecar.predict",return_value=self.enriched) as prediction:
   value=capture_one(object(),{"fixture":True},self.rgb,_image(self.manifest[0]),self.base[0]["instances"],
     device="cpu",scoring_policy="center")
   self.assertEqual(prediction.call_args.kwargs,{"include_refined_quality":True})
   self.assertEqual(value["sample_id"],"fixture")
  with patch("eval.action4_refined_quality_sidecar.predict",return_value=self.base[0]["instances"]):
   with self.assertRaises(ValueError) as failed:capture_one(object(),{},self.rgb,_image(self.manifest[0]),self.base[0]["instances"],device="cpu",scoring_policy="center")
   self.assertEqual(failed.exception.capture_evidence["sample_id"],"fixture")
   self.assertEqual(failed.exception.capture_evidence["actual_enriched"],self.base[0]["instances"])

 def test_incomplete_capture_and_fixture_not_full650(self):
  with self.assertRaises(ValueError):make_sidecar(self.refs,self.base,self.manifest,[],scope="cpu_fixture",model_config_id="fixture",selected_step=9500)
  ref=self.bind(self.root/"notfull.json",self.doc)
  with self.assertRaises(ValueError):RefinedQualitySidecar(ref,bindings=self.refs,model_config_id="fixture",selected_step=9500)

if __name__=="__main__":unittest.main()
