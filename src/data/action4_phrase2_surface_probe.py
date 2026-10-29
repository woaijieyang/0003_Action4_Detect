from pathlib import Path
import sys,json
R=Path("/data/zy/vln/a-test");sys.path.insert(0,str(R/"src"))
from data.action4_phrase2_surface import SurfaceDeriver
root=R/"outputs/phrase2_generation_v001/render_benchmark_v003";done=json.loads((root/"COMPLETE.json").read_text());deriver=SurfaceDeriver();out=R/"outputs/phrase2_generation_v001/surface_contract";out.mkdir(exist_ok=False);rows=[]
for row in done["rows"]:
 if row["samples"]!=32:continue
 record=json.loads((R/row["record_ref"]["path"]).read_text());dest=out/row["template"];dest.mkdir();record=deriver.derive(record,dest)
 p=dest/"record.json";p.write_text(json.dumps(record));rows.append({"record":str(p.relative_to(R)),"template":row["template"],"objects":[x["surface_supervision"] for x in record["instances"]]})
(out/"COMPLETE.json").write_text(json.dumps({"rows":rows,"diagnostic_only":True},indent=2));print("SURFACE_CONTRACT_COMPLETE")
