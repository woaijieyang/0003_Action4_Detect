"""Read-only audit of target/occluder co-occurrence in published training recipes."""
from collections import Counter
from pathlib import Path
import json
from data.action4_paths import PROJECT_ROOT,contained_path


def audit(manifest,output):
    records=[json.loads(line) for line in contained_path(PROJECT_ROOT,manifest,exists=True).read_text().splitlines()]
    counts=Counter();details=[];no_anchor=0
    for record in records:
        source=contained_path(PROJECT_ROOT,record['template_parameters_ref']['path'],exists=True)
        proposal=json.loads(source.read_text());patches=proposal.get('occluder_proposals',{})
        active={side:p for side,p in patches.items() if p.get('status')=='GEOMETRY_PROPOSAL_RENDER_APPLICATION_PENDING'}
        counts[(record['target_count'],len(active))]+=1
        for patch in active.values():
            if not any(key in patch for key in ('hand_anchor','garment_anchor','attachment')):no_anchor+=1
        details.append({'sample_id':record['sample_id'],'target_count':record['target_count'],'patch_count':len(active),'parameters':str(source.relative_to(PROJECT_ROOT))})
    report={'scope':'recipe metadata co-occurrence, not measured image visibility or causal proof',
            'images':len(records),'contingency':[{'targets':t,'patches':p,'images':n} for (t,p),n in sorted(counts.items())],
            'patch_records_without_attachment_fields':no_anchor,'details':details,
            'source_logic':'make_occluder_proposals skips absent targets; TemplateRuntime creates unparented four-vertex quads',
            'training_data_modified':False}
    target=contained_path(PROJECT_ROOT,output,exists=False);target.parent.mkdir(parents=True,exist_ok=True);target.write_text(json.dumps(report,indent=2))
    return {k:v for k,v in report.items() if k!='details'}

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--manifest',default='data/train/annotations.jsonl');p.add_argument('--output',required=True);a=p.parse_args();print(json.dumps(audit(a.manifest,a.output)))
