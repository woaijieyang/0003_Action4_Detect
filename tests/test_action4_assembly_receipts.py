"""Receipt-location regression: real small manifests/PNG validation, no GPU."""
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from PIL import Image
from data import action4_synthesis as synth

class AssemblyReceiptContracts(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='assembly_receipt_fixture_',dir=synth.ROOT/'data')
        self.root=Path(self.temp.name);self.runroot=synth._run_output_dir(self.root)
        self.shard=self.root/'shard';self.shard.mkdir();self.run=synth._run_output_dir(self.shard);self.run.mkdir(parents=True)
        self.plan_path=self.root/'plan.json';self.output=self.root/'assembled'
        records=[];entries=[]
        for idx,split in enumerate(('train','eval')):
            sid=f'unit_fixture_{idx}';image=self.shard/f'{sid}.png';Image.new('RGB',(16,16),(30+idx,40,50)).save(image)
            entry={'sample_id':sid,'split':split,'group_id':sid,'target_count':0};entries.append(entry)
            records.append({'schema_version':'action4.v1',**entry,'image':str(image.relative_to(synth.ROOT)), 'rgb_sha256':synth._hash(image),'background_source_id':sid,'scene_family':'hard_negative','width':16,'height':16,'instances':[],'camera':{},'corners_3d_mm':[], 'synthesis_code_sha256':'a'*64,'render_base_sha256':'b'*64,'status':'unit_test_fixture_not_training'})
        self.plan={'entries':entries,'split_counts':{'train':1,'eval':1,'test':0},'synthesis_code_sha256':'a'*64,'render_base_sha256':'b'*64}
        self.plan_path.write_text(json.dumps(self.plan));(self.shard/'manifest.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in records))
        self.receipt_data={'plan_sha256':synth._hash(self.plan_path),'expected_sample_ids':[e['sample_id'] for e in entries],'command':['fixture_only']}
    def tearDown(self):
        assert self.runroot.resolve().is_relative_to((synth.ROOT/'outputs/action4_synthesis_runs').resolve())
        if self.runroot.exists():shutil.rmtree(self.runroot)
        self.temp.cleanup()
    def receipt(self,where,value=None):
        path=(self.run if where=='current' else self.shard)/'launch.json';path.write_text(json.dumps(self.receipt_data if value is None else value));return path
    def call(self):
        try:return synth.assemble(self.plan_path,[self.shard],self.output)
        except FileNotFoundError as error:self.fail(f'Assembly missed canonical launch receipt: {error}')
    def test_current_receipt_complete_fixture_assembles_without_training_acceptance(self):
        self.receipt('current');result=self.call()
        self.assertEqual(result['count'],2);self.assertFalse(result['training_accepted']);self.assertFalse((self.shard/'launch.json').exists())
        for split,n in [('train',1),('eval',1),('test',0)]:
            rows=(self.output/f'{split}.jsonl').read_text().splitlines();self.assertEqual(len(rows),n)
            for row in rows:self.assertEqual(json.loads(row)['status'],'render_candidate_not_training_accepted')
    def test_legacy_only_receipt_remains_compatible(self):
        self.receipt('legacy');self.assertEqual(self.call()['count'],2)
    def test_both_identical_semantic_receipts_allowed(self):
        self.receipt('current');self.receipt('legacy');self.assertEqual(self.call()['count'],2)
    def test_both_same_plan_but_conflicting_receipt_content_rejected(self):
        self.receipt('current');self.receipt('legacy',{**self.receipt_data,'command':['conflicting_fixture']})
        with self.assertRaisesRegex(ValueError,'Conflicting'):self.call()
        self.assertFalse(self.output.exists())
    def test_malformed_current_not_hidden_by_legacy_fallback(self):
        self.receipt('legacy');(self.run/'launch.json').write_text('{bad-json')
        with self.assertRaises(json.JSONDecodeError):self.call()
        self.assertFalse(self.output.exists())
    def test_wrong_plan_receipt_rejected_before_validation(self):
        self.receipt('current',{**self.receipt_data,'plan_sha256':'0'*64})
        with self.assertRaisesRegex(ValueError,'different frozen plan'):self.call()
        self.assertFalse((self.run/'geometry_validation.json').exists());self.assertFalse(self.output.exists())
    def test_current_receipt_reaches_complete_sample_set_check(self):
        self.plan['entries'].append({'sample_id':'missing','split':'test','group_id':'missing','target_count':0});self.plan_path.write_text(json.dumps(self.plan));self.receipt_data['plan_sha256']=synth._hash(self.plan_path);self.receipt('current')
        with self.assertRaisesRegex(ValueError,'every planned sample'):self.call()
        self.assertTrue((self.run/'geometry_validation.json').exists());self.assertFalse(self.output.exists())
    def test_source_identity_gate_unchanged(self):
        self.plan['synthesis_code_sha256']='c'*64;self.plan_path.write_text(json.dumps(self.plan));self.receipt_data['plan_sha256']=synth._hash(self.plan_path);self.receipt('current')
        with self.assertRaisesRegex(ValueError,'different source code'):self.call()
        self.assertFalse(self.output.exists())
    def test_missing_both_receipts_rejected(self):
        with self.assertRaises(FileNotFoundError):synth.assemble(self.plan_path,[self.shard],self.output)
        self.assertFalse(self.output.exists())

if __name__=='__main__':unittest.main()