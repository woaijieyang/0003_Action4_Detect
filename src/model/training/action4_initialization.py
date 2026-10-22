"""Explicit weights-only A0 warm start; never relax exact-resume identity."""
import copy

def validate_weights_initialization(stage,config,state):
 stage_config=config['stages'][stage]
 enabled=stage_config.get('weights_only_roi_migration',False)
 if type(enabled) is not bool:raise ValueError('weights_only_roi_migration must be boolean')
 if stage=='A0' and enabled:
  if state['stage']!='A0' or state['config']['network']!=config['network']:raise ValueError('A0 warm start requires matching A0 network')
  before=state['config']['preprocess'];after=config['preprocess'];changed={k for k in set(before)|set(after) if before.get(k)!=after.get(k)}
  if changed-{'roi_size','roi_expansion'}:raise ValueError('Only explicit ROI size/expansion migration is supported')
  return {'mode':'weights_only_A0_roi_migration_new_optimizer','source_preprocess':copy.deepcopy(before),'target_preprocess':copy.deepcopy(after),'changed_preprocess_fields':sorted(changed)}
 if stage=='B0' and stage_config.get('weights_only_b0_continuation') is True:
  if state['stage']!='B0' or state['config']['network']!=config['network']:raise ValueError('B0 continuation requires B0 parent with same network')
  if state.get('identity',{}).get('trainability_policy',{}).get('policy')!='independent_A0_roi':raise ValueError('B0 continuation requires independent ROI parent')
  before=state['config']['preprocess'];after=config['preprocess']
  if {k:v for k,v in before.items() if k not in ('roi_size','roi_expansion')}!={k:v for k,v in after.items() if k not in ('roi_size','roi_expansion')}:raise ValueError('Full preprocessing cannot change')
  if before!=after and not stage_config.get('roi_init_weights'):raise ValueError('ROI migration requires explicit ROI parent')
  return {'mode':'B0_weights_continuation_new_optimizer_schedule','ROI_parent':stage_config.get('roi_init_weights')}
 if stage not in ('B0','C1'):raise ValueError('D0/A0 start from the declared ImageNet backbone, not debug/task weights')
 allowed=('A0',) if stage=='B0' else ('B0','C1')
 if state['stage'] not in allowed or state['config']['network']!=config['network'] or state['config']['preprocess']!=config['preprocess']:raise ValueError('Initialization stage/architecture/preprocessing is incompatible')
 return {'mode':'weights_only_new_optimizer_schedule'}
