from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
if __name__=='__main__':
    args=sys.argv[sys.argv.index('--')+1:]
    if len(args)!=2 or args[0]!='--config':raise ValueError('Expected --config')
    from data.action4_counterfactual_render import render
    render(args[1])
