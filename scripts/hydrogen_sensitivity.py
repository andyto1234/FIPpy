"""Repeatable timestep and snapshot-cadence checks on the held-out reference."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from hydrogen import reconstruct_hydrogen, HydrogenHistory
from hydrogen.validation import validate_history


def compare(a,b):
    maximum=0.
    errors=[]
    for index in np.intersect1d(a.indices,b.indices):
        difference=abs(a.fraction(index)-b.fraction(index))
        maximum=max(maximum,float(difference.max()))
        errors.append(float(difference.mean()))
    return dict(maximum_absolute_difference=maximum,mean_cell_absolute_difference=float(np.mean(errors)))


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('root'); parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(); args.output.mkdir(parents=True,exist_ok=True)
    base=HydrogenHistory.load(args.output/'time_dependent.npz')
    report={}
    for label,kw in [('half_step',{'max_step':.125}),('tenth_cadence',{'input_stride':10}),('saved_initial',{'initial':'saved'})]:
        print(label,flush=True)
        h=reconstruct_hydrogen(args.root,mode='time_dependent',cache=args.output/f'{label}.npz',**kw)
        v=validate_history(h,args.root,output=args.output/f'{label}.json')
        report[label]={'against_base':compare(base,h),'against_reference_max':v['full_loop_max_abs'],
                       'carrier_relative_l1':h.metadata['maximum_carrier_relative_l1']}
    (args.output/'sensitivity.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
