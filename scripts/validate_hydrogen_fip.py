"""Run the actual batch FIP pipeline with native and reconstructed hydrogen."""
import argparse
import contextlib
import json
from pathlib import Path
import sys
import numpy as np
import astropy.units as u
import matplotlib
matplotlib.use('Agg')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from main_fip_bias_batch import process_hydrad_data
from hydrogen import HydrogenHistory


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('root')
    parser.add_argument('cache')
    parser.add_argument('--indices',type=int,nargs='+',default=[30,677])
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    history=HydrogenHistory.load(args.cache)
    rows=[]
    for index in args.indices:
        outputs=[]
        for name,h in [('native',None),('reconstructed',history)]:
            folder=args.output/f'{index}_{name}'
            print(f'Running profile {index}: {name}',flush=True)
            with (args.output/f'{index}_{name}.log').open('w') as log, contextlib.redirect_stdout(log):
                outputs.append(process_hydrad_data(args.root,.206/u.s,.97*u.km/u.s,str(folder),
                               strand_num=index,acoustic_flux=1e8*u.erg/u.cm**2/u.s,
                               v_add=0*u.km/u.s,hydrogen_history=h))
        a,b=outputs
        for element in a['fip_bias_heights']:
            for side in ('left','right'):
                x=np.asarray(a['fip_bias_heights'][element][side]['bias'])
                y=np.asarray(b['fip_bias_heights'][element][side]['bias'])
                if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
                    raise ValueError('Nonfinite FIP bias in validation')
                endpoint = -1 if side == 'left' else 0
                rows.append(dict(index=index,element=element,side=side,native_integrated=float(x[endpoint]),
                                 reconstructed_integrated=float(y[endpoint]),
                                 maximum_relative_difference=float(np.max(abs(y-x)/np.maximum(abs(x),1e-30)))))
        (args.output/'comparison.json').write_text(json.dumps(rows,indent=2))
    print('Maximum relative FIP curve difference:',max(r['maximum_relative_difference'] for r in rows))

if __name__=='__main__': main()
