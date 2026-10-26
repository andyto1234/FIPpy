"""Compare the Python rate port to the user's actual HYDRAD C++ implementation.

Requires a C++ compiler; creates its executable in a temporary directory. The
HYDRAD source and atomic data are read only. Run from the repository root.
"""
import argparse
import json
from pathlib import Path
import subprocess
import tempfile
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hydrogen.rates import HydrogenRates, equilibrium


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('root',type=Path)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    root=args.root.resolve()
    header=root/'Radiation_Model/source/OpticallyThick/RadiativeRates.h'
    wrapper='''#include <cstdio>
#include <cmath>
#include HEADER
int main() {
 CRadiativeRates r((char*)"Radiation_Model/atomic_data/OpticallyThick/radiative_rates/rates_files.txt");
 double te, ne, tr, lr[10], u[6],d[6],bf[4],fb[4],cu[10],cd[10],ci[5],cr[5],f[6];
 while (scanf("%lf %lf %lf",&te,&ne,&tr)==3) {
  for(int j=0;j<10;j++) lr[j]=log10(tr);
  r.GetBoundBoundRates(u,d,lr); r.GetBoundFreeRates(bf,lr+6);
  r.GetFreeBoundRates(fb,lr+6,log10(te),ne);
  r.GetCollisionalRatesRH(cu,cd,ci,cr,log10(te),ne);
  for(int j=0;j<6;j++) f[j]= j==0 ? 1. : 0.;
  r.SolveHIIFraction(f,cu,cd,ci,cr,u,d,bf,fb);
  for(int j=0;j<6;j++) printf("%.17g ",f[j]);
  printf("\\n");
 }
}
'''.replace('HEADER',json.dumps(str(header)))
    cases=np.array([(t,n,r) for t in (3000,6000,9000,15000,30000,1e6) for n in (1e8,1e11,1e14) for r in (4000,6000,10000)])
    with tempfile.TemporaryDirectory() as temp:
        temp=Path(temp); source=temp/'check.cpp'; source.write_text(wrapper)
        exe=temp/'check'
        files=[source,root/'Radiation_Model/source/OpticallyThick/RadiativeRates.cpp',root/'Resources/source/fitpoly.cpp',root/'Resources/source/file.cpp']
        result=subprocess.run(['c++','-O2','-Wno-write-strings',*[str(f) for f in files],'-o',str(exe)],capture_output=True,text=True)
        if result.returncode: raise RuntimeError(result.stderr)
        output=subprocess.run([str(exe)],cwd=root,input='\n'.join(' '.join(map(str,c)) for c in cases),capture_output=True,text=True,check=True)
    reference=np.fromstring(output.stdout,sep=' ').reshape(-1,6)
    rates=HydrogenRates(root)
    computed=equilibrium(rates.generator(cases[:,0],cases[:,1],np.repeat(cases[:,2,None],10,axis=1)))
    error=float(abs(computed-reference).max())
    np.testing.assert_allclose(computed,reference,atol=2e-11,rtol=2e-8)
    report={'cases':len(cases),'maximum_absolute_population_error':error,'passed':True,
            'reference':'Compiled original HYDRAD CRadiativeRates::SolveHIIFraction'}
    print(report)
    if args.output: args.output.write_text(json.dumps(report,indent=2))

if __name__=='__main__': main()
