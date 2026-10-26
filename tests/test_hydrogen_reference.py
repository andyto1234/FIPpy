"""Optional integration tests against an external, read-only reference run.

HYDRAD_REFERENCE_ROOT=/path/to/run python -m unittest discover -s tests -v
"""
import os
from pathlib import Path
import tempfile
import unittest
import numpy as np
from hydrogen import reconstruct_hydrogen
from hydrogen.profiles import read_profile


@unittest.skipUnless(os.environ.get('HYDRAD_REFERENCE_ROOT'), 'Set HYDRAD_REFERENCE_ROOT for raw-data integration checks')
class ReferenceIntegrationTests(unittest.TestCase):
    def test_reconstruct_without_any_hstate_and_reject_stale_cache(self):
        source=Path(os.environ['HYDRAD_REFERENCE_ROOT']).resolve()
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            for name in ('HYDRAD','Radiation_Model','Heating_Model'):
                (root/name).symlink_to(source/name,target_is_directory=True)
            (root/'Results').mkdir()
            for i in range(31):
                for suffix in ('amr','phy'):
                    name=f'profile{i}.{suffix}'
                    # Copies, so the stale-file check never modifies reference data.
                    (root/'Results'/name).write_bytes((source/'Results'/name).read_bytes())
            self.assertFalse(list((root/'Results').glob('*.Hstate')))
            h=reconstruct_hydrogen(root)
            p=read_profile(root,30)
            x=h.fraction(30,p.s,root=root,time=p.time)
            self.assertTrue(np.all((x>=0)&(x<=1)))
            with self.assertRaises(ValueError):
                h.fraction(30,p.s,root=root,time=p.time+1)
            path=root/'Results/profile30.phy'
            path.write_text(path.read_text()+'\n')
            with self.assertRaisesRegex(ValueError,'changed since cache'):
                h.fraction(30,p.s,root=root,time=p.time)

if __name__=='__main__': unittest.main()
