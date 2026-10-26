import tempfile
import unittest
from pathlib import Path
import numpy as np
from scipy.linalg import expm
from hydrogen.rates import equilibrium, backward_euler
from hydrogen.transport import remap, advect
from hydrogen.profiles import check_populations
from hydrogen import HydrogenHistory


class HydrogenNumericsTests(unittest.TestCase):
    def test_equilibrium_known_distribution(self):
        target = np.array([.1,.2,.15,.05,.1,.4])
        q = np.tile(target[:, None], (1,6)) - np.eye(6)
        np.testing.assert_allclose(equilibrium(q[None])[0], target, atol=1e-14)

    def test_reaction_convergence_to_exact_transient(self):
        target = np.array([.1,.2,.15,.05,.1,.4])
        q = 8*(np.tile(target[:, None], (1,6))-np.eye(6))
        initial = np.array([[1.,0,0,0,0,0]])
        exact = expm(q*.5) @ initial[0]
        errors=[]
        for dt in (.05,.025,.0125):
            f=initial.copy()
            for _ in range(round(.5/dt)):
                f=backward_euler(q[None], f, dt)
            self.assertGreaterEqual(f.min(),0)
            self.assertAlmostEqual(f.sum(),1)
            errors.append(np.max(abs(f[0]-exact)))
        self.assertLess(errors[1],.6*errors[0])
        self.assertLess(errors[2],.6*errors[1])

    def test_remap_conserves_species_on_changed_mesh(self):
        old=np.array([0.,.1,.6,1.])
        new=np.array([0.,.2,.3,.7,1.])
        species=np.arange(18).reshape(3,6)+1.
        output=remap(old,species,new)
        np.testing.assert_allclose((species*np.diff(old)[:,None]).sum(0), (output*np.diff(new)[:,None]).sum(0),atol=1e-14)

    def test_advection_closed_boundaries_conserves_each_species(self):
        edges=np.linspace(0,1,101)
        n=np.ones(100)
        f=np.zeros((100,6)); f[:50,0]=1; f[50:,5]=1
        v=np.ones(100); v[[0,-1]]=0
        out,carrier=advect(edges,n,f,v,.01,np.array([f[0],f[-1]]))
        np.testing.assert_allclose((out*carrier[:,None]*.01).sum(0),(f*.01).sum(0),atol=1e-13)
        self.assertGreater(out[50,0],0)

    def test_advection_keeps_uniform_fraction_with_compression(self):
        edges=np.linspace(0,1,21); n=np.ones(20)
        target=np.array([.1,.2,.15,.05,.1,.4])
        f=np.tile(target,(20,1)); v=np.linspace(.1,-.1,20)
        out,_=advect(edges,n,f,v,.1,np.array([target,target]))
        np.testing.assert_allclose(out,f,atol=1e-14)

    def test_invalid_populations_are_not_silently_clipped(self):
        with self.assertRaises(ValueError): check_populations(np.array([[1.1,0,0,0,0,-.1]]))

    def test_cache_round_trip_and_wrong_grid(self):
        h=HydrogenHistory(np.array([0]),np.array([0.]),np.array([0,2]),np.array([.1,.3]),
                          np.array([[1.,0,0,0,0,0],[0,0,0,0,0,1.]]),{'schema_version':1,'hydrad_root':'/example'})
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'h.npz'; h.save(path); loaded=HydrogenHistory.load(path)
            np.testing.assert_array_equal(loaded.fraction(0),[0,1])
            with self.assertRaises(ValueError): loaded.fraction(0,[.1,.4])
            with self.assertRaises(ValueError): loaded.fraction(0,root='/wrong')
            with self.assertRaises(KeyError): loaded.fraction(1)

class LoopInterfaceTests(unittest.TestCase):
    def make_profile(self):
        from unittest.mock import Mock
        import astropy.units as u
        from pydrad.parse.parse import Profile
        p=Mock(spec=Profile)
        p.coordinate=np.array([0.,1.,2.])*u.cm
        p.level_population_hydrogen_6=np.array([.1,.5,.9])*u.dimensionless_unscaled
        p.level_population_hydrogen_1=np.array([.7,.3,.05])*u.dimensionless_unscaled
        p.electron_density=np.ones(3)*1e10/u.cm**3
        p.ion_density=p.electron_density
        p.ion_temperature=np.ones(3)*1e4*u.K
        p.ion_pressure=np.ones(3)*u.dyne/u.cm**2
        p.sound_speed=np.ones(3)*1e6*u.cm/u.s
        p.velocity=np.zeros(3)*u.cm/u.s
        return p

    def test_native_uses_continuum_not_one_minus_ground(self):
        from unittest.mock import patch
        from loop.properties import LoopProperties
        with patch('loop.properties.MagneticField'):
            p=LoopProperties(self.make_profile())
        np.testing.assert_allclose(p.h_ionisation(np.array([0.,1.,2.])),[.1,.5,.9])
        np.testing.assert_allclose(p.h_ionisation(np.array([-1.,3.])),[.1,.9])

    def test_explicit_override_and_invalid_fraction(self):
        from unittest.mock import patch
        from loop.properties import LoopProperties
        with patch('loop.properties.MagneticField'):
            p=LoopProperties(self.make_profile(),hydrogen_fraction=np.array([.2,.3,.4]))
            np.testing.assert_allclose(p.h_ionisation(np.array([0.,1.,2.])),[.2,.3,.4])
            with self.assertRaises(ValueError):
                LoopProperties(self.make_profile(),hydrogen_fraction=np.array([1.1,.3,.4]))

    def test_quasi_neutral_when_populations_missing_and_ne_equals_nh(self):
        from unittest.mock import patch
        from loop.properties import LoopProperties
        profile = self.make_profile()
        profile.level_population_hydrogen_6 = type('Missing', (), {
            'value': property(lambda self: (_ for _ in ()).throw(ValueError('No data available')))
        })()
        with patch('loop.properties.MagneticField'):
            with self.assertWarns(UserWarning) as caught:
                p = LoopProperties(profile)
        self.assertIn('No hydrogen populations', str(caught.warning))
        np.testing.assert_allclose(p.h_ionisation(np.array([0., 1., 2.])), [1., 1., 1.])
        self.assertEqual(p.hydrogen_metadata['source'], 'quasi-neutral')

    def test_missing_populations_rejected_when_ne_differs_from_nh(self):
        from unittest.mock import patch
        import astropy.units as u
        from loop.properties import LoopProperties
        profile = self.make_profile()
        profile.level_population_hydrogen_6 = type('Missing', (), {
            'value': property(lambda self: (_ for _ in ()).throw(ValueError('No data available')))
        })()
        profile.ion_density = np.array([1., 2., 3.]) * 1e10 / u.cm**3
        with patch('loop.properties.MagneticField'):
            with self.assertRaises(ValueError):
                LoopProperties(profile)


if __name__ == '__main__':
    unittest.main()
