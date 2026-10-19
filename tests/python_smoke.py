import unittest
import numpy as np
from gymnasium.utils.env_checker import check_env
from training.env import PokeEnv

class BridgeTests(unittest.TestCase):
    def test_gym_contract_and_cleanup(self):
        env=PokeEnv({'duration':2})
        try:
            check_env(env,skip_render_check=True)
            first,_=env.reset(seed=77)
            trajectory=[env.step(i%9)[0] for i in range(10)]
            env.reset(seed=77)
            for i,obs in enumerate(trajectory):np.testing.assert_array_equal(obs,env.step(i%9)[0])
            self.assertTrue(np.isfinite(first).all())
            with self.assertRaises(RuntimeError):env.request(op='step',action=99)
        finally:env.close()
        self.assertIsNotNone(env.process.poll())

if __name__=='__main__':unittest.main()
