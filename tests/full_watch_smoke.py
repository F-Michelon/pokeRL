"""Rendering must leave the policy's evaluation trajectory unchanged."""
import tempfile
from pathlib import Path

import numpy as np

from training.full_game_env import FullGameEnv, LocalServer


def main():
    with tempfile.TemporaryDirectory() as directory, LocalServer() as server:
        reference = FullGameEnv(server.url, duration=8)
        try:
            obs, _ = reference.reset(seed=123)
            reference.page.evaluate("G.Weather.debugStart('rain'); G.Game.debugLevelUp()")
            trajectory = []
            while True:
                mask = reference.action_masks()
                direction = len(trajectory) % 9
                action = direction if mask[direction] else int(np.flatnonzero(mask)[0])
                result = reference.step(action)
                trajectory.append((action, result))
                if result[2] or result[3]:
                    break
        finally:
            reference.close()
        visual = FullGameEnv(server.url, duration=8, render_mode='human',
                             headless=True, video_dir=directory)
        try:
            drawn, _ = visual.reset(seed=123)
            np.testing.assert_array_equal(obs, drawn)
            # Exercise rendering with random cosmetic weather particles and a menu.
            visual.page.evaluate("G.Weather.debugStart('rain'); G.Game.debugLevelUp()")
            steps = 0
            for action, expected in trajectory:
                actual = visual.step(action)
                for _ in range(2):
                    visual.render()
                np.testing.assert_array_equal(actual[0], expected[0])
                assert actual[1:] == expected[1:], (actual[1:], expected[1:])
                # Check rendering itself hasn't changed the structured observation.
                snapshot = visual.page.evaluate('fullTraining.snapshot()')
                np.testing.assert_array_equal(np.asarray(snapshot['observation'], dtype=np.float32), actual[0])
                steps += 1
            assert not visual.errors, visual.errors
            assert visual.page.evaluate('''() => {
                const c = document.querySelector('#game');
                return c.getContext('2d').getImageData(0, 0, c.width, c.height).data.some(v => v);
            }'''), 'The game canvas was not rendered'
            visual.page.wait_for_timeout(200)
            video = visual.page.video
            visual.context.close()
            visual.context = None
            assert Path(video.path()).stat().st_size > 1000
            visual.close()
            visual = None
            print(f'Visual replay matches evaluation across {steps} decisions; canvas and video verified.')
        finally:
            if visual is not None:
                visual.close()


if __name__ == '__main__':
    main()
