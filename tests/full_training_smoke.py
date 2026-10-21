"""Real full-game resets and decision callbacks, without hand-written arena logic."""
import json
import numpy as np
from training.full_game_env import FullGameEnv, LocalServer


def main():
    with LocalServer() as server:
        env=FullGameEnv(server.url,duration=8)
        try:
            first,_=env.reset(seed=123)
            trajectory=[env.step(i%9) for i in range(30)]
            obs,_=env.reset(seed=123);np.testing.assert_array_equal(obs,first)
            for i,expected in enumerate(trajectory):
                actual=env.step(i%9)
                np.testing.assert_array_equal(actual[0],expected[0]);assert actual[1:]==expected[1:]
            # Exercise genuine full-game level-up flow at the next simulation tick.
            env.page.evaluate('G.Game.debugLevelUp()')
            env.step(0)
            assert env.latest['info']['decision']=='upgrade'
            assert not env.action_masks()[:45].any()
            env.step(49);assert env.latest['info']['rerolls']==1
            env.step(45);assert env.latest['info']['upgrades']==1
            assert env.latest['info']['decision']=='movement'
            env.page.evaluate("G.Game.player.addMove('ember')")
            env.latest=env.page.evaluate('fullTraining.snapshot()')
            env.step(18);assert env.latest['info']['moveSwitches']==1
            assert env.page.evaluate('G.Game.player.active')==1
            # Collect an actual item; exercise the game's pause/choice/resume callback.
            env.page.evaluate("G.Game.player.equip('leftovers');G.Pickups.addRemote(99999,'item',G.Game.player.x,G.Game.player.y,'shellbell')")
            env.step(0)
            assert env.latest['info']['decision']=='item' and env.latest['info']['state']=='choice'
            env.step(46);assert env.page.evaluate('G.Game.player.item')=='shellbell'
            assert env.latest['info']['state']=='playing'
            # Genuine bond/fusion state machine, including its two-step choice.
            env.page.evaluate('G.Game.player.bonds=1;G.Game.offerBond()')
            env.latest=env.page.evaluate('fullTraining.snapshot()')
            assert env.latest['info']['decision']=='bond'
            env.step(45);env.step(45)
            assert env.latest['info']['bondChoices']==1
            while not env.latest['terminated'] and not env.latest['truncated']:
                env.step(int(np.flatnonzero(env.action_masks())[0]))
            assert env.latest['truncated']
            summary={'observation_size':env.observation_space.shape[0],'actions':int(env.action_space.n),'deterministic_replay':True,**env.latest['info']}
            obs,_=env.reset(seed=999)
            assert env.latest['info']['upgrades']==0 and env.latest['info']['itemChoices']==0
            # A distant heal stays visible even with >20 closer XP drops.
            env.page.evaluate('''() => {
                const p=G.Game.player;G.Pickups.clear();
                for(let i=0;i<30;i++)G.Pickups.addRemote(90000+i,'xp',p.x+30+i,p.y,2);
                G.Pickups.addRemote(91000,'heal',p.x+200,p.y,0);
            }''')
            snapshot=env.page.evaluate('fullTraining.snapshot()')
            layout=env.page.evaluate('fullTraining.layout()')
            heal_slot=snapshot['observation'][layout['healing']['start']:]
            assert heal_slot[0]==1 and heal_slot[1]==0.2 and heal_slot[8]==1
            # Count actual healing (capped by missing HP), not requested amount.
            env.page.evaluate('''() => {
                const p=G.Game.player;G.Pickups.clear();p.hp=p.maxHp-1;p.invuln=100;
                G.Pickups.addRemote(92000,'heal',p.x,p.y,0);
            }''')
            env.step(0)
            assert env.latest['info']['healingPickups']==1
            assert env.latest['info']['healed']==1
            assert env.page.evaluate('fullTraining.snapshot().info.rewardComponents')==env.latest['info']['rewardComponents']
            env.page.evaluate('G.Game.player.hp=1;G.Game.player.invuln=0;G.Game.player.hurt(10000)')
            assert env.page.evaluate('fullTraining.snapshot().terminated')
            print(json.dumps(summary,indent=2))
        finally:env.close()
if __name__=='__main__':main()
