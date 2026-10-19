import {Arena} from './arena.js';
import {observation} from './observations.js';
import {reward} from './rewards.js';
export class Environment extends Arena {
  reset(seed=0){super.reset(seed);return this.getObservation();}
  getObservation(){return observation(this);}
  getMetrics(){const survival=this.frame/60;return {seed:this.seed,survival,xp:this.xp,kills:this.kills,damage:this.damage,dead:this.player.dead,xpPerMinute:this.xp*60/Math.max(survival,1/60),damagePerMinute:this.damage*60/Math.max(survival,1/60),level:this.player.level,spawned:this.spawned};}
  step(action,frames=6){
    if(!Number.isInteger(action)||action<0||action>8)throw new Error('action must be 0..8');
    if(!Number.isInteger(frames)||frames<1||frames>600)throw new Error('frames must be 1..600');
    if(this.done)throw new Error('Episode ended; call reset');
    if(this.pending)throw new Error('Choose an upgrade before movement');
    const before=this.getMetrics();
    for(let i=0;i<frames&&!this.done&&!this.pending;i++)this.tick(action);
    const metrics=this.getMetrics();
    return {observation:this.getObservation(),reward:reward(before,metrics),terminated:this.player.dead,truncated:!this.player.dead&&this.done,info:{...metrics,pendingUpgrade:this.pending>0}};
  }
}
