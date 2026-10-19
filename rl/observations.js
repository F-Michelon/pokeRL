// Fixed, padded slots include a presence mask. All distances are world units.
export const OBS_SIZE=12+20*8+20*6+20*4+25;
export function observation(a) {
  const p=a.player,G=a.G;
  const v=[p.x/600,p.y/330,p.hp/p.maxHp,p.xp/p.xpNext,p.level/20,(p.svx||0)/150,(p.svy||0)/150,a.frame/(a.config.duration*60),p.activeMove().t/5,p.dex/1000,a.pending?1:0,p.atkMul/5];
  const nearest=arr=>arr.filter(e=>!e.dead).slice().sort((a,b)=>Math.hypot(a.x-p.x,a.y-p.y)-Math.hypot(b.x-p.x,b.y-p.y)).slice(0,20);
  function slots(items,size,encode){for(let i=0;i<20;i++)v.push(...(items[i]?encode(items[i]):Array(size).fill(0)));}
  slots(nearest(G.EnemyMgr.all()),8,e=>[1,(e.x-p.x)/1200,(e.y-p.y)/660,(e.vx||0)/200,(e.vy||0)/200,e.hp/e.maxHp,e.behavior==='ranged'?1:0,e.spd/200]);
  slots(nearest(G.Projectiles.all()),6,e=>[1,(e.x-p.x)/1200,(e.y-p.y)/660,e.vx/300,e.vy/300,e.friendly?1:0]);
  slots(nearest(G.Pickups.all()),4,e=>[1,(e.x-p.x)/1200,(e.y-p.y)/660,e.kind==='xp'?1:0]);
  for(let y=-2;y<=2;y++)for(let x=-2;x<=2;x++)v.push(a.blocked(p.x+x*40,p.y+y*40,p.r)?1:0);
  if(v.length!==OBS_SIZE||!v.every(Number.isFinite))throw new Error('Invalid observation');
  return v;
}
