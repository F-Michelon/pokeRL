import {createGame} from './upstream-source.js';
import {seededRandom} from './seeds.js';

export const ACTIONS = [[0,0],[0,-1],[0,1],[-1,0],[1,0],[-1,-1],[1,-1],[-1,1],[1,1]].map(([x,y]) => {const n=Math.hypot(x,y)||1;return [x/n,y/n];});
const noop = () => {};
class Animator {update(){} play(){} loop(){} busy(){return false;} progress(){return 1;}}
export const DEFAULTS = {duration:60, stage:0, spawnEvery:0.8, enemySpeed:56, enemyHp:17, maxEnemies:100, strategic:false};

export class Arena {
  constructor(config={}) {
    this.config={...DEFAULTS,...config};
    const c=this.config;
    for (const k of Object.keys(config)) if (!(k in DEFAULTS)) throw new Error(`Unknown config: ${k}`);
    for (const k of ['duration','spawnEvery','enemySpeed','enemyHp','maxEnemies']) if (!Number.isFinite(c[k]) || c[k]<=0) throw new Error(`Invalid ${k}`);
    if (!Number.isInteger(c.stage)||c.stage<0||c.stage>4) throw new Error('stage must be 0..4');
    if (!Number.isInteger(c.maxEnemies)||c.maxEnemies>500) throw new Error('maxEnemies must be 1..500');
  }
  reset(seed=0) {
    if (!Number.isInteger(seed)||seed<0||seed>4294967295) throw new Error('seed must be uint32');
    this.seed=seed; this.frame=0; this.axis=[0,0]; this.damage=0; this.xp=0; this.kills=0; this.pending=0; this.spawned=0;
    this.rng=seededRandom(seed);
    const math=Object.create(Math); math.random=this.rng;
    const G=this.G=createGame(math,{now:()=>this.frame*1000/60});
    this.obstacles=this.config.stage>=3 ? [{x:-160,y:-100,w:50,h:200},{x:110,y:-100,w:50,h:200}] : [];
    G.Audio={sfx:noop}; G.Pad={rumble:noop}; G.Settings={get:()=>false};
    G.FX=Object.fromEntries(['burst','ring','dmgText','spark','trail','beam','slash','sp','twinkle'].map(k=>[k,noop]));
    G.VFX={forMove:()=>null,forType:()=>null,spins:()=>false,isDir:()=>false};
    G.Sprites={Animator,playerScale:()=>1,playerRadius:()=>12,bodyRadius:()=>12,bodyHeight:()=>24,hasShiny:()=>false,hovers:()=>false,animMeta:()=>null,dirFromAngle:()=>0,preload:noop};
    G.Coop={active:false,isHost:false}; G.Weather={speedFor:()=>1,mult:()=>1};
    G.Judgment={active:false,noteDmg:noop}; G.Progress={kill:()=>this.kills++};
    G.Rift={inArena:true}; G.Path={dirFor:()=>null};
    G.Camera={kick:noop,sees:()=>true,outerRadius:()=>600};
    G.Input={axis:()=>this.axis}; G.Game={st:{}};
    Object.defineProperty(G.Game.st,'time',{get:()=>this.frame/60});
    G.World={collide:e=>this.collide(e),isLiquid:()=>false,nearestBreakable:()=>null,
      propsInCircle:()=>[],isFree:(x,y,r=0)=>!this.blocked(x,y,r),isWall:(x,y)=>this.blocked(x,y),projectileBlock:(x,y,r)=>this.blocked(x,y,r)};
    const choices=[[7,'water-gun'],[4,'ember'],[1,'vine-whip']];
    const [dex,move]=choices[this.config.stage===4?Math.floor(this.rng()*3):0];
    this.player=new G.Player(G.DEX_BY[dex]); this.player.addMove(move);
    const hurt=this.player.hurt.bind(this.player);
    this.player.hurt=(...args)=>{const before=this.player.hp;hurt(...args);this.damage+=before-this.player.hp;};
    return this;
  }
  blocked(x,y,r=0){return Math.abs(x)+r>600||Math.abs(y)+r>330||this.obstacles.some(o=>x+r>o.x&&x-r<o.x+o.w&&y+r>o.y&&y-r<o.y+o.h);}
  collide(e) {
    const ox=e.x,oy=e.y;
    e.x=Math.max(-600+e.r,Math.min(600-e.r,e.x)); e.y=Math.max(-330+e.r,Math.min(330-e.r,e.y));
    for(const o of this.obstacles) {
      if(e.x+e.r<=o.x||e.x-e.r>=o.x+o.w||e.y+e.r<=o.y||e.y-e.r>=o.y+o.h)continue;
      const d=[Math.abs(e.x-(o.x-e.r)),Math.abs(e.x-(o.x+o.w+e.r)),Math.abs(e.y-(o.y-e.r)),Math.abs(e.y-(o.y+o.h+e.r))];
      const side=d.indexOf(Math.min(...d));
      if(side===0)e.x=o.x-e.r;if(side===1)e.x=o.x+o.w+e.r;if(side===2)e.y=o.y-e.r;if(side===3)e.y=o.y+o.h+e.r;
    }
    return ox!==e.x||oy!==e.y;
  }
  spawn() {
    const G=this.G,c=this.config;
    if(G.EnemyMgr.count>=c.maxEnemies)return;
    const a=this.rng()*Math.PI*2;
    const ranged=c.stage>=2 && this.spawned%4===3;
    const def={dex:ranged?25:19,hp:c.enemyHp,spd:c.enemySpeed,dmg:6,xp:3.5,behavior:ranged?'ranged':'chase',shotCd:2.5,range:200,shotSpeed:130};
    const e=new G.Enemy(def,Math.cos(a)*570,Math.sin(a)*300,{hp:1,spd:1,dmg:1});
    this.collide(e);G.EnemyMgr.add(e);this.spawned++;
  }
  tick(action) {
    const G=this.G,p=this.player,dt=1/60;
    this.axis=ACTIONS[action];
    p.update(dt);
    const interval=Math.max(1,Math.round(this.config.spawnEvery*60/(this.config.stage>=1?2:1)));
    if(this.frame%interval===0)this.spawn();
    const previous=new Map(G.EnemyMgr.all().map(e=>[e.id,[e.x,e.y]]));
    G.EnemyMgr.update(dt,[p]);G.EnemyMgr.rebuildGrid();
    for(const e of G.EnemyMgr.all()){const old=previous.get(e.id);e.vx=old?(e.x-old[0])/dt:0;e.vy=old?(e.y-old[1])/dt:0;}
    G.Combat.tick(dt,p,this.frame/60);G.Projectiles.update(dt,p,G.EnemyMgr.all());
    G.Combat.resolve(dt,p,this.frame/60);G.Combat.updateFields(dt);
    G.Pickups.update(dt,[p],o=>{if(o.kind==='xp'||o.kind==='xpBig'){this.xp+=o.value;const n=p.gainXp(o.value);if(this.config.strategic)this.pending+=n;}else if(o.kind==='heal')p.heal(p.maxHp*.2);});
    this.frame++;
  }
  chooseUpgrade(choice) {
    if(!this.pending||![0,1,2].includes(choice))throw new Error('No pending upgrade or invalid choice');
    if(choice===0)this.player.atkMul*=1.15;
    if(choice===1)this.player.spdMul*=1.1;
    if(choice===2)this.player.addMaxHp(15);
    this.pending--;
  }
  get done(){return this.player.dead||this.frame>=Math.ceil(this.config.duration*60);}
}
