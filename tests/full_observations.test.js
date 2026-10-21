import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';

// Exercise the actual adapter with a minimal deterministic game fixture.
function fixture(version='v2', rewardWeights) {
  const enemies=[],pickups=[],projectiles=[],hazards=[];
  const player={x:0,y:0,r:10,hp:100,maxHp:100,xp:0,xpNext:10,level:1,kills:0,xpMul:1,
    slots:[],mon:{types:['water']},hasItem:()=>false,
    hurt(n){this.hp=Math.max(0,this.hp-n);this.dead=this.hp===0;},
    heal(n){this.hp=Math.min(this.maxHp,this.hp+n);},gainXp(n){this.xp+=n;}};
  const G={
    U:{TYPE_COLOR:{water:'#00f'},dist2:(x,y,a,b)=>(x-a)**2+(y-b)**2},
    LevelUp:{STATS:[]},Items:{BY:{}},Moves:{BY_ID:{}},
    Input:{},Audio:{},RunUI:{},DB:{playAsGuest(){},save:{},grant(){}},DEX_BY:{7:{}},
    EnemyMgr:{all:()=>enemies},Projectiles:{all:()=>projectiles},
    Pickups:{all:()=>pickups,update(_dt,players,collect){
      for(let i=pickups.length-1;i>=0;i--)if(Math.hypot(pickups[i].x,pickups[i].y)<15)collect(pickups.splice(i,1)[0],players);
    }},
    World:{visibleProps:()=>[],isFree:()=>true,isLiquid:()=>false},
    Hazards:{all:hazards,inside:()=>false},Rift:{inArena:false},Weather:{KINDS:{}},
    Game:{player,st:{time:0},state:'playing',startRun(){},rlTick(){
      G.Game.st.time+=1/60;
      for(const e of enemies){e.x+=(e.testVx||0)/60;e.y+=(e.testVy||0)/60;}
      G.Pickups.update(1/60,player,p=>{if(p.kind==='heal')player.heal(30);else if(p.kind.startsWith('xp'))player.gainXp(p.value);});
    }}
  };
  const context=vm.createContext({G,window:{POKERL_TRAINING:true,pokerlSeed(){},pokerlRandom:()=>0.5}});
  vm.runInContext(fs.readFileSync(new URL('../full_game/training.js',import.meta.url),'utf8'),context);
  const api=context.window.fullTraining;
  api.start({seed:1,duration:120,observationVersion:version,rewardWeights});
  return {G,player,enemies,pickups,projectiles,hazards,api};
}
function block(f,name) {
  const obs=f.api.snapshot().observation;
  const {start,length}=f.api.layout()[name];
  return Array.from(obs.slice(start,start+length));
}

test('healing and utility pickups survive XP crowding; values and types are distinct',()=>{
  const f=fixture();
  for(let i=0;i<30;i++)f.pickups.push({x:i+20,y:0,kind:'xp',value:2});
  f.pickups.push({x:200,y:0,kind:'heal',value:0},{x:210,y:0,kind:'coin',value:5});
  const heal=block(f,'healing'),utility=block(f,'utility'),xp=block(f,'xp');
  assert.equal(heal[0],1);assert.equal(heal[1],0.2);assert.equal(heal[4],0.3);
  assert.equal(heal[8],1);assert.equal(utility[11],1);
  assert.equal(xp[3],0.02);
  assert.equal(heal.length,5*16);assert.ok(heal.slice(16).every(x=>x===0));
  const len=f.api.snapshot().observation.length;
  f.pickups.length=0;
  assert.equal(f.api.snapshot().observation.length,len);
});

test('hostile projectiles stay visible among friendly shots; expired hazards are excluded',()=>{
  const f=fixture();
  for(let i=0;i<30;i++)f.projectiles.push({x:i,y:0,vx:10,vy:0,friendly:true});
  f.projectiles.push({x:300,y:0,vx:-100,vy:0,r:6,dmg:20,life:2,t:0,friendly:false});
  f.hazards.push({x:0,y:0,kind:'zone',r:40,hit:true,delay:1,t:2},
    {x:10,y:0,kind:'zone',r:50,delay:1,t:0.25});
  const shots=block(f,'hostileProjectiles');
  assert.equal(shots[0],1);assert.equal(shots[1],0.3);assert.equal(shots[3],-0.2);
  const hazards=block(f,'activeHazards');
  assert.equal(hazards[1],0.01);assert.equal(hazards[9],0.15);assert.equal(hazards[12],0);
});

test('enemy motion is measured through ticks, attack state is encoded, snapshots do not change it',()=>{
  const f=fixture();
  f.enemies.push({x:100,y:0,hp:10,maxHp:10,r:12,spd:50,dmg:2,boss:false,flying:false,def:{range:200},
    keepsDistance:()=>true,shotCd:2,phase:'aim',pt:0.5,testVx:-50,testVy:20});
  assert.equal(block(f,'enemyDanger')[6],0); // Newly seen, no velocity sample yet.
  f.api.step(0);
  const features=block(f,'enemyDanger');
  assert.ok(Math.abs(features[4]+0.1)<1e-10);
  assert.ok(Math.abs(features[5]-0.04)<1e-10);
  assert.equal(features[6],1);assert.equal(features[9],0.4);assert.equal(features[14],1);
  assert.deepEqual(block(f,'enemyDanger'),features);
  const length=f.api.snapshot().observation.length;
  f.enemies[0].ai={cd:1,cast:{id:'beam',wind:0.6,t:0.2,ang:Math.PI/2}};
  f.enemies[0].jump={tx:123,ty:456,dur:0.65,t:-0.5};
  const attack=block(f,'enemyDanger');
  assert.equal(attack[22],1);assert.equal(attack[27],1);assert.equal(attack[31],1);
  assert.ok(Math.abs(attack[30]-0.08)<1e-10);
  assert.equal(attack[33],1);assert.equal(attack[34],0.123);
  assert.ok(Math.abs(attack[36]-0.23)<1e-10);
  assert.equal(f.api.snapshot().observation.length,length);
  f.enemies.length=0;assert.equal(f.api.snapshot().observation.length,length);
});

test('legacy prefix is preserved and observation variants have fixed dimensions',()=>{
  const legacy=fixture('v1').api.snapshot().observation;
  for(const version of ['v2-pickups','v2']){
    const f=fixture(version),obs=f.api.snapshot().observation;
    assert.deepEqual(Array.from(obs.slice(0,legacy.length)),Array.from(legacy));
    assert.equal(obs.length-legacy.length,version==='v2-pickups'?560:1722);
    assert.ok(obs.every(Number.isFinite));
  }
});

test('reward components sum to return; healing counts actual recovery and adds no direct bonus',()=>{
  const f=fixture('v2',{survival:0.2,xp:0.1,kills:0.1,damage:0.2,death:20});
  f.player.hp=95;
  f.pickups.push({x:0,y:0,kind:'heal',value:0},{x:0,y:0,kind:'xp',value:4});
  const step=f.api.step(0);
  assert.equal(step.info.healed,5);assert.equal(step.info.healingPickups,1);assert.equal(step.info.xpPickups,1);
  assert.ok(Math.abs(step.reward-0.42)<1e-10);
  assert.ok(Math.abs(Object.values(step.info.rewardComponents).reduce((a,b)=>a+b,0)-step.reward)<1e-10);
  assert.deepEqual(f.api.snapshot().info.rewardComponents,step.info.rewardComponents);
});

test('damage and death penalties are charged once at the terminal transition',()=>{
  const f=fixture('v2',{survival:0.2,xp:0.1,kills:0.1,damage:0.2,death:20});
  f.G.Game.rlTick=()=>{f.G.Game.st.time+=1/60;f.player.hurt(100);};
  const step=f.api.step(0);
  assert.equal(step.terminated,true);
  assert.equal(step.info.rewardComponents.damage,-20);
  assert.equal(step.info.rewardComponents.death,-20);
  assert.ok(Math.abs(step.reward-(-40+0.2/60))<1e-10);
  assert.deepEqual(f.api.snapshot().info.rewardComponents,step.info.rewardComponents);
  assert.throws(()=>f.api.step(0),/Episode ended/);
});
