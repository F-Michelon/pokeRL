/* Full-game structured training adapter. Loaded by Python only in ?training=1.
   Gameplay runs through the upstream Game.update loop; no arena substitutes. */
(() => {
  if (!window.POKERL_TRAINING) throw new Error('Training page required');
  const AXES = [[0,0],[0,-1],[0,1],[-1,0],[1,0],[-1,-1],[1,-1],[-1,1],[1,1]].map(([x,y])=>{const n=Math.hypot(x,y)||1;return[x/n,y/n];});
  const TYPES = Object.keys(G.U.TYPE_COLOR);
  const KINDS = ['projectile','melee','beam','orbit','nova','aura','buff','boomerang','chain','mine','meteor','turret','dash','trail','cone'];
  const PROPS = ['tree','rock','grass','chest','bigchest','spring','altar','trap'];
  const PICKUPS = ['xp','xpBig','heal','magnet','bomb','coin','item','ticket','ticket10'];
  const PHASES = ['walk','aim','dash','fuse'];
  const CASTS = ['slam','dash','leap','rain','beam','nova','volley'];
  const LEGACY_REWARD = {survival:0.1,xp:0.1,kills:0.5,damage:0.1,death:10};
  const STATS = G.LevelUp.STATS.map(s=>s.id);
  const ITEM_IDS = Object.keys(G.Items.BY).sort();
  const MOVE_IDS = Object.keys(G.Moves.BY_ID).sort();
  const onehot = (keys,key) => keys.map(k=>Number(k===key));
  const norm = (v,k=1) => Number.isFinite(v)?Math.max(-10,Math.min(10,v/k)):0;
  let pending=null,axis=[0,0],damage=0,xp=0,seed=0,limit=120,decisions=0;
  let counts={upgrades:0,moveSwitches:0,itemChoices:0,bondChoices:0,rerolls:0};
  let observationVersion='v2',rewardWeights={...LEGACY_REWARD},velocities=new Map();
  let healed=0,healingPickups=0,xpPickups=0,rewardTotals={};
  let observationLayout={};
  G.Input.axis=()=>axis;
  // Audio and animation rendering are presentation only, and are not advanced.
  G.Audio.sfx=()=>{}; G.Audio.music=()=>{};
  G.RunUI.showLevelUp=(_level,cards,callback)=>{pending={kind:'upgrade',cards,callback};};
  G.RunUI.showItemChoice=(cur,next,callback)=>{pending={kind:'item',cards:[cur,next],callback};};
  G.RunUI.showBond=(opt,callback)=>{pending={kind:'bond',cards:opt.loose,opt,callback,first:null};};

  function cardFeatures(card,kind) {
    const isUpgrade=kind==='upgrade';
    const id=isUpgrade?card.key.split(':').slice(1).join(':'):kind==='bond'?card.id:card;
    const move=(isUpgrade&&card.type!=='stat')||kind==='bond' ? G.Moves.BY_ID[id] : null;
    return [1,...onehot(['new','up','evo','stat','item','bond'],isUpgrade?card.type:kind),
      ...onehot(STATS,isUpgrade&&card.type==='stat'?id:null),
      ...moveFeatures(move),...onehot(ITEM_IDS,kind==='item'?id:null)];
  }
  function moveFeatures(m) {
    return [Number(!!m),norm(m?.dmg,100),norm(m?.cd,10),norm(m?.radius,500),norm(m?.count,10),
      norm(m?.lvl,5),norm(m?.t,10),norm(m?.speed,500),...onehot(TYPES,m?.type),...onehot(KINDS,m?.kind),
      // Stable pinned-catalog identity; stats + type/kind carry semantic features.
      m?norm(MOVE_IDS.indexOf(m.id)+1,MOVE_IDS.length):0];
  }
  function metrics(){const p=G.Game.player;return {seed,survival:G.Game.st.time,hp:p.hp,hpFraction:p.hp/p.maxHp,level:p.level,kills:p.kills,xp,damage,healed,healingPickups,xpPickups,
    dead:(p.dead&&!G.Rift.inArena)||G.Game.state==='over',state:G.Game.state,decision:pending?.kind||'movement',decisions,...counts};}
  function masks(){
    const p=G.Game.player,m=Array(50).fill(false);
    if(pending){for(let i=0;i<Math.min(4,pending.cards.length);i++)m[45+i]=true;
      m[49]=pending.kind==='upgrade'&&p.rerolls<1+(p.rerollBonus||0);
    }else for(let slot=0;slot<5;slot++)if(slot===0||(p.slots[slot-1]&&(!p.hasItem('choiceband')||p.active===slot-1)))for(let a=0;a<9;a++)m[slot*9+a]=true;
    return m;
  }
  function observation(){
    const p=G.Game.player;
    const v=[norm(p.x,3000),norm(p.y,3000),p.hp/p.maxHp,p.xp/p.xpNext,norm(p.level,30),
      norm(p.svx,200),norm(p.svy,200),G.Game.st.time/limit,norm(p.maxHp,300),norm(p.atk,5),
      norm(p.spd,200),norm(p.cdMul),norm(p.reduction),norm(p.magnet,300),norm(p.active,3),
      ...onehot(['movement','upgrade','item','bond'],pending?.kind||'movement'),...onehot(TYPES,p.mon.types[0])];
    const nearest=arr=>arr.filter(e=>!e.dead).sort((a,b)=>G.U.dist2(p.x,p.y,a.x,a.y)-G.U.dist2(p.x,p.y,b.x,b.y)).slice(0,20);
    const slots=(arr,n,width,fn)=>{for(let i=0;i<n;i++)v.push(...(arr[i]?fn(arr[i]):Array(width).fill(0)));};
    observationLayout={};
    // Keep the original prefix byte-for-byte compatible with v1 checkpoints.
    slots(nearest(G.EnemyMgr.all().slice()),20,9+TYPES.length,e=>[1,norm(e.x-p.x,1000),norm(e.y-p.y,1000),e.hp/e.maxHp,norm(e.maxHp,300),norm(e.spd,200),norm(e.dmg,50),Number(e.boss),Number(e.keepsDistance()),...onehot(TYPES,e.atkType)]);
    slots(nearest(G.Projectiles.all().slice()),20,6,e=>[1,norm(e.x-p.x,1000),norm(e.y-p.y,1000),norm(e.vx,500),norm(e.vy,500),Number(e.friendly)]);
    slots(nearest(G.Pickups.all().slice()),20,4,e=>[1,norm(e.x-p.x,1000),norm(e.y-p.y,1000),Number(e.kind.startsWith('xp'))]);
    const cam={left:()=>p.x-500,top:()=>p.y-500,w:1000,h:1000,sees:()=>true};
    slots(nearest(G.World.visibleProps(cam)),20,7+PROPS.length,e=>[1,norm(e.x-p.x,1000),norm(e.y-p.y,1000),norm(e.hp,100),norm(e.charge),Number(e.used),norm(e.open,5),...onehot(PROPS,e.type)]);
    slots(nearest(G.Hazards.all.slice()),20,11,h=>[1,norm(h.x-p.x,1000),norm(h.y-p.y,1000),Number(h.kind==='beam'),
      norm(h.r,300),norm(h.len,500),norm(h.w,100),Math.sin(h.angle||0),Math.cos(h.angle||0),norm(h.delay-h.t,5),norm(h.dmg,100)]);
    const rift=G.Rift.rift;
    v.push(Number(!!rift),norm(rift?rift.x-p.x:0,1000),norm(rift?rift.y-p.y:0,1000),Number(G.Rift.inArena),norm(G.Rift.timeLeft,180),
      ...onehot(Object.keys(G.Weather.KINDS),G.Weather.current?.kind));
    for(let y=-2;y<=2;y++)for(let x=-2;x<=2;x++){
      const wx=p.x+x*32,wy=p.y+y*32;v.push(Number(!G.World.isFree(wx,wy,p.r)),Number(G.World.isLiquid(wx,wy)));
    }
    for(let i=0;i<4;i++){
      const group=p.slots[i]||[];
      // Include every move in a fused slot, not only its first attack.
      for(let j=0;j<4;j++)v.push(...moveFeatures(group[j]));
    }
    const width=cardFeatures({key:'stat:hp',type:'stat'},'upgrade').length;
    slots(pending?.cards||[],4,width,c=>cardFeatures(c,pending.kind));
    v.push(...onehot(ITEM_IDS,p.item),...masks().map(Number));
    const block=(name,fn)=>{const start=v.length;fn();observationLayout[name]={start,length:v.length-start};};
    if(observationVersion!=='v1'){
      const pickups=G.Pickups.all().slice();
      const encodePickup=e=>[1,norm(e.x-p.x,1000),norm(e.y-p.y,1000),
        norm(typeof e.value==='number'?e.value:0,100),
        e.kind==='heal'?0.3:0,Number(!!e.pulled),...onehot(PICKUPS,e.kind),
        e.kind==='item'?norm(ITEM_IDS.indexOf(e.value)+1,ITEM_IDS.length):0];
      // Independent budgets: an XP carpet cannot hide healing or utility drops.
      block('xp',()=>slots(nearest(pickups.filter(e=>e.kind.startsWith('xp'))),20,16,encodePickup));
      block('healing',()=>slots(nearest(pickups.filter(e=>e.kind==='heal')),5,16,encodePickup));
      block('utility',()=>slots(nearest(pickups.filter(e=>!e.kind.startsWith('xp')&&e.kind!=='heal')),10,16,encodePickup));
    }
    if(observationVersion==='v2'){
      block('playerDanger',()=>v.push(norm(p.r,100),norm(p.invuln,5)));
      block('enemyDanger',()=>slots(nearest(G.EnemyMgr.all().slice()),20,38,e=>{
        const velocity=velocities.get(e),cast=e.ai?.cast,jump=e.jump;
        return [1,norm(e.x-p.x,1000),norm(e.y-p.y,1000),norm(e.r,100),
          norm(velocity?.vx,500),norm(velocity?.vy,500),Number(!!velocity),
          norm(Math.max(0,e.touchCd||0),5),Number(e.keepsDistance()),
          norm(Math.max(0,e.shotCd||0),5),norm(e.def?.range,500),
          Number(e.behavior==='jumper'||e.behavior==='shielder'),
          norm(Math.max(0,e.skillCd||0),10),...onehot(PHASES,e.phase),
          norm(Math.max(0,e.pt||0),5),e.phase==='dash'?norm(e.dx):0,e.phase==='dash'?norm(e.dy):0,
          Number(!!e.ai),norm(Math.max(0,e.ai?.cd||0),5),Number(!!cast),
          ...onehot(CASTS,cast?.id),norm(cast?Math.max(0,cast.wind-cast.t):0,5),
          cast?Math.sin(cast.ang):0,cast?Math.cos(cast.ang):0,Number(!!jump),
          norm(jump?jump.tx-p.x:0,1000),norm(jump?jump.ty-p.y:0,1000),
          norm(jump?Math.max(0,jump.dur-jump.t):0,5),Number(!!e.flying)];
      }));
      block('hostileProjectiles',()=>slots(nearest(G.Projectiles.all().filter(e=>!e.friendly)),20,8,e=>
        [1,norm(e.x-p.x,1000),norm(e.y-p.y,1000),norm(e.vx,500),norm(e.vy,500),norm(e.r,100),norm(e.dmg,100),norm(e.life-e.t,5)]));
      block('activeHazards',()=>slots(nearest(G.Hazards.all.filter(h=>!h.hit)),20,12,h=>
        [1,norm(h.x-p.x,1000),norm(h.y-p.y,1000),Number(h.kind==='beam'),norm(h.r,300),
          norm(h.len,500),norm(h.w,100),Math.sin(h.angle||0),Math.cos(h.angle||0),
          norm(Math.max(0,h.delay-h.t),5),norm(h.dmg,100),Number(G.Hazards.inside(h,p.x,p.y,p.r))]));
    }
    if(!v.every(Number.isFinite))throw new Error('Non-finite full-game observation');
    return v;
  }
  function result(before=null){
    const info=metrics(),parts={survival:0,xp:0,kills:0,damage:0,death:0};
    if(before){
      for(const key of ['survival','xp','kills'])parts[key]=rewardWeights[key]*(info[key]-before[key]);
      parts.damage=-rewardWeights.damage*(info.damage-before.damage);
      parts.death=info.dead&&!before.dead?-rewardWeights.death:0;
      for(const key of Object.keys(parts))rewardTotals[key]+=parts[key];
    }
    info.rewardComponents={...rewardTotals};
    for(const key of Object.keys(rewardTotals))info['reward_'+key]=rewardTotals[key];
    return {observation:observation(),mask:masks(),info,reward:Object.values(parts).reduce((a,b)=>a+b,0),
      terminated:info.dead,truncated:!info.dead&&(info.survival>=limit||decisions>=Math.ceil(limit*10)+1000)};
  }
  function start(config){
    seed=config.seed;limit=config.duration;pending=null;axis=[0,0];damage=xp=decisions=0;
    observationVersion=config.observationVersion||'v2';
    if(!['v1','v2-pickups','v2'].includes(observationVersion))throw new Error('Unknown observation version');
    rewardWeights={...LEGACY_REWARD,...config.rewardWeights};
    if(Object.keys(rewardWeights).some(k=>!(k in LEGACY_REWARD)||!Number.isFinite(rewardWeights[k])||rewardWeights[k]<0))throw new Error('Invalid reward weights');
    velocities=new Map();healed=healingPickups=xpPickups=0;
    rewardTotals=Object.fromEntries(Object.keys(LEGACY_REWARD).map(k=>[k,0]));
    counts={upgrades:0,moveSwitches:0,itemChoices:0,bondChoices:0,rerolls:0};
    window.pokerlClock=0;
    G.DB.playAsGuest();
    const dex=config.dex||7;
    if(!G.DEX_BY[dex])throw new Error('Unknown Pokémon');
    G.DB.save.partner=dex;G.DB.save.starter=dex;G.DB.grant(dex,'starter');
    G.Game.startRun();
    const p=G.Game.player,hurt=p.hurt.bind(p),gain=p.gainXp.bind(p),heal=p.heal.bind(p);
    p.hurt=(...args)=>{const before=p.hp;hurt(...args);damage+=Math.max(0,before-p.hp);};
    p.gainXp=n=>{xp+=n*p.xpMul;return gain(n);};
    p.heal=n=>{const before=p.hp;const out=heal(n);healed+=Math.max(0,p.hp-before);return out;};
    return result();
  }
  function step(action){
    if(result().terminated||result().truncated)throw new Error('Episode ended');
    if(!Number.isInteger(action)||!masks()[action])throw new Error(`Invalid full-game action ${action}`);
    const before=metrics();decisions++;
    if(action===49){G.Game.player.rerolls++;pending.cards=G.LevelUp.offer(G.Game.player);counts.rerolls++;}
    else if(action>=45){
      const current=pending,card=current.cards[action-45];
      if(current.kind==='bond'&&current.opt.mode==='pair'&&!current.first){current.first=card;current.cards=current.cards.filter(c=>c!==card);}
      else {
        pending=null;
        if(current.kind==='upgrade'){counts.upgrades++;current.callback(card);}
        if(current.kind==='item'){counts.itemChoices++;current.callback(card);}
        if(current.kind==='bond'){counts.bondChoices++;current.callback(current.first?[current.first,card]:[card]);}
      }
    }else{
      const slot=Math.floor(action/9)-1,p=G.Game.player;
      if(slot>=0&&slot!==p.active){p.setActive(slot);counts.moveSwitches++;}
      axis=AXES[action%9];
      for(let i=0;i<6&&!pending&&(!p.dead||G.Rift.inArena)&&G.Game.state!=='over'&&G.Game.st.time<limit;i++){
        const positions=observationVersion==='v2'?new Map(G.EnemyMgr.all().map(e=>[e,{x:e.x,y:e.y}])):null;
        window.pokerlClock+=1000/60;G.Game.rlTick();
        if(positions){
          velocities=new Map();
          for(const e of G.EnemyMgr.all()){
            const old=positions.get(e);
            // Respawn/teleport discontinuities are not physical velocity.
            if(old&&Math.hypot(e.x-old.x,e.y-old.y)<100)velocities.set(e,{vx:(e.x-old.x)*60,vy:(e.y-old.y)*60});
          }
        }
        if(G.Game.state!=='playing'&&!pending&&G.Game.state!=='over')throw new Error(`Unhandled game state ${G.Game.state}`);
      }
    }
    return result(before);
  }
  // Async upstream audio/asset/UI tasks must not consume simulation randomness.
  function seededCall(fn){const previous=Math.random;Math.random=window.pokerlRandom;try{return fn();}finally{Math.random=previous;}}
  const pickupUpdate=G.Pickups.update;
  G.Pickups.update=(dt,players,onCollect)=>pickupUpdate(dt,players,(pickup,player)=>{
    if(player===G.Game.player){
      if(pickup.kind==='heal')healingPickups++;
      if(pickup.kind.startsWith('xp'))xpPickups++;
    }
    return onCollect(pickup,player);
  });
  window.fullTraining={start:config=>{window.pokerlSeed(config.seed);return seededCall(()=>start(config));},
    step:action=>seededCall(()=>step(action)),snapshot:()=>result(),
    layout:()=>({...observationLayout}),version:'full-game-v2'};
})();
