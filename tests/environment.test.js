import test from 'node:test';
import assert from 'node:assert/strict';
import {Environment} from '../rl/environment.js';
import {OBS_SIZE} from '../rl/observations.js';
import {seededRandom} from '../rl/seeds.js';
function episode(e,seed){const rng=seededRandom(seed+1000);e.reset(seed);const out=[];while(!e.done){if(e.pending)e.chooseUpgrade(2);out.push(e.step(Math.floor(rng()*9)));}return out;}
test('deterministic resets, isolated interleaved environments, all curriculum stages',()=>{
  for(let stage=0;stage<5;stage++){
    const config={stage,duration:12,spawnEvery:.2};const a=new Environment(config),b=new Environment(config);
    const one=episode(a,42);episode(a,9);assert.deepEqual(episode(a,42),one);
    assert.deepEqual(episode(b,42),one);assert.notDeepEqual(episode(b,43),one);
    a.reset(1);b.reset(1);for(let i=0;i<30;i++)assert.deepEqual(a.step(i%9),b.step(i%9));
  }
});
test('frame grouping and observation contract',()=>{const a=new Environment(),b=new Environment();a.reset(0);b.reset(0);const r=a.step(8,60);let total=0;for(let i=0;i<10;i++)total+=b.step(8).reward;assert.deepEqual(a.getObservation(),b.getObservation());assert.ok(Math.abs(total-r.reward)<1e-10);assert.equal(r.observation.length,OBS_SIZE);});
test('death, truncation, invalid requests and upgrade boundary',()=>{
 const e=new Environment({duration:.1});e.reset(1);assert.throws(()=>e.step(9));assert.throws(()=>e.step(0,0));assert.throws(()=>e.reset(-1));assert.throws(()=>new Environment({stage:5}));
 assert.equal(e.step(0).truncated,true);assert.throws(()=>e.step(0));
 e.reset(1);e.player.hurt(10000);assert.equal(e.getMetrics().dead,true);assert.throws(()=>e.step(0));
 e.reset(1);e.pending=1;assert.throws(()=>e.step(0));e.chooseUpgrade(2);assert.equal(e.player.maxHp,105);assert.equal(e.pending,0);
});
