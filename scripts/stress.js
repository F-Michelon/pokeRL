import {Environment} from '../rl/environment.js';
import {seededRandom} from '../rl/seeds.js';
import fs from 'node:fs';
const n=Number(process.argv[2]||1000),start=Date.now();let deaths=0;
for(let seed=0;seed<n;seed++){
 const env=new Environment({stage:seed%5,spawnEvery:.65,enemySpeed:70,enemyHp:28});const rng=seededRandom(seed);env.reset(seed);
 while(!env.done)env.step(Math.floor(rng()*9));
 deaths+=Number(env.player.dead);
 if(seed%100===99)console.error(`${seed+1}/${n} episodes`);
}
const result={episodes:n,deaths,seconds:(Date.now()-start)/1000,revision:JSON.parse(fs.readFileSync('upstream.lock.json')).revision};
fs.writeFileSync('results/validation.json',JSON.stringify(result,null,2)+'\n');console.log(result);
