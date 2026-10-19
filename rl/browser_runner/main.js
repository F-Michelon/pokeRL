import {Environment} from '../environment.js';
import {ACTIONS} from '../arena.js';
import {seededRandom} from '../seeds.js';
import {GameRenderer} from './renderer.js';
const params=new URLSearchParams(location.search),seed=Number(params.get('seed')||0);
const config=params.has('config')?await (await fetch(`/configs/${params.get('config')}.json`)).json():{};
const env=new Environment(config);env.reset(seed);
// Teacher-only interface; the student runner never reads it.
window.teacher={observation:()=>env.getObservation(),metrics:()=>env.getMetrics()};
const canvas=document.querySelector('canvas'),ctx=canvas.getContext('2d');
const graphics=params.get('graphics')||'upstream';
if(!['upstream','debug'].includes(graphics))throw new Error('Unknown graphics mode');
const renderer=graphics==='upstream'?new GameRenderer(window.G,seed):null;
if(renderer)await renderer.ready();
await document.fonts.ready;
const keys=new Set();let started=false,last=null,acc=0;
const random=seededRandom(seed^0xa50),aug=params.get('randomize')==='1';
const scale=aug?.85+random()*.3:1,brightness=aug?.8+random()*.4:1;
const renderFps=aug?[30,45,60][Math.floor(random()*3)]:60;
if(aug){canvas.width=[960,1280,1600][Math.floor(random()*3)];canvas.height=canvas.width*9/16;}
let lastDraw=-Infinity;
addEventListener('keydown',e=>{keys.add(e.code);if(e.code==='Space'){started=true;e.preventDefault();}if(e.code==='KeyR'){env.reset(seed);started=false;acc=0;}if(env.pending&&['Digit1','Digit2','Digit3'].includes(e.code))env.chooseUpgrade(Number(e.code.slice(-1))-1);});
addEventListener('keyup',e=>keys.delete(e.code));addEventListener('blur',()=>keys.clear());
function action(){const x=Number(keys.has('KeyD'))-Number(keys.has('KeyA')),y=Number(keys.has('KeyS'))-Number(keys.has('KeyW'));return ACTIONS.findIndex(([a,b])=>Math.sign(a)===x&&Math.sign(b)===y);}
function dot(x,y,r,color){ctx.fillStyle=color;ctx.beginPath();ctx.arc(x,y,r,0,Math.PI*2);ctx.fill();}
function draw(){
 ctx.setTransform(canvas.width/1280,0,0,canvas.height/720,0,0);
 if(renderer){renderer.draw(ctx,env,scale,brightness);}else{
 ctx.fillStyle='#172d28';ctx.fillRect(0,0,1280,720);ctx.save();ctx.filter=`brightness(${brightness})`;ctx.translate(640,360);ctx.scale(scale,scale);
 ctx.strokeStyle='#36534a';ctx.strokeRect(-600,-330,1200,660);
 for(const o of env.obstacles){ctx.fillStyle='#62777b';ctx.fillRect(o.x,o.y,o.w,o.h);}
 for(const e of env.G.Pickups.all())dot(e.x,e.y,4,e.kind.startsWith('xp')?'#67e9ff':'#5bffa2');
 for(const e of env.G.EnemyMgr.all())if(!e.dead)dot(e.x,e.y,e.r,e.behavior==='ranged'?'#e8a14b':'#ef6b73');
 for(const e of env.G.Projectiles.all())dot(e.x,e.y,Math.max(3,e.r),e.friendly?'#8dd9ff':'#ffcc44');
 dot(env.player.x,env.player.y,12,'#54c9ef');ctx.restore();
 ctx.fillStyle='#000a';ctx.fillRect(20,20,350,54);ctx.fillStyle='#fff';ctx.font='18px sans-serif';ctx.fillText(`${env.player.mon.name}   HP ${Math.ceil(env.player.hp)}   ${Math.floor(env.frame/60)}s   XP ${env.xp.toFixed(0)}`,30,52);
 }
 if(aug){ctx.fillStyle='#ffffff18';for(let i=0;i<15;i++)ctx.fillRect((i*173+env.frame)%1280,(i*113)%720,3,3);}
 const status=env.player.dead?'DEFEAT':env.done?'COMPLETE':env.pending?'LEVEL UP — 1: attack  2: speed  3: health':!started?'SPACE TO START':null;
 if(status){ctx.fillStyle='#07121be8';ctx.fillRect(280,270,720,140);ctx.fillStyle='#fff';ctx.font='24px sans-serif';ctx.fillText(status,310,345);}
 // Visual status swatch belongs to the local arena UI, not a state API.
 ctx.fillStyle=env.player.dead?'#ff0000':env.done?'#00ff00':env.pending?'#ff00ff':!started?'#0000ff':'#000000';ctx.fillRect(0,0,12,12);
}
function loop(now){if(last!==null&&started&&!env.done&&!env.pending){acc+=Math.min((now-last)/1000,.25);while(acc>=1/60&&!env.done&&!env.pending){env.step(action(),1);acc-=1/60;}}last=now;if(now-lastDraw>=1000/renderFps){draw();lastDraw=now;}requestAnimationFrame(loop);}
requestAnimationFrame(loop);window.arenaReady=true;
