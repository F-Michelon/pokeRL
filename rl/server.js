import readline from 'node:readline';
import {Environment} from './environment.js';
let env;
for await (const line of readline.createInterface({input:process.stdin,crlfDelay:Infinity})) {
  try {
    const msg=JSON.parse(line);let result;
    switch(msg.op){
      case 'configure':env=new Environment(msg.config);result={ok:true};break;
      case 'reset':result={observation:env.reset(msg.seed),info:env.getMetrics()};break;
      case 'step':result=env.step(msg.action,msg.frames);break;
      case 'upgrade':env.chooseUpgrade(msg.choice);result={observation:env.getObservation()};break;
      case 'metrics':result=env.getMetrics();break;
      case 'close':process.exit(0);break;
      default:throw new Error('Unknown operation');
    }
    process.stdout.write(JSON.stringify(result)+'\n');
  } catch(e) {process.stdout.write(JSON.stringify({error:e.message})+'\n');}
}
