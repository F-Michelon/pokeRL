/** Read-only presentation using the pinned game's original graphics modules. */
export class GameRenderer {
  constructor(assets, seed) {
    this.assets = assets;
    this.seed = seed;
    this.animations = new WeakMap();
    this.chunks = new Map();
    this.lastFrame = 0;
  }

  async ready() {
    await this.assets.Sprites.preload([1, 4, 7, 19, 25]);
    for (const dex of [1, 4, 7, 19, 25]) {
      if (!this.assets.Sprites.sheet(dex, 'Idle').ready) {
        throw new Error(`Could not load Pokémon ${dex}. Run python3 scripts/bootstrap.py.`);
      }
    }
  }

  floor(ctx, player) {
    const {Tiles, U} = this.assets;
    // Original forest tile painter, with a deliberately obstacle-free floor.
    // The arena's actual collision rectangles are drawn separately below.
    const chunkSize = 512;
    const cx = Math.floor(player.x / chunkSize), cy = Math.floor(player.y / chunkSize);
    for (let y = cy - 1; y <= cy + 1; y++) {
      for (let x = cx - 1; x <= cx + 1; x++) {
        const key = `${x},${y}`;
        if (!this.chunks.has(key)) {
          this.chunks.set(key, Tiles.paintChunk(x, y, 16, () => 'floor',
            (px, py) => .45 + .15 * Math.sin(px / 90) + .15 * Math.cos(py / 110) + U.hash(Math.floor(px / 80), Math.floor(py / 80), this.seed) * .1,
            0, this.seed));
        }
        ctx.drawImage(this.chunks.get(key), x * chunkSize, y * chunkSize, chunkSize, chunkSize);
      }
    }
  }

  entity(ctx, entity, player, dt) {
    const {Sprites} = this.assets;
    let state = this.animations.get(entity);
    if (!state) {
      state = {anim: new Sprites.Animator(entity.dex), cooldown: 0, hp: entity.hp, dead: false};
      this.animations.set(entity, state);
    }
    const moving = entity === player ? player.walking : Math.hypot(entity.vx || 0, entity.vy || 0) > 1;
    const angle = entity === player ? (player.attackFace > 0 ? player.aim : player.moveAngle) : Math.atan2(entity.vy || 0, entity.vx || 0);
    state.anim.dir = Sprites.dirFromAngle(angle);
    state.anim.loop(moving ? 'Walk' : 'Idle');
    if (entity.dead && !state.dead) state.anim.play('Faint', {hold: true});
    else if (entity.hp < state.hp && !entity.dead) state.anim.play('Hurt');
    else if (entity === player && player.activeMove().t > state.cooldown + .1) {
      state.anim.play(player.activeMove().kind === 'melee' ? 'Attack' : 'Shoot');
    }
    state.cooldown = entity === player ? player.activeMove().t : 0;
    state.hp = entity.hp;
    state.dead = entity.dead;
    state.anim.update(dt);
    Sprites.drawShadow(ctx, entity.dex, entity.x, entity.y);
    state.anim.draw(ctx, entity.x, entity.y, {flash: entity.flash > 0, softFlash: entity === player});
    if (entity !== player && entity.hp < entity.maxHp) {
      ctx.fillStyle = '#182024'; ctx.fillRect(entity.x - 15, entity.y - 42, 30, 4);
      ctx.fillStyle = '#ff6b75'; ctx.fillRect(entity.x - 15, entity.y - 42, 30 * Math.max(0, entity.hp / entity.maxHp), 4);
    }
  }

  draw(ctx, env, scale, brightness) {
    const p = env.player;
    const dt = Math.max(0, env.frame - this.lastFrame) / 60;
    this.lastFrame = env.frame;
    ctx.save();
    ctx.imageSmoothingEnabled = false;
    ctx.filter = `brightness(${brightness})`;
    ctx.translate(640, 360);
    ctx.scale(1.6 * scale, 1.6 * scale);
    ctx.translate(-p.x, -p.y);
    this.floor(ctx, p);
    // Visible arena boundaries correspond exactly to the simulation's rectangle.
    ctx.fillStyle = '#18251bd9';
    ctx.fillRect(-1800, -1200, 1200, 2400); ctx.fillRect(600, -1200, 1200, 2400);
    ctx.fillRect(-600, -1200, 1200, 870); ctx.fillRect(-600, 330, 1200, 870);
    ctx.strokeStyle = '#e5ce81'; ctx.lineWidth = 2; ctx.strokeRect(-600, -330, 1200, 660);
    for (const o of env.obstacles) {
      ctx.fillStyle = '#353e2e'; ctx.fillRect(o.x, o.y, o.w, o.h);
      ctx.fillStyle = '#687a51'; ctx.fillRect(o.x, o.y, o.w, o.h - 6);
      ctx.strokeStyle = '#a2b184'; ctx.strokeRect(o.x, o.y, o.w, o.h - 6);
    }
    for (const pickup of env.G.Pickups.all()) {
      const xp = pickup.kind.startsWith('xp');
      ctx.save(); ctx.translate(pickup.x, pickup.y - 4 + Math.sin(pickup.t) * 2);
      ctx.rotate(Math.PI / 4); ctx.fillStyle = xp ? '#173b73' : '#2d653d'; ctx.fillRect(-4, -4, 8, 8);
      ctx.fillStyle = xp ? '#8deaff' : '#9fff9d'; ctx.fillRect(-2, -2, 4, 4); ctx.restore();
    }
    const entities = [...env.G.EnemyMgr.all().filter(e => !e.dead), p].sort((a, b) => a.y - b.y);
    for (const e of entities) this.entity(ctx, e, p, dt);
    for (const projectile of env.G.Projectiles.all()) {
      const style = this.assets.VFX.forType(projectile.type || (projectile.friendly ? p.activeMove().type : 'electric'));
      this.assets.VFX.draw(ctx, style.proj, style.pal, projectile.x, projectile.y - 18,
        {angle: Math.atan2(projectile.vy, projectile.vx), frame: Math.floor(env.frame / 6)});
    }
    ctx.restore();
    this.hud(ctx, env);
  }

  hud(ctx, env) {
    const p = env.player;
    ctx.fillStyle = '#131b2ded'; ctx.fillRect(20, 20, 290, 90);
    ctx.strokeStyle = '#9fa9cb'; ctx.strokeRect(20, 20, 290, 90);
    ctx.font = '22px Pixelify, monospace'; ctx.fillStyle = '#fff4c2';
    ctx.fillText(`${p.mon.name}  Lv. ${p.level}`, 36, 48);
    ctx.fillStyle = '#39283e'; ctx.fillRect(36, 60, 258, 16);
    ctx.fillStyle = '#69cf81'; ctx.fillRect(36, 60, 258 * p.hp / p.maxHp, 16);
    ctx.font = '15px Pixelify, monospace'; ctx.fillStyle = '#fff'; ctx.fillText(`HP ${Math.ceil(p.hp)} / ${p.maxHp}`, 44, 73);
    ctx.fillStyle = '#344462'; ctx.fillRect(36, 90, 258, 6);
    ctx.fillStyle = '#88dfff'; ctx.fillRect(36, 90, 258 * p.xp / p.xpNext, 6);
    ctx.fillStyle = '#131b2ded'; ctx.fillRect(552, 20, 176, 52);
    ctx.fillStyle = '#fff4c2'; ctx.font = '28px Pixelify, monospace';
    ctx.fillText(`${Math.floor(env.frame / 3600)}:${String(Math.floor(env.frame / 60) % 60).padStart(2, '0')}`, 610, 55);
    ctx.fillStyle = '#131b2ded'; ctx.fillRect(455, 642, 370, 58);
    ctx.strokeStyle = '#8dcced'; ctx.strokeRect(455, 642, 370, 58);
    ctx.fillStyle = '#fff'; ctx.font = '20px Pixelify, monospace';
    ctx.fillText(`${p.activeMove().name || p.activeMove().id} · AUTO`, 475, 678);
  }
}
