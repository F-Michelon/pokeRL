/* Separate local accounts, settings and progression from other game entrypoints. */
(() => {
  const storage = window.localStorage;
  const prefix = 'pokerl.fullgame.';
  const keys = () => Array.from({length: storage.length}, (_, i) => storage.key(i))
    .filter(key => key && key.startsWith(prefix));
  const isolated = {
    getItem(key) { return storage.getItem(prefix + String(key)); },
    setItem(key, value) { storage.setItem(prefix + String(key), String(value)); },
    removeItem(key) { storage.removeItem(prefix + String(key)); },
    clear() { for (const key of keys()) storage.removeItem(key); },
    key(index) { const key = keys()[Number(index)]; return key ? key.slice(prefix.length) : null; },
    get length() { return keys().length; }
  };
  Object.defineProperty(window, 'localStorage', {value: Object.freeze(isolated)});
})();
