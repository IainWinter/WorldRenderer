import init, { worker_main } from './pkg/worldrenderer.js';

await init();
worker_main();
