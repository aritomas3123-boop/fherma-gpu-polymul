// Data-movement feasibility check for end-to-end FHERMA timing.
// This is an analytical lower bound, NOT a benchmark or guaranteed speed.
const points=[{name:'Ranking',N:65536,W:1747,L:55},{name:'Secondary',N:32768,W:868,L:28}];
const throughputs=[64,56,48]; // hypothetical one-direction effective PCIe GB/s
const leaderUs=828, targetUs=400;
for(const p of points) {
 const bytes=p.N*p.L*4;
 const trip=bytes*3;
 const minA=trip/(64e9)*1e6;
 console.log(p.name,'N='+p.N,'W='+p.W,'L='+p.L);
 console.log('  one tensor', (bytes/1048576).toFixed(3),'MiB; 2 input + 1 output',(trip/1048576).toFixed(3),'MiB');
 for(const bw of throughputs)console.log('  transfer-only optimistic @'+bw+' GB/s:',(trip/(bw*1e9)*1e6).toFixed(1),'us');
 console.log('  aggregate required for 400 us:',(trip/(targetUs*1e-6)/1e9).toFixed(1),'GB/s');
 console.log('  NOTE: assumes transfers need PCIe; ignores GPU compute and launch overhead.');
}
console.log('Important: simultaneous duplex/asynchronous overlap, mapped memory and caching can change the effective limit; not a proof of impossibility.');
