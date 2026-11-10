// Exact CPU equivalence test for legacy reindex followed by encode gather
// versus fused gather, across both published polynomial dimensions.
'use strict';
function rev(v,bits) {let r=0;for(let i=0;i<bits;i++){r=(r<<1)|(v&1);v>>>=1;}return r;}
function sample(n,l) {
  const bits=Math.log2(n),words=n*l;
  const a=new Uint32Array(words), b=new Uint32Array(words);
  for(let j=0;j<words;j++){a[j]=(Math.imul(j,2654435761)^0x12345678)>>>0;b[j]=(Math.imul(j,2246822519)^0x87654321)>>>0;}
  const oldA=a.slice(),oldB=b.slice();
  for(let i=0;i<n;i++){const ri=rev(i,bits);if(ri<=i)continue;for(let j=0;j<l;j++){
    let x=i*l+j,y=ri*l+j;
    [oldA[x],oldA[y]]=[oldA[y],oldA[x]];
    [oldB[x],oldB[y]]=[oldB[y],oldB[x]];
  }}
  let comparisons=0;
  for(let start=0;start<n;start+=32) {
    for(let row=0;row<Math.min(32,n-start);row++) {
      const src=rev(start+row,bits);
      for(let col=0;col<l;col++) {
        const oldIdx=(start+row)*l+col,newIdx=src*l+col;
        if(oldA[oldIdx]!==a[newIdx] || oldB[oldIdx]!==b[newIdx])throw Error('Mismatch '+n+' row '+row+' col '+col);
        comparisons+=2;
      }
    }
  }
  console.log('PASS N='+n+' limbs='+l+' compared='+comparisons);
}
sample(32768,28);sample(65536,55);
console.log('CPU reindex equivalence PASS. CUDA runtime/performance still UNTESTED.');
