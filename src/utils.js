
export function relu(v){return v.map(x=>Math.max(0,x));}
export function softmax(logits){
  const max = Math.max(...logits);
  const exps = logits.map(x=>Math.exp(x - max));
  const sum = exps.reduce((a,b)=>a+b,0);
  return exps.map(x=>x/sum);
}
export function matVecMul(W, x){
  // W: rows x cols, x: cols
  const rows = W.length;
  const cols = W[0].length;
  const out = new Array(rows).fill(0);
  for(let i=0;i<rows;i++){
    let s = 0;
    const Wi = W[i];
    for(let j=0;j<cols;j++) s += Wi[j]*x[j];
    out[i] = s;
  }
  return out;
}
export function addVec(a,b){ return a.map((v,i)=>v+b[i]); }
export function argmax(a){
  let bi=0, bv=a[0];
  for(let i=1;i<a.length;i++){ if(a[i]>bv){bv=a[i];bi=i;} }
  return bi;
}
export function toPct(x){ return (x*100).toFixed(1)+'%'; }
