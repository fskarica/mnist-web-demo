
import {MNISTNet} from './nn.js';
import {argmax, toPct} from './utils.js';

let net=null;
let draw, off, ctx, offCtx;
let invert = true; // MNIST is white digit on black

async function loadWeights(){
  const res = await fetch('./assets/weights/mnist_weights.json');
  return await res.json();
}

function setupCanvas(){
  draw = document.getElementById('draw');
  ctx = draw.getContext('2d');
  draw.width = draw.height = 280; // nice & big; we will downscale
  ctx.fillStyle = '#000';
  ctx.fillRect(0,0,draw.width,draw.height);
  ctx.lineCap='round';
  ctx.lineJoin='round';
  ctx.lineWidth=26;
  ctx.strokeStyle='#fff';

  off = document.createElement('canvas');
  off.width = off.height = 28;
  offCtx = off.getContext('2d');
}

let painting=false;
function pointerPos(e){
  const rect = draw.getBoundingClientRect();
  const x = (e.touches? e.touches[0].clientX : e.clientX) - rect.left;
  const y = (e.touches? e.touches[0].clientY : e.clientY) - rect.top;
  return {x,y};
}
function startDraw(e){ painting=true; const p=pointerPos(e); ctx.beginPath(); ctx.moveTo(p.x,p.y); e.preventDefault(); }
function moveDraw(e){ if(!painting) return; const p=pointerPos(e); ctx.lineTo(p.x,p.y); ctx.stroke(); e.preventDefault(); }
function endDraw(){ painting=false; }

function clearCanvas(){
  ctx.fillStyle='#000'; ctx.fillRect(0,0,draw.width,draw.height);
}

function getInput784(){
  // downscale 280->28 and convert to grayscale 0..1 where 1=white ink on black
  offCtx.clearRect(0,0,28,28);
  offCtx.drawImage(draw, 0,0,28,28);
  const img = offCtx.getImageData(0,0,28,28).data;
  const x = new Array(28*28);
  for(let i=0;i<28*28;i++){
    const r = img[4*i], g = img[4*i+1], b = img[4*i+2];
    const v = (0.299*r + 0.587*g + 0.114*b)/255; // 0..1
    x[i] = invert ? v : (1-v);
  }
  return x;
}

function renderProbs(probs){
  const grid = document.getElementById('probs');
  grid.innerHTML='';
  const best = argmax(probs);
  for(let d=0; d<10; d++){
    const p = document.createElement('div');
    p.className='prob'+(d===best?' best':'');
    p.innerHTML = \`
      <div><strong>\${d}</strong></div>
      <div class="bar"><span style="width:\${(probs[d]*100).toFixed(1)}%"></span></div>
      <div><small>\${toPct(probs[d])}</small></div>
    \`;
    grid.appendChild(p);
  }
  const guess = document.getElementById('guess');
  guess.textContent = String(best);
  const conf = document.getElementById('confidence');
  conf.textContent = toPct(probs[best]);
}

async function predict(){
  const x = getInput784();
  const {probs} = net.forward(x);
  renderProbs(probs);
}

function setupUI(){
  document.getElementById('btn-predict').addEventListener('click', predict);
  document.getElementById('btn-clear').addEventListener('click', clearCanvas);
  document.getElementById('toggle-invert').addEventListener('change', (e)=>{
    invert = e.target.checked;
    predict();
  });
  draw.addEventListener('mousedown', startDraw);
  draw.addEventListener('mousemove', moveDraw);
  window.addEventListener('mouseup', endDraw);
  draw.addEventListener('touchstart', startDraw, {passive:false});
  draw.addEventListener('touchmove', moveDraw, {passive:false});
  draw.addEventListener('touchend', endDraw, {passive:false});
}

function fillHowItWorks(){
  const el = document.getElementById('how');
  el.innerHTML = `
  <div class="note"><strong>Model:</strong> A compact 2-layer neural net (784 → H → 10) trained on MNIST.
  The app loads your exported <code>W1, b1, W2, b2</code> and runs pure‑JS inference in the browser.</div>

  <h3>Preprocessing</h3>
  <p>We draw at 280×280 px for a smooth pen, then downscale to 28×28 and convert to grayscale in [0,1].
  MNIST uses white digits on black, so the <span class="kbd">Invert</span> toggle keeps the polarity consistent.</p>

  <pre><code>// Downscale and normalize
offCtx.drawImage(draw, 0, 0, 28, 28);
const img = offCtx.getImageData(0,0,28,28).data;
const x = Array.from({length:784}, (_,i)=>{
  const r=img[4*i], g=img[4*i+1], b=img[4*i+2];
  const v=(0.299*r+0.587*g+0.114*b)/255;
  return invert ? v : (1-v);
});</code></pre>

  <h3>Forward pass</h3>
  <pre><code>// h = ReLU(W1·x + b1)
const h = relu(addVec(matVecMul(W1,x), b1));
// logits = W2·h + b2; probs = softmax(logits)
const logits = addVec(matVecMul(W2,h), b2);
const probs  = softmax(logits);</code></pre>

  <h3>Prediction</h3>
  <p>All ten class probabilities are shown. The most confident class is highlighted.</p>
  `;
}

function setupTabs(){
  const tabs = [...document.querySelectorAll('.tab')];
  const sections = tabs.map(t=>document.getElementById(t.dataset.for));
  tabs.forEach((t,i)=>t.addEventListener('click', ()=>{
    tabs.forEach(tt=>tt.classList.remove('active'));
    sections.forEach(s=>s.hidden=true);
    t.classList.add('active'); sections[i].hidden=false;
  }));
  tabs[0].click();
}

(async function init(){
  setupCanvas();
  setupUI();
  setupTabs();
  fillHowItWorks();
  const weights = await loadWeights();
  net = new MNISTNet(weights);
  await predict();
})();
