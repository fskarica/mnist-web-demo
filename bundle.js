
// ===== utils.js =====
function relu(v){
  if (!v || v.some(x => isNaN(x))) {
    console.error('Invalid input to relu', v);
    return new Array(v ? v.length : 0).fill(0);
  }
  return v.map(x => Math.max(0, x));
}
function softmax(logits) {
    if (!logits || logits.length === 0) {
        console.error('Invalid logits input');
        return Array(10).fill(0.1);
    }

    // Check if all values are the same
    const allSame = logits.every(x => x === logits[0]);
    if (allSame) {
        console.warn('All logits are the same:', logits[0]);
    }

    // Numerical stability
    const max = Math.max(...logits);
    const exps = logits.map(x => Math.exp(Math.min(x - max, 20))); // limit for numerical stability
    const sum = exps.reduce((a, b) => a + b, 0);

    if (sum === 0 || !isFinite(sum)) {
        console.error('Invalid softmax computation');
        return Array(logits.length).fill(0.1);
    }

    const probs = exps.map(x => x/sum);
    
    // Validate output
    const probSum = probs.reduce((a, b) => a + b, 0);
    if (Math.abs(probSum - 1) > 1e-6) {
        console.error('Softmax probabilities do not sum to 1:', probSum);
    }

    return probs;
}
function matVecMul(W, x){
  if (!W || !x || !W[0]) {
    console.error('Invalid input to matVecMul', {W: !!W, x: !!x, 'W[0]': !!W[0]});
    return new Array(W ? W.length : 0).fill(0);
  }
  
  const rows = W.length;
  const cols = W[0].length;
  const out = new Array(rows).fill(0);
  
  if (x.length !== cols) {
    console.error(`Dimension mismatch: W[0] length = ${cols}, x length = ${x.length}`);
    return out;
  }
  
  for(let i=0; i<rows; i++){
    let s = 0;
    const Wi = W[i];
    for(let j=0; j<cols; j++) {
      if (isNaN(Wi[j]) || isNaN(x[j])) {
        console.error(`NaN detected at position [${i},${j}]`);
        continue;
      }
      s += Wi[j] * x[j];
    }
    out[i] = s;
  }
  return out;
}
function addVec(a,b){ return a.map((v,i)=>v+b[i]); }
function argmax(a){
  let bi=0, bv=a[0];
  for(let i=1;i<a.length;i++){ if(a[i]>bv){bv=a[i];bi=i;} }
  return bi;
}
function toPct(x){ return (x*100).toFixed(1)+'%'; }

// ===== nn.js =====
class MNISTNet {
    constructor(weights) {
    if (!weights || !weights.W1 || !weights.W2 || !weights.b1 || !weights.b2) {
      throw new Error('Invalid weights provided to MNISTNet constructor');
    }
        
        // Store and validate weights
        this.W1 = weights.W1;
        this.b1 = weights.b1;
        this.W2 = weights.W2;
        this.b2 = weights.b2;
        
        // Verify dimensions
        if (this.W1[0].length !== 784) {
            throw new Error(`W1 should have 784 columns, has ${this.W1[0].length}`);
        }
        if (this.W2.length !== 10) {
            throw new Error(`W2 should have 10 rows, has ${this.W2.length}`);
        }
        if (this.b2.length !== 10) {
            throw new Error(`b2 should have length 10, has ${this.b2.length}`);
        }
        
    console.log('Network initialized with correct dimensions');
    if (weights._isDefault) console.warn('Using default randomized weights (mnist_weights.json failed to load or was invalid)');

    // Log weight shapes and small samples to help debug
    try {
      console.log('Weight shapes:', {
        W1_rows: this.W1.length,
        W1_cols: this.W1[0].length,
        W2_rows: this.W2.length,
        W2_cols: this.W2[0].length,
        b1_len: this.b1.length,
        b2_len: this.b2.length
      });
      console.log('W1[0] sample:', this.W1[0].slice(0,10));
      console.log('b1 sample:', this.b1.slice(0,10));
      console.log('W2[0] sample:', this.W2[0].slice(0,10));
      console.log('b2 sample:', this.b2.slice(0,10));
    } catch (e) {
      console.warn('Could not log weight samples', e);
    }
    }

    forward(x784) {
        // Input validation
        if (!Array.isArray(x784) || x784.length !== 784) {
            console.error('Invalid input size:', x784?.length);
            return { logits: Array(10).fill(0), probs: Array(10).fill(0.1) };
        }
        
    // First layer
    const wx1 = matVecMul(this.W1, x784);
    console.log('First layer raw output length:', wx1.length, 'sample:', (wx1 && wx1.slice) ? wx1.slice(0,10) : wx1);
    if (!wx1.length) {
      console.error('First layer multiplication failed');
      return { logits: Array(10).fill(0), probs: Array(10).fill(0.1) };
    }

    const z1 = addVec(wx1, this.b1);
    // Log z1 before ReLU to check for NaNs
    console.log('z1 sample before ReLU:', (z1 && z1.slice) ? z1.slice(0,10) : z1);
    const h1 = relu(z1);
    console.log('Hidden layer h1 length:', h1.length, 'sample:', (h1 && h1.slice) ? h1.slice(0,10) : h1);
        
        // Second layer
        const wx2 = matVecMul(this.W2, h1);
        if (!wx2.length) {
            console.error('Second layer multiplication failed');
            return { logits: Array(10).fill(0), probs: Array(10).fill(0.1) };
        }
        
        const logits = addVec(wx2, this.b2);
        
        // Check logits
        if (logits.some(x => !isFinite(x))) {
            console.error('Invalid logits detected:', logits);
            return { logits: Array(10).fill(0), probs: Array(10).fill(0.1) };
        }
        
        const probs = softmax(logits);
        
        // Log prediction info
        console.log('Prediction:', {
            maxProb: Math.max(...probs),
            prediction: probs.indexOf(Math.max(...probs))
        });
        
        return { logits, probs };
    }
}

// ===== app.js =====
let net=null;
let draw, off, ctx, offCtx;

async function loadWeights(){
    try {
        // Default weights for testing - replace these with your actual weights
        const defaultWeights = {
            "W1": Array(128).fill().map(() => Array(784).fill().map(() => Math.random() * 0.1 - 0.05)),
            "b1": Array(128).fill().map(() => Math.random() * 0.1 - 0.05),
            "W2": Array(10).fill().map(() => Array(128).fill().map(() => Math.random() * 0.1 - 0.05)),
      "b2": Array(10).fill().map(() => Math.random() * 0.1 - 0.05),
      "_isDefault": true
        };

        console.log('Attempting to load weights...');
        try {
            const res = await fetch('./assets/weights/mnist_weights.json');
            if (!res.ok) {
                console.warn('Could not load weights file, using default initialization');
                return defaultWeights;
            }
      let weights = await res.json();

      // Validate loaded weights
      if (!weights || !weights.W1 || !weights.W2 || !weights.b1 || !weights.b2) {
        console.warn('Invalid weights file format, using default initialization');
        return defaultWeights;
      }

      // Normalize weights: some exporters wrap scalars in single-element arrays
      function normalizeWeights(w) {
        const toNumber = (v) => {
          if (Array.isArray(v) && v.length === 1) v = v[0];
          if (typeof v === 'number') return v;
          if (typeof v === 'string') {
            const n = parseFloat(v);
            return isNaN(n) ? 0 : n;
          }
          // fallback
          return Number(v) || 0;
        };

        // Normalize biases
        w.b1 = w.b1.map(toNumber);
        w.b2 = w.b2.map(toNumber);

        // Normalize weight matrices (rows x cols)
        w.W1 = w.W1.map(row => row.map(toNumber));
        w.W2 = w.W2.map(row => row.map(toNumber));

        return w;
      }

      try {
        weights = normalizeWeights(weights);
      } catch (e) {
        console.warn('Failed to normalize weights, using default initialization', e);
        return defaultWeights;
      }

            // Check if weights are all zero or very close to zero
            const isZeroWeights = 
                weights.W1.every(row => row.every(w => Math.abs(w) < 1e-6)) &&
                weights.W2.every(row => row.every(w => Math.abs(w) < 1e-6)) &&
                weights.b1.every(b => Math.abs(b) < 1e-6) &&
                weights.b2.every(b => Math.abs(b) < 1e-6);

            if (isZeroWeights) {
                console.warn('Weights appear to be all zero, using default initialization');
        defaultWeights._isDefault = true;
        return defaultWeights;
            }

            weights._isDefault = false;
            return weights;
        } catch (error) {
            console.warn('Error loading weights, using default initialization:', error);
            return defaultWeights;
        }
    } catch (error) {
        console.error('Critical error in loadWeights:', error);
        throw error;
    }
}
function setupCanvas(){
  draw = document.getElementById('draw');
  ctx = draw.getContext('2d');
  draw.width = draw.height = 280;
  ctx.fillStyle = '#000';
  ctx.fillRect(0,0,draw.width,draw.height);
  ctx.lineCap='round';
  ctx.lineJoin='round';
  // default line width; set at ~25% of slider range (min=4,max=40 -> ~13)
  ctx.lineWidth=13;
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
function clearCanvas(){ ctx.fillStyle='#000'; ctx.fillRect(0,0,draw.width,draw.height); }
function getInput784(){
  // Improved preprocessing:
  // 1) find bounding box of ink on main canvas (280x280)
  // 2) crop to bbox, scale so largest side = 20, draw centered into 28x28
  // 3) convert to grayscale 0..1, apply invert and normalize contrast

  // Read full-size canvas pixels
  const W = draw.width, H = draw.height;
  const full = ctx.getImageData(0,0,W,H).data;
  let minX = W, minY = H, maxX = 0, maxY = 0;
  const thresh = 10; // pixel brightness threshold (0-255)
  for(let y=0;y<H;y++){
    for(let x=0;x<W;x++){
      const i = (y*W + x) * 4;
      const r = full[i], g = full[i+1], b = full[i+2];
      const v = (r+g+b)/3;
      if (v > thresh){
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
      }
    }
  }

  // If no ink found, return blank
  if (maxX < minX || maxY < minY) return new Array(28*28).fill(0);

  // Add small padding
  const pad = 8; // in original canvas pixels
  minX = Math.max(0, minX - pad);
  minY = Math.max(0, minY - pad);
  maxX = Math.min(W-1, maxX + pad);
  maxY = Math.min(H-1, maxY + pad);

  const sw = maxX - minX + 1;
  const sh = maxY - minY + 1;

  // Compute scale to fit 20x20 box
  const target = 20;
  const scale = target / Math.max(sw, sh);
  const dw = Math.max(1, Math.round(sw * scale));
  const dh = Math.max(1, Math.round(sh * scale));
  const dx = Math.floor((28 - dw) / 2);
  const dy = Math.floor((28 - dh) / 2);

  // Clear offscreen and draw scaled crop centered
  offCtx.fillStyle = '#000';
  offCtx.fillRect(0,0,28,28);
  offCtx.drawImage(draw, minX, minY, sw, sh, dx, dy, dw, dh);

  const img = offCtx.getImageData(0, 0, 28, 28).data;
  const x = new Array(28*28);
  let minVal = 1, maxVal = 0;
  for(let i=0;i<28*28;i++){
    const r = img[4*i], g = img[4*i+1], b = img[4*i+2];
    let v = (0.299*r + 0.587*g + 0.114*b)/255; // 0..1 white on black
    v = Math.max(0, Math.min(1, v));
    x[i] = v;
    if (v < minVal) minVal = v;
    if (v > maxVal) maxVal = v;
  }

  // Contrast normalize: stretch min..max -> 0..1
  if (maxVal - minVal > 1e-3){
    for(let i=0;i<x.length;i++) x[i] = (x[i] - minVal) / (maxVal - minVal);
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
    p.innerHTML = `
      <div><strong>${d}</strong></div>
      <div class="bar"><span style="width:${(probs[d]*100).toFixed(1)}%"></span></div>
      <div><small>${toPct(probs[d])}</small></div>
    `;
    grid.appendChild(p);
  }
  document.getElementById('guess').textContent = String(best);
  document.getElementById('confidence').textContent = toPct(probs[best]);
}
async function predict(){
  if (!net) {
    console.error('Neural network not initialized');
    return;
  }
  const x = getInput784();
  console.log('Input sample (first 20):', x.slice(0,20));
  console.log('Input non-zero count:', x.filter(v=>v>0).length);
  const {logits, probs} = net.forward(x);
  console.log('Logits from network:', logits);
  console.log('Probs from network:', probs);
  renderProbs(probs);
}
function setupUI(){
  document.getElementById('btn-predict').addEventListener('click', predict);
  document.getElementById('btn-clear').addEventListener('click', clearCanvas);
  
  // Brush size slider
  const brush = document.getElementById('brush-size');
  if (brush) {
    const setBrush = (v) => { ctx.lineWidth = Number(v); };
    setBrush(brush.value);
    brush.addEventListener('input', (e) => {
      setBrush(e.target.value);
      console.log('Brush size set to', e.target.value);
    });
  }
  
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
  The app loads your exported <code>W1, b1, W2, b2</code> and runs pure-JS inference in the browser.</div>
  <h3>Preprocessing</h3>
  <p>We draw at 280×280 px for a smooth pen, then downscale to 28×28 and convert to grayscale in [0,1].</p>
  <p><strong>Note:</strong> Model accuracy can depend on the stroke thickness. Use the <em>Brush size</em> slider to adjust pen width — typically values around 14–30 work best for this model. If your strokes are too thin or too thick the network may misclassify.</p>
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
async function init(){
    try {
        console.log('Starting initialization...');
        
        // Setup basic UI
        setupCanvas();
        setupUI();
        setupTabs();
        fillHowItWorks();
        
        // Load weights
        const weights = await loadWeights();
        console.log('Weights loaded, creating neural network...');
        
        // Initialize network
        net = new MNISTNet(weights);
        
        // Clear canvas and reset UI
        clearCanvas();
        document.getElementById('guess').textContent = '?';
        document.getElementById('confidence').textContent = 'Ready';
        
        console.log('Initialization complete');
    } catch (error) {
        console.error('Initialization failed:', error);
        document.getElementById('guess').textContent = 'Error';
        document.getElementById('confidence').textContent = error.message;
        throw error;
    }
}
document.addEventListener('DOMContentLoaded', init);
