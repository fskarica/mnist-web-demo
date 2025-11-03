
import {relu, softmax, matVecMul, addVec} from './utils.js';

export class MNISTNet{
  constructor(weights){
    this.W1 = weights.W1; // shape: H x 784
    this.b1 = weights.b1; // shape: H
    this.W2 = weights.W2; // shape: 10 x H
    this.b2 = weights.b2; // shape: 10
  }
  forward(x784){
    // x784: Float64Array or number[] of length 784 scaled 0..1
    const h1 = relu(addVec(matVecMul(this.W1, x784), this.b1));
    const logits = addVec(matVecMul(this.W2, h1), this.b2);
    const probs = softmax(logits);
    return {logits, probs};
  }
}
