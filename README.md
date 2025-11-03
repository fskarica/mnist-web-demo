
# Handwritten Digit Classifier (MNIST)

A lightweight browser-based implementation of a handwritten digit classifier using a neural network trained on the MNIST dataset. The model runs entirely in your browser with no external dependencies.

## Features

- Interactive drawing canvas with adjustable brush size
- 2-layer neural network (784 → 10 → 10) with ~90% accuracy
- Fast, client-side inference with pure JavaScript
- Real-time probability visualization for all digits
- No external ML libraries or dependencies
- Responsive design that works on both desktop and mobile

## Project Structure

```
mnist-web-demo/
├─ index.html          # Main HTML file
├─ bundle.js           # Combined JavaScript code
├─ styles.css         # Styling
└─ assets/
   └─ weights/
      └─ mnist_weights.json  # Trained model weights
```

## Technical Details

- **Neural Network Architecture**: 2-layer network with ReLU activation and softmax output
- **Input Processing**: 28x28 grayscale images (scaled down from 280x280 drawing)
- **Training**: Model trained on the MNIST dataset achieving ~90% accuracy
- **Implementation**: Pure JavaScript with optimized matrix operations
- **Performance**: Real-time inference with no delay between drawing and prediction

## Local Development

Any static server will work. For example, using Python:

```bash
# Python 3
cd mnist-web-demo
python -m http.server 8000
# Open http://localhost:8000 in your browser
```

## Deploy to GitHub Pages

1. Fork or clone this repository
2. Go to **Settings → Pages**
3. Set source to **Deploy from a branch**
4. Select **main** branch and **/** (root) folder
5. Your site will be available at `https://<username>.github.io/<repo>/`

## Usage Tips

- Use the brush size slider to adjust stroke thickness for optimal recognition
- Draw digits clearly in the center of the canvas
- The confidence score shows how sure the model is about its prediction
- You can clear and redraw if the prediction isn't accurate

## Author

Created by Fran Škarica

## License

MIT License

## License

MIT
