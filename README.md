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

1. Create a new repository on GitHub:
   - Go to github.com and click "New repository"
   - Name it `mnist-web-demo`
   - Keep it public
   - Don't initialize with any files

2. Initialize your local repository and push to GitHub:
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git branch -M main
   git remote add origin https://github.com/fskarica/mnist-web-demo.git
   git push -u origin main
   ```

3. Enable GitHub Pages:
   - Go to your repository's **Settings**
   - Navigate to **Pages** in the left sidebar
   - Under "Source", select **Deploy from a branch**
   - Choose **main** branch and **/** (root) folder
   - Click **Save**

4. Your site will be published at:
   `https://fskarica.github.io/mnist-web-demo/`
   (It might take a few minutes to become available)

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
