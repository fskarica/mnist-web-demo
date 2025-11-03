# Handwritten Digit Classifier

[![Live Demo](https://img.shields.io/badge/demo-live-green.svg)](https://fskarica.github.io/mnist-web-demo/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Interactive web-based handwritten digit recognition using neural networks. Draw a digit and see real-time predictions directly in your browser.

## Demo

Try it now: [Live Demo](https://fskarica.github.io/mnist-web-demo/)

![Demo Screenshot](assets/demo.png)

## Features

- Draw digits using adjustable brush size
- Real-time predictions with confidence scores
- Runs completely in browser - no server needed
- Works on both desktop and mobile devices
- ~90% accuracy using MNIST-trained model

## Usage

1. Open the [live demo](https://fskarica.github.io/mnist-web-demo/)
2. Draw a digit (0-9) in the canvas
3. See the prediction and confidence score update in real-time
4. Use the brush size slider to adjust stroke thickness
5. Click "Clear" to try another digit

## Local Development

```bash
git clone https://github.com/fskarica/mnist-web-demo.git
cd mnist-web-demo
python -m http.server 8000
# Open http://localhost:8000 in your browser
```

## Technical Details

- Browser-based neural network (784 → 10 → 10)
- Pure JavaScript implementation
- Trained on MNIST dataset
- No external dependencies

## Author

Created by [Fran Škarica](https://github.com/fskarica)

## License

MIT
