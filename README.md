# Gesture-Controlled Spotify Player

This project implements a gesture recognition system to control Spotify playback using hand gestures captured through a webcam.

## Features

-   Real-time hand gesture recognition
-   Spotify playback control (play/pause, next/previous track, toggle loop)
-   Multi-processing for smooth performance

## Requirements

-   Python 3.7+
-   OpenCV
-   NumPy
-   Pandas
-   Mediapipe
-   Seaborn
-   Joblib
-   Scikit-learn
-   Matplotlib
-   Spotipy

## Installation

### Clone this repository:

```bash
git clone https://github.com/NotThareesh/RIG-Inductions.git
```

### Install required packages:

```bash
pip install -r requirements.txt
```

### Set up Spotify API credentials:

Create a Spotify Developer account and create a new app
Set environment variables for CLIENT_ID and CLIENT_SECRET

## Usage

Run the main script:

```bash
python main.py
```

Ensure your Spotify account is connected to an active device before running the program.

## Development Process

### Data Collection

-   Custom dataset with 4000+ records of hand landmark coordinates
-   Gestures: OK, Single Finger Pointer, Left/Right Pointer, Rock & Roll, Open Hand

### Model Training

-   Used StandardScaler for data normalization
-   Implemented Kernel SVC with RBF kernel
-   Utilized GridSearch for hyperparameter optimization
-   K-Fold Cross Validation (5 folds) for model evaluation

### Performance Optimization

-   Multiprocessing implementation to prevent camera lag
-   3-second cooldown between API calls
-   96% accuracy threshold for gesture recognition

### Future Improvements

-   Implement seek functionality using pointer gestures
-   Expand gesture set for more controls
-   Improve model generalization across different users

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License. See the [License](LICENSE) file for details.
