# Upscaled Video Detection at 4K Resolution

This repository contains models and scripts for detecting upscaled videos at 4K resolution. It includes two checkpoint models based on ResNet18 and ConvNext architectures.

## Features

- Detect upscaled videos at 4K resolution
- Two checkpoint models:
  - ResNet18-based model
  - ConvNext-based model
- Evaluation script with configurable parameters
- Detection of multiple super-resolution models and techniques

## Training Data

The training dataset is composed of upscaled videos from [BVI-DVC](https://fan-aaron-zhang.github.io/BVI-DVC/). THe 200 videos have been upscaled with the following methods to train the SR recognition model (only deep learning-based methods are included for download):
- "RVRT"
- "BasicVSR"
- "Bicubic interpolation"
- "Linear interpolation"
- "Nearest N."
- "real-esrgan"
- "Real BasicVSR"
- "SwinIR Classical"
- "SwinIR Real"

To get access, fill the form at https://docs.hpai.cloud/apps/forms/s/tAjr9RLNKKzjSyssy6yMEmm7. You will get a link with temporary access to the compressed files.

## Requirements

All required dependencies are listed in the `requirements.txt` file. You can install them using pip:

```bash
pip install -r requirements.txt
```

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/upscaled-video-detection.git
   cd upscaled-video-detection
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running Evaluations

To run evaluations, use the `eval.py` script. You need to specify the `result_save_path` and `test_dataset_path` as arguments:

```bash
python eval.py result_save_path=/path/to/save/results test_dataset_path=/path/to/test/dataset
```

### Changing Configuration

The evaluation script uses Hydra for configuration management. The default configuration is specified in the `eval.py` script.

## Model Checkpoints

The checkpoint models are available via Google Drive. You can download them using the following link:

[Download Model Checkpoints](https://drive.google.com/drive/folders/17jMBLb8nrVBjM2ujtXIS-W1fzoTkVGbx?usp=sharing)

After downloading, place the checkpoint files in the `checkpoints/` directory:

1. ResNet18-based model: `checkpoints/resnet-18_staircase.ckpt`
2. ConvNext-based model: `checkpoints/convnext.ckpt`

## Input Format

The expected input is a folder containing several subfolders, each corresponding to a different video. Each video subfolder should contain the video frames as individual PNG files.

Example structure:
```
test_dataset/
├── video1/
│   ├── frame_0001.png
│   ├── frame_0002.png
│   └── ...
├── video2/
│   ├── frame_0001.png
│   ├── frame_0002.png
│   └── ...
└── ...
```

Ensure that your input data follows this structure for the evaluation script to work correctly.

## Model Outputs

The models have been trained to detect several super-resolution models and techniques. The output labels correspond to the following list of upscaling methods:

1. "4k" (native 4K)
2. "RVRT"
3. "BasicVSR"
4. "Bicubic interpolation"
5. "Linear interpolation"
6. "Nearest N."
7. "real-esrgan"
8. "Real BasicVSR"
9. "SwinIR Classical"
10. "SwinIR Real"

When the model classifies a video, it will output a number from 1 to 10, corresponding to the index of the detected upscaling method in this list.

## License

For detailed license information, please refer to the `LICENSES.md` file in this repository.
