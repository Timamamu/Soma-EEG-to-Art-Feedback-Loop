# Soma: EEG-to-Art Feedback Loop

An interactive art installation that captures live EEG data from a Muse headset, converts it to spectrograms, and uses trained pix2pix models to generate abstract visual art in real-time.

## Concept

Soma investigates how AI systems trained on collective human responses can subtly influence and manipulate individual users through seemingly neutral feedback. Participants wear an EEG headset and observe visuals generated from their own brainwaves but those visuals are shaped by a model trained on other people's neural responses. This creates a feedback loop where users unknowingly respond to representations of others' minds.

**Central Question:** Are you seeing yourself, or are you seeing others?

As AI becomes integrated into work, therapy, entertainment, and relationships, Soma examines the power dynamics embedded in these ostensibly neutral systems. The installation encourages participants to consider:

- How do feedback loops between humans and machines affect our sense of agency?
- Can a "neutral" visualization system actually manipulate emotional states?
- What happens when our responses are shaped by collective, anonymized data?

## Demo

### User Experience

![User Experience](assets/VID5.mp4)
![User Experience](assets/VID3.mp4)

*Participant wearing the Muse headset observing real-time visual generation from their brainwaves*

### Live Generation

![Live EEG to Art Generation](assets/VID2.mp4)
![Live EEG to Art Generation](assets/VID4.mov)

*Side-by-side view: Raw EEG spectrogram (left) and AI-generated artwork (right)*

### Training Process

![Training Process](assets/training)

*Model training workflow showing the pix2pix learning process*

## Prerequisites

### Hardware
- **Muse EEG headset** (any model with TP9 electrode)
- **CUDA-capable GPU** (recommended for real-time inference; CPU inference possible but slower)
- **Computer** running Windows, macOS, or Linux

### Software
- **Python**: 3.8 or higher
- **Git**: For cloning repositories
- **pip**: Python package manager

## Installation

### 1. Clone the pytorch-CycleGAN-and-pix2pix Framework

This project builds on the pytorch-CycleGAN-and-pix2pix framework:

```bash
git clone https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix.git
cd pytorch-CycleGAN-and-pix2pix
```

### 2. Install Python Dependencies

```bash
pip install torch torchvision numpy opencv-python scipy pylsl pandas matplotlib
```

**Note:** If you have a CUDA-capable GPU, ensure you install the CUDA-enabled version of PyTorch for better performance. See [PyTorch installation guide](https://pytorch.org/get-started/locally/).

### 3. Install muse-lsl

muse-lsl is required to stream data from the Muse headset via Lab Streaming Layer (LSL).

```bash
pip install muselsl
```

For detailed installation instructions and troubleshooting, see the [muse-lsl documentation](https://github.com/alexandrebarachant/muse-lsl).

### 4. Download Custom Scripts

Download the Soma scripts from this repository and place them in the pytorch-CycleGAN-and-pix2pix folder:

```bash
# From inside the pytorch-CycleGAN-and-pix2pix directory
git clone https://github.com/Timamamu/Soma-EEG-to-Art-Feedback-Loop.git soma_scripts
cp soma_scripts/*.py .
```

### 5. Download Pre-trained Models

Download the pre-trained pix2pix models from Google Drive:

**Models & Dataset:** [Download from Google Drive](https://drive.google.com/drive/folders/1oNXM0qSZwXAZmDkxKG9M9QMHzRAurd7T?usp=sharing)

The folder contains:
- `LANDSCAPE_MODEL/` - Model trained on landscape images
- `SHAPE_MODEL/` - Model trained on colored shapes
- `datasets/` - Training data (optional, for reference)

**Installation:**
1. Download the models folder from Google Drive
2. Extract it into your pytorch-CycleGAN-and-pix2pix directory
3. Your structure should look like:
   ```
   pytorch-CycleGAN-and-pix2pix/
   ├── models/
   │   ├── LANDSCAPE_MODEL/
   │   └── SHAPE_MODEL/
   ├── tp9_test.py
   ├── tp9_live2.py
   └── ... (other framework files)
   ```

## Usage

### Step 1: Connect Your Muse Headset

1. **Turn on your Muse headset** and ensure it's charged
2. **Pair it with your computer** via Bluetooth
3. **Start the muse-lsl stream:**

   ```bash
   muselsl stream
   ```

   You should see output indicating the stream is active. Keep this terminal window open.

### Step 2: Run the Scripts

Open a new terminal window in the pytorch-CycleGAN-and-pix2pix directory.

#### View Raw Spectrograms (No AI Processing)

To verify your EEG stream is working:

```bash
python tp9_test.py
```

This displays real-time spectrograms from the TP9 electrode without any AI transformation.

#### Run Live Art Generation

For the full installation experience with AI-generated visuals:

```bash
python tp9_live2.py
```

**Controls:**
- Press **'q'** in the display window to close the visualization
- Press **'s'** then **Enter** in the terminal to stop data collection

**What you'll see:** A side-by-side display showing:
- Left: Raw EEG spectrogram
- Right: AI-generated artwork based on the spectrogram

### Configuration

Edit `tp9_live2.py` to customize the experience:

```python
MODEL_FOLDER = os.path.join(THIS_FOLDER, f'models/LANDSCAPE_MODEL')  # Change to SHAPE_MODEL
MIN_DB, MAX_DB = -100, -20  # Adjust spectrogram normalization (reduces flicker)
duration = 7  # Seconds of EEG data per window
sampling_rate = 256  # Muse default sampling rate
```

**Available Models:**
- `LANDSCAPE_MODEL` - Generates abstract landscapes with horizon lines
- `SHAPE_MODEL` - Generates colorful geometric shapes

## How It Works

### Technical Pipeline

1. **EEG Capture**: Muse headset records electrical brain activity from the TP9 electrode
2. **Streaming**: muse-lsl sends data via Lab Streaming Layer protocol
3. **Windowing**: Script collects 7-second sliding windows of continuous EEG data
4. **Spectrogram Generation**: 
   - Applies Short-Time Fourier Transform (STFT)
   - 256 sample segments with 192 sample overlap
   - Converts to decibel scale and normalizes
5. **Visualization**: Applies VIRIDIS colormap
6. **AI Transformation**: pix2pix model generates artistic output
7. **Display**: Shows side-by-side comparison at 1920x1080 resolution

### Training Data Collection Process

The models were trained on data collected from participants who:

1. Viewed stimuli in four categories:
   - Colors (100 images)
   - Colored Shapes (100 images)
   - Words (50 words)
   - Landscapes with horizon lines (50 images)

2. Wore a Muse headset recording 7 seconds of EEG response per stimulus

3. Closed their eyes and took deep breaths between stimuli to reset baseline

The pix2pix model learned to map EEG spectrograms to the visual stimuli, creating an implicit representation of collective neural responses.

## Troubleshooting

### No EEG Stream Found

**Problem:** Script shows "No EEG stream found"

**Solutions:**
- Verify Muse headset is paired via Bluetooth
- Ensure `muselsl stream` is running in a separate terminal
- Check that the headset is properly positioned and making good contact
- Try restarting the muse-lsl stream

### TP9 Channel Not Found

**Problem:** "TP9 channel not found in EEG stream"

**Solutions:**
- Confirm you're using a Muse headset (not another EEG device)
- Check muse-lsl output to see which channels are available
- Verify the headset firmware is up to date

### Slow or Choppy Performance

**Problem:** Low frame rate or laggy visuals

**Solutions:**
- **Enable CUDA:** Ensure PyTorch is installed with CUDA support
- **Check GPU:** Verify your GPU is recognized: `python -c "import torch; print(torch.cuda.is_available())"`
- **Lower resolution:** Edit `tp9_live2.py` and reduce the output resolution
- **Close other applications:** Free up GPU memory

### Black Screen or No Visuals

**Problem:** Window opens but shows black screen

**Solutions:**
- Verify models downloaded correctly and path is set properly
- Check that EEG data is being received (run `tp9_test.py` first)
- Look for error messages in the terminal

### Flickering Visuals

**Problem:** Output flickers or changes dramatically between frames

**Solutions:**
- Use `tp9_live2.py` (not `tp9_live.py`) - it has flicker-free normalization
- Adjust `MIN_DB` and `MAX_DB` values in the script to constrain the dynamic range

## Training Your Own Models

If you want to train custom models on your own imagery, follow this workflow:

### Step 1: Record EEG Data

Use muse-lsl to record raw EEG data while participants view stimuli:

```bash
muselsl record --duration 7 --filename participant01_image01.csv
```

This creates CSV files containing 7 seconds of EEG data from all electrodes (TP9, AF7, AF8, TP10).

**Data Collection Protocol:**
1. Participant views a stimulus image (landscape, shape, color, etc.)
2. Record 7 seconds of EEG response
3. Participant closes eyes and breathes deeply to reset baseline
4. Repeat for each stimulus

Organize your CSVs in a folder structure like:
```
data/
├── landscapes_csv/
│   ├── image01.csv
│   ├── image02.csv
│   └── ...
└── shapes_csv/
    ├── shape01.csv
    └── ...
```

### Step 2: Convert CSVs to Spectrograms

Use `muse_spectogram_generator.py` to batch-convert all CSVs to spectrogram images:

1. **Upload the script to Google Colab** (it's designed for Colab environment)
2. **Mount your Google Drive:**
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
3. **Update the paths** in the script:
   ```python
   csv_folder = Path("/content/drive/MyDrive/YourProject/landscapes_csv")
   output_dirs = {
       'TP9': Path("/content/drive/MyDrive/YourProject/spectrograms_tp9"),
       # ... other electrodes
   }
   ```
4. **Run the script** - it will process all CSVs and generate 256x256 spectrogram images for each electrode

The script applies:
- STFT with 256-sample segments and 192-sample overlap
- Log-scale conversion to decibels
- Normalization and resize to 256x256
- VIRIDIS colormap

### Step 3: Prepare Paired Dataset

Organize your data for pix2pix training:

```
datasets/your_dataset/
├── train/
│   ├── A/  # EEG spectrograms
│   │   ├── image01.jpeg
│   │   ├── image02.jpeg
│   │   └── ...
│   └── B/  # Target images (landscapes, shapes, etc.)
│       ├── image01.jpeg
│       ├── image02.jpeg
│       └── ...
└── test/
    ├── A/
    └── B/
```

**Important:** Filenames in folders A and B must match exactly so pix2pix knows which pairs correspond.

### Step 4: Train the pix2pix Model

From the pytorch-CycleGAN-and-pix2pix directory:

```bash
python train.py --dataroot ./datasets/your_dataset \
                --name your_model_name \
                --model pix2pix \
                --direction AtoB \
                --n_epochs 200 \
                --n_epochs_decay 200
```

Training will take several hours depending on your GPU. Monitor progress in the `checkpoints/` folder.

### Step 5: Test Your Model

Update `MODEL_FOLDER` in `tp9_live2.py`:

```python
MODEL_FOLDER = os.path.join(THIS_FOLDER, f'models/YOUR_MODEL_NAME')
```

The provided dataset in the Google Drive link can serve as a reference for data structure and formatting.

## Project Structure

```
pytorch-CycleGAN-and-pix2pix/
├── models/
│   ├── LANDSCAPE_MODEL/      # Pre-trained landscape model
│   └── SHAPE_MODEL/          # Pre-trained shape model
├── tp9_test.py               # View raw spectrograms (no AI)
├── tp9_live2.py              # Live art generation (recommended)
├── tp9_live.py               # Alternative version (has flicker)
├── muse_spectogram_generator.py  # Generate training data
├── vis_utils.py              # Visualization utilities
└── ... (pytorch-CycleGAN-and-pix2pix framework files)
```

## Credits

### Team
**Akhil Dayal, Darren Chin, Fatima Mamu, James Bedford**  
Master of Design Engineering '25, Harvard GSD / MIT  
Final project for course on AI and Human Experience

**Development:** All custom code written by Fatima Mamu.

### Technical Dependencies
- [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) by Jun-Yan Zhu et al.
- [muse-lsl](https://github.com/alexandrebarachant/muse-lsl) by Alexandre Barachant

## License

This project is open source and available under the [MIT License](LICENSE).

This project builds upon MIT-licensed code. Please review dependency licenses before use.

---

*An experimental art project exploring feedback loops, agency, and the hidden biases in seemingly neutral AI systems.*
