# Audio-Shield: Multi-Target PGD Attacks on Voice Safety Classifiers

**An implementation of Projected Gradient Descent (PGD) to audit and bypass the Roblox Voice Safety Model.**

---

## 🛡️ Project Overview

This project demonstrates a **white-box adversarial attack** on the `roblox/voice-safety-classifier-v2` acoustic model. By leveraging **Projected Gradient Descent (PGD)**, the system generates stealthy audio perturbations that "blind" the model to specific safety violations such as **Profanity, Harassment, or Discrimination**.

The core objective is to minimize the model's confidence scores for targeted labels below a specified failure threshold ($1 \times 10^{-2}$) while ensuring the attack remains stealthy to the human ear.

---

## 🚀 Key Features

* **Multi-Target Loss Function**: Simultaneously targets multiple classification labels by calculating the mean of logits for all identified safety violations.
* **Stealthy Perturbations**: Utilizes a constrained $\epsilon$ (epsilon) of $5 \times 10^{-4}$ to ensure the added noise is virtually imperceptible.
* **Automated Audio Pipeline**: Features automated resampling to 16kHz, chunking, and padding to meet WavLM input requirements.
* **Efficiency-First Attack**: Includes early-stopping logic that terminates the PGD loop as soon as the model's confidence drops below the required threshold.

---

## 🛠️ Technical Stack

* **Deep Learning**: PyTorch
* **Model Architecture**: WavLM (Transformers)
* **Audio Engineering**: Librosa & SoundFile
* **Analysis**: NumPy

---

## 📊 Attack Pipeline & Methodology

The attack follows an iterative PGD process:

1. **Initialization**: A perturbation tensor ($\eta$) is initialized with small random values to match the input shape.
2. **Forward Pass**: The adversarial input is calculated as $x_{adv} = x + \eta$.
3. **Gradient Calculation**: Backpropagation is used to calculate the gradient of the mean logits of the targeted classes.
4. **Update & Project**: $\eta$ is updated iteratively and clamped within the $\epsilon$-ball to maintain stealth.
5. **Finalization**: The final perturbation is added to the original signal and cropped to the proper length.

---

## ⚙️ Optimized Hyperparameters

| Parameter                | Value              | Description                                                              |
| :----------------------- | :----------------- | :----------------------------------------------------------------------- |
| **Epsilon ($\epsilon$)** | $5 \times 10^{-4}$ | Ensures a better result in perturbation and makes the attack stealthy    |
| **Step Size**            | $0.001$            | The magnitude of the gradient update                                     |
| **Max Iterations**       | $500$              | Standard iteration count; can be increased to 1000+ for higher precision |
| **Failure Threshold**    | $1 \times 10^{-2}$ | Stops iterations once the required result is achieved                    |

---

## 💻 Installation & Usage

### 1. Requirements

Ensure you have Python installed, then install the dependencies:

```bash
pip install torch librosa transformers soundfile numpy
```

---

### 2. Prepare the Model

The script will automatically download the `roblox/voice-safety-classifier-v2` model from Hugging Face upon the first run.

---

### 3. Run the Attack

To generate an adversarial version of an audio file, use the following command:

```bash
python optimal4.py --audio_file "your_input.wav" --threshold 0.5
```

**Arguments:**

* `--audio_file`: Path to the source audio you wish to perturb
* `--threshold`: The confidence level (0.0 - 1.0) above which a class is targeted for suppression

---

### 4. Verify Results

Run the inference script to see how the model classifies the new adversarial file:

```bash
python inference.py --audio_file "adversarial_runs/adversarial_audio/001.wav"
```

---

## 📁 Repository Structure

```
optimal4.py              # Core PGD attack script utilizing Torch and WavLM
inference.py             # Utility for testing model classification on audio files
adversarial_runs/        # Directory where generated adversarial audio is stored
.gitignore               # Prevents virtual environments and local test audio from being committed
```

---

## 📝 Results

The attack yields significant results where the input audio, despite containing safety violations, is misclassified or ignored by the model due to the added adversarial noise.

---

## 👤 Author

**Shreyas R Shetty**
*MSc in Cyber Security*
