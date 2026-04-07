# Copyright © 2024 Roblox Corporation


import torch
import librosa
import numpy as np
import argparse
from transformers import WavLMForSequenceClassification
import soundfile as sf
import warnings
import os
from pathlib import Path

warnings.filterwarnings("ignore") 

labels_name_list = [
    "Discrimination", "Harassment", "Sexual",
    "IllegalAndRegulated", "DatingAndRomantic", "Profanity",
]
model_path = "roblox/voice-safety-classifier-v2"
target_failure_threshold = 1e-2

epsilon = 5e-4
step_size = 0.001
max_iter = 500

original_audio_length_samples = 0

output_dir = Path("adversarial_runs")
adversarial_dir = output_dir / "adversarial_audio"


def feature_extract_simple(
    wav,
    sr=16_000,
    win_len=15.0,
    win_stride=15.0,
    
    do_normalize=False,
):
    global original_audio_length_samples 
    
    if type(wav) == str:
        signal, _ = librosa.load(wav, sr=sr)
    else:
        try:
            signal = np.array(wav).squeeze()
        except Exception as e:
            print(e)
            raise RuntimeError

    
    signal = signal.astype(np.float32)
    
    if signal.size == 0:
        original_audio_length_samples = 0
        return np.array([])
        
    original_audio_length_samples = len(signal)
    
    batched_input = []
    stride = int(win_stride * sr)
    l = int(win_len * sr)
    
    if len(signal) / sr > win_len:
        for i in range(0, len(signal), stride):
            if i + l > len(signal):
                chunked = np.pad(signal[i:], (0, l - len(signal[i:])))
            else:
                chunked = signal[i : i + l]
            
            if do_normalize:
                std_dev = np.std(chunked)
                if std_dev > 1e-7:
                    chunked = (chunked - np.mean(chunked)) / std_dev
                
            batched_input.append(chunked)
            
            if i + l >= len(signal):
                break
    else:

        if len(signal) < l:
            signal = np.pad(signal, (0, l - len(signal)))
            
        if do_normalize:
            std_dev = np.std(signal)
            if std_dev > 1e-7:
                 signal = (signal - np.mean(signal)) / std_dev
            
        batched_input.append(signal)

    if len(batched_input) == 0:
         return np.array([])

    return np.stack(batched_input)  # [N, T]


def infer(model, inputs):
    #Run inference (forward pass) and apply Sigmoid to get probabilities
    if not isinstance(inputs, torch.Tensor):
        inputs = torch.Tensor(inputs)
        
    with torch.no_grad():
        output = model(inputs)
        probs = torch.sigmoid(output.logits)
    return probs.cpu().numpy()


def get_next_run_id(sub_dir):

    os.makedirs(sub_dir, exist_ok=True)
    existing_files = [f for f in os.listdir(sub_dir) if f.endswith('.wav')]
    if not existing_files:
        return 1
    
    max_id = 0
    for filename in existing_files:
        try:
            file_id = int(filename.split('.')[0])
            if file_id > max_id:
                max_id = file_id
        except ValueError:
            continue
    
    return max_id + 1

def save_audio_file(audio_data, sr, sub_dir, run_id):
    
    os.makedirs(sub_dir, exist_ok=True)
    file_name = f"{run_id:03d}.wav"
    file_path = sub_dir / file_name

    sf.write(str(file_path), audio_data, sr)
    
    return str(file_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--audio_file",
        type=str,
        required=True,
        help="Path to the audio file (.wav, .mp3, etc.) to run inference",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=model_path,
        help="Checkpoint file of model (Hugging Face path or local dir)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Probability threshold to determine if a class is 'positive' for multi-target attack.",
    )
    args = parser.parse_args()
    
    model = WavLMForSequenceClassification.from_pretrained(
        args.model_path, num_labels=len(labels_name_list)
    )
    model.eval() 

    run_id = get_next_run_id(adversarial_dir)
    
    print(f"Starting experiment run {run_id:03d}...")

    audio_path = args.audio_file
    audio, _ = librosa.load(audio_path, sr=16000) 
    input_np = feature_extract_simple(audio, sr=16000)
    
    original_length = original_audio_length_samples
    
    if original_length == 0:
        print("Error: Input audio file is empty or could not be processed.")
        exit()

    x = torch.Tensor(input_np).clone().detach() 

    with torch.no_grad():
        base_output = model(x)
        base_probs = torch.sigmoid(base_output.logits)
        
    P_indices = torch.where(base_probs.mean(dim=0) > args.threshold)[0].tolist()

    if not P_indices:
        print(f"⚠️ Warning: No classes detected above the threshold ({args.threshold}). Targeting the highest scored class instead.")
        target_idx = torch.argmax(base_probs.mean(dim=0)).item()
        P_indices = [target_idx]

    target_labels = [labels_name_list[i] for i in P_indices]
    print(f"Targeting Multi-Classes: {target_labels} (Goal: Minimize Scores < {target_failure_threshold:.2e})")
    
    # Initialize perturbation
    eta = torch.rand_like(x) * 2 * epsilon - epsilon  
    eta.requires_grad = True

    # PGD attack LOOP
    print(f"Starting PGD attack (Max {max_iter} iterations)...")
    
    for i in range(max_iter):
        x_adv = x + eta
        output = model(x_adv)
        
        # Multi-Target Loss
        target_logits = output.logits[:, P_indices]
        loss = target_logits.mean() 
        
        model.zero_grad()
        loss.backward()
        
        gradient_sign = eta.grad.data.sign()
        eta.data = torch.clamp(eta.data - step_size * gradient_sign, -epsilon, epsilon)
        
        #early checking
        with torch.no_grad():
            current_probs = torch.sigmoid(output.logits)
            max_targeted_score = current_probs[:, P_indices].max().item()
        
        print(f"Iter {i+1}/{max_iter}: Max Targeted Score={max_targeted_score:.4f}, Loss={loss.item():.4f}")
        
        if max_targeted_score < target_failure_threshold:
            print(f"\n✅ Attack successful! Stopping early at iteration {i+1}.")
            break

   
    final_x_adv = (x + eta).detach().numpy()
    cropped_adv_audio = final_x_adv[0][:original_length].astype(np.float32)
    
    adv_file_path = save_audio_file(cropped_adv_audio, 16000, adversarial_dir, run_id)
    print(f"Saved ADVERSARIAL AUDIO to: {adv_file_path}")

    final_probs_np = infer(model, torch.Tensor(final_x_adv))
    
    print("\n--- Summary of Classification on ADVERSARIAL Audio ---")
    
    final_probs_segment = final_probs_np.reshape(-1, len(labels_name_list))[0]
    
    print(f"Segment 0 (Original Length: {original_length} samples):")
    for label_idx, label in enumerate(labels_name_list):
        is_targeted = "(*TARGETED*)" if label_idx in P_indices else ""
        print(f"{label} : {final_probs_segment[label_idx]*100:.2f}% {is_targeted}")

