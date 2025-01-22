import numpy as np
from pydub import AudioSegment, effects
import noisereduce as nr
import matplotlib.pyplot as plt

# Load audio
audio = AudioSegment.from_mp3("bat-customer.mp3")

# Convert to mono if stereo
audio = audio.set_channels(1)

# Convert audio to numpy array
samples = np.array(audio.get_array_of_samples())

# Adjust noise reduction parameters for better speech preservation
reduced_noise = nr.reduce_noise(
    y=samples,
    sr=audio.frame_rate,
    prop_decrease=0.75,  # Less aggressive noise reduction
    n_fft=1024,
    win_length=512,
    hop_length=128,
    n_std_thresh_stationary=1.5,    # Lower threshold for noise detection
    stationary=True
)

# Convert reduced noise signal back to audio
reduced_audio = AudioSegment(
    reduced_noise.tobytes(), 
    frame_rate=audio.frame_rate,
    sample_width=audio.sample_width,
    channels=1
)

# Enhance speech frequencies (300Hz - 3kHz)
reduced_audio = reduced_audio.high_pass_filter(300)
reduced_audio = reduced_audio.low_pass_filter(3000)

# Normalize audio
reduced_audio = effects.normalize(reduced_audio)

# Increase volume slightly
reduced_audio = reduced_audio + 3

# Export enhanced audio
reduced_audio.export("enhanced_output.mp3", format="mp3", bitrate="192k")

# Plot for visualization
fig, ax = plt.subplots(2, 1, figsize=(15,8))
ax[0].set_title("Original signal")
ax[0].plot(samples)
ax[1].set_title("Enhanced signal")
ax[1].plot(reduced_noise)
plt.show()