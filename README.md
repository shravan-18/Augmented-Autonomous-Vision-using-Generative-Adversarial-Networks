# AMD-Pervasive-AI-Developer-Contest: Augmented Autonomous Vision using Generative Adversarial Networks

This repository contains a CycleGAN implementation for translating images between day and night domains. The CycleGAN model is designed to generate realistic images of the opposite lighting condition, making it useful for applications such as autonomous driving, urban planning, and data augmentation.

This repository contains a CycleGAN implementation for translating images between day and night domains. The CycleGAN model is designed to generate realistic images of the opposite lighting condition, making it useful for applications such as autonomous driving, urban planning, and data augmentation.

## What is our project about?

Our project involves developing a CycleGAN model to translate images between day and night scenes. The aim is to create a model capable of generating realistic nighttime images from daytime inputs and vice versa, enhancing the versatility of visual data for various applications.

## Why did we decide to make it?

The inspiration for this project stems from the growing need for robust image translation techniques in fields like autonomous driving, urban planning, and security. Day-to-night translation can be crucial for improving visibility in different lighting conditions, optimizing surveillance systems, and enhancing autonomous vehicle navigation in varying light scenarios.

## How does it work?

The CycleGAN model operates using two main components: a generator and a discriminator. The generator learns to translate images from one domain (day) to another (night) and vice versa, while the discriminator evaluates the authenticity of these generated images. By iteratively training both components, the model improves its ability to create realistic images in the target domain. The CycleGAN architecture ensures that the generated images maintain consistent features and details, even when translated across different lighting conditions.

## Model Workflow

Below is the overall model workflow diagram, illustrating the CycleGAN architecture and its components:

![Model Workflow](https://github.com/shravan-18/AMD-Pervasive-AI-Developer-Contest/workflow.png)

## Instructions for Day-Night CycleGAN

This script implements a CycleGAN model for translating images between day and night domains. Follow these instructions to set up and run the CycleGAN model:

### Prerequisites

1. **Python Libraries**: Ensure you have the following libraries installed:
   - PyTorch
   - torchvision
   - numpy
   - tqdm
   - (Optional) any other dependencies for loading and processing images.

2. **Data Preparation**:
   - Prepare two datasets: one containing images from the day domain and the other from the night domain.
   - The datasets should be organized in a format compatible with the DataLoader. Each dataset should be a directory containing images.

### Setup

1. **Directory Structure**:
   - Place your day images in a directory (e.g., `data/day`).
   - Place your night images in a separate directory (e.g., `data/night`).

2. **Parameters**:
   - `batch_size`: Set the batch size for training.
   - `target_shape`: Define the target shape for resizing images.
   - `n_epochs`: Specify the number of training epochs.
   - `lambda_identity`: Weight for the identity loss.
   - `lambda_cycle`: Weight for the cycle consistency loss.
   - `device`: Set the computing device (e.g., 'cuda' for GPU or 'cpu').

### Running the Code

1. **Loading the Data**:
   - The DataLoader will automatically load images from the specified directories. Ensure that the `dataset` variable is correctly set up to point to your image directories.

2. **Training**:
   - Call the `train()` function to start the training process. This function handles the training loop, updates the generators and discriminators, and saves model checkpoints periodically.
   - Optionally, set `save_model=True` in the `train()` function to save the trained models.

3. **Monitoring**:
   - The training process will print out the generator and discriminator losses at regular intervals defined by `display_step`.

4. **Model Checkpoints**:
   - Model checkpoints are saved during training to allow you to resume training or use the model for inference later.

### Example

To run the training, simply execute the script. Ensure you have configured the parameters and data directories as needed.

```python
# Example usage
train(save_model=True)
```

## Outcome and Problem Solved

The outcome of this project is a CycleGAN model capable of translating images between day and night scenes with high fidelity. This solution addresses the following challenges:

- **Adaptability in Varying Lighting Conditions**: The model adapts visual data to different lighting conditions, critical for applications like autonomous driving and surveillance.

- **Enhanced Data Usability**: By generating nighttime images from daytime sources, the model improves the performance of other models trained on diverse lighting conditions.

- **Improved Visualization**: Facilitates visualization and planning for environments with varying lighting without needing actual nighttime imagery.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- This implementation is inspired by the original CycleGAN paper by Zhu et al.
- Special thanks to the open-source community for their contributions and support.
