import matplotlib.pyplot as plt

def plot_training_losses(cnn_losses, transformer_losses, fusion_losses, filename="training_losses.png"):
    plt.figure(figsize=(10, 6))
    plt.plot(cnn_losses, label='PatchCNN Loss', marker='o')
    plt.plot(transformer_losses, label='GridTransformer Loss', marker='s')
    plt.plot(fusion_losses, label='FusionModel Loss', marker='^')
    plt.xlabel('Training Iteration')
    plt.ylabel('Average Loss')
    plt.title('Training Loss per Task')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()