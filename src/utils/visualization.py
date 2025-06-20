import matplotlib.pyplot as plt

def show_distances(epochs, positive_distances, negative_distances, plot_dir):
    epochs_range = range(1, epochs + 1)
    plt.figure(figsize=(8, 6))
    plt.plot(epochs_range, positive_distances, label='Positive Distances', color='green', marker='x')
    plt.plot(epochs_range, negative_distances, label='Negative Distances', color='red', marker='s')
    
    plt.title('Positive and Negative Distances Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Distance')
    plt.legend(loc='upper right')
    plt.grid(True)
    
    plt.savefig(plot_dir + '/distance.png')
    plt.show()

def show_trainingDetial(epochs, train_losses_np, val_losses_np, train_accuracies_np, val_accuracies_np, plot_dir):
    epochs_range = range(1, epochs + 1)
    plt.figure(figsize=(18, 6))
    plt.subplot(1, 3, 1)
    plt.plot(epochs_range, train_losses_np, label='Training Loss')
    plt.plot(epochs_range, val_losses_np, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Testing Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    plt.plot(epochs_range, train_accuracies_np, label='Training Accuracy')
    plt.plot(epochs_range, val_accuracies_np, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Testing Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.savefig(plot_dir + '/loss_accuracy.png')
    plt.show()