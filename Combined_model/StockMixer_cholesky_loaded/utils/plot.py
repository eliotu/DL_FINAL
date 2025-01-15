from matplotlib import pyplot as plt

def plot_history(
    train_loss: list,
    val_loss: list,
    labels: list[str],
    title: str,
    test_loss: list = None,
):
    
    assert len(train_loss) == len(val_loss), "Train and validation loss must have the same length"
    if test_loss:
        assert len(train_loss) == len(test_loss), "Train and test loss must have the same length"
    
    plt.figure(figsize=(12, 6))
    plt.plot(train_loss, label=f'{labels[0]}')
    plt.plot(val_loss, label=f'{labels[1]}')
    if test_loss:
        plt.plot(test_loss, f'{labels[2]}')
    plt.title(f'{title}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')  # Set y-axis to logarithmic scale
    plt.legend()
    plt.grid(True)
    plt.show()