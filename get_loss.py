import matplotlib.pyplot as plt

# 训练损失
train_loss = [ 15.35, 10.11, 8.78, 6.78, 6.12, 5.58, 5.24, 5.07, 4.96, 4.92, 4.87, 4.89, 4.84, 4.83, 4.81, 4.80]
# 验证损失
val_loss =   [ 11.69, 9.47, 7.88, 6.22, 5.80, 5.63, 5.46, 5.06, 5.35, 5.21, 5.01, 4.99, 5.04, 4.95, 4.92, 4.90]

epochs = range(1, len(train_loss) + 1)

plt.plot(epochs, train_loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()