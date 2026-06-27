import matplotlib.pyplot as plt

# 不同阈值下的准确率
precision = [0.8, 0.7, 0.65, 0.6, 0.55]
# 不同阈值下的召回率
recall = [0.6, 0.55, 0.5, 0.45, 0.4]

thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]


mAP= [0.73, 0.78, 0.81, 0.831, 0.84, 0.85, 0.87, 0.88, 0.886, 0.882,0.895,0.898,0.909,0.90,0.891,0.902]
epochs = range(1, len(mAP) + 1)

plt.plot(epochs, mAP, 'bo-', label='mAP')
#plt.plot(thresholds, recall, 'r+-', label='Recall')
plt.title('mAP')
plt.xlabel('Epochs')
plt.ylabel('Value')
plt.legend()
plt.show()