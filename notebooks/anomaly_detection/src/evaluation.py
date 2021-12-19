import pandas as  pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import torch


def create_loss_label_df(model, dl, device):
    losses = []
    labels = []
    model.eval()
    with torch.no_grad():
        for x, y in dl:
            pred = model(x.to(device)).cpu().numpy()
            loss = np.mean(np.square(pred - x.numpy()), axis=1)
            losses.extend(loss)
            labels.extend(y.numpy().astype(np.int8))
    
    return pd.DataFrame({'loss': losses, 'label': labels})


def cal_threshold(df):
    normal_df = df[df['label'] == 0]
    return normal_df['loss'].mean() + normal_df['loss'].std()


def predict(losses, threshold):
    return list(map(lambda loss: 1 if loss > threshold else 0, losses))


def show_confusion_matrix(df, pred):
    conf_matrix = confusion_matrix(df['label'].values, pred)
    sns.set(font_scale=1.2)
    plt.figure(figsize=(10, 10))
    sns.heatmap(conf_matrix,
                xticklabels=['Not Attack', 'Attack'],
                yticklabels=['Not Attack', 'Attack'],
                annot=True,
                fmt="d")
    plt.title("Confusion matrix")
    plt.ylabel('True class')
    plt.xlabel('Predicted class')
    plt.show()