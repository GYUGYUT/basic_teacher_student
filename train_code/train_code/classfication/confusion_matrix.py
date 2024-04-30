from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import os
import matplotlib.pyplot as plt
import cv2 as cv
def confusion(con_y_label,con_y_pred,classes,args,save_path):
    filepath = os.path.join(args.save_path_conpusion, '{}_plot_conpusion.png'.format(save_path))
    cf_matrix = confusion_matrix(con_y_label, con_y_pred)
    df_cm = pd.DataFrame(cf_matrix, index = [i for i in classes],
                        columns = [i for i in classes])
    plt.figure(figsize = (12,7))
    sn.heatmap(df_cm, annot=True)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(filepath)