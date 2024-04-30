from sklearn.metrics import classification_report
def report(file_path,y_pred,y_true,classes):
    f = open(file_path, 'w')
    f.write(classification_report( y_pred ,y_true ,target_names = classes, digits = 5))
    f.close()
    