from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import os
import pandas as pd
from urllib.parse import urlparse
import numpy as np
import joblib
from mlProject.utils.common import save_json
from pathlib import Path
from mlProject.config.configuration import ModelEvaluationConfig



class ModelEvaluation:
    def __init__(self, config = ModelEvaluationConfig):
        self.config = config
    
    def eval_metrics(self,actual, pred):
        accuracy = accuracy_score(actual,pred)
        clas_report = classification_report(actual,pred)
        conf_matrix= confusion_matrix(actual,pred)
        return accuracy, clas_report, conf_matrix
    
    def save_results(self):
        data = pd.read_csv(self.config.data_path)
        test_data = pd.read_csv(self.config.test_data_path, usecols=lambda column: column != 'id')
        model = joblib.load(self.config.model_path)
        
        target_column_schema =list(self.config.target_column.keys())[0]
    
        label_encoder = LabelEncoder()
        data[target_column_schema] = label_encoder.fit_transform(data[target_column_schema])
        test_data[target_column_schema] = label_encoder.transform(test_data[target_column_schema])

        data[target_column_schema]
    
        test_x = test_data.drop(columns=[target_column_schema],axis =1)

        test_y = test_data[target_column_schema]
    
        predicted_qualities = model.predict(test_x)
        
        (accuracy, clas_report, conf_matrix) = self.eval_metrics(test_y, predicted_qualities)
        conf_matrix = conf_matrix.tolist()
        
    # Saving ,metrics as local
        scores = {'Accuracy': accuracy, 'clas_report': clas_report, 'conf_matrix': conf_matrix}
        save_json(path= Path(self.config.metric_file_name),data = scores)