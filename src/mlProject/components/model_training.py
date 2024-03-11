import pandas as pd
import os
from mlProject import logger
from sklearn.linear_model import LogisticRegression
from lazypredict.Supervised import LazyClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
from mlProject.entity.config_entity import ModelTrainingConfig


class ModelTraining:
    def __init__(self, config = ModelTrainingConfig):
        self.config = config
        
    def train_model(self):
        data = pd.read_csv(self.config.data_path)
        train_data = pd.read_csv(self.config.train_data_path, usecols=lambda column: column != 'id')
        test_data = pd.read_csv(self.config.test_data_path, usecols=lambda column: column != 'id')
    
        
        target_column_schema =list(self.config.target_column.keys())[0]
        label_encoder = LabelEncoder()
        data[target_column_schema] = label_encoder.fit_transform(data[target_column_schema])
        train_data[target_column_schema] = label_encoder.transform(train_data[target_column_schema])
        test_data[target_column_schema] = label_encoder.transform(test_data[target_column_schema])

        train_x = train_data.drop(columns=[target_column_schema], axis = 1)
        test_x = test_data.drop(columns=[target_column_schema], axis = 1)
        train_y = train_data[target_column_schema]
        test_y = test_data[target_column_schema]
        
        lr = LogisticRegression(max_iter= 1000)
        lr.fit(train_x, train_y)
        
        joblib.dump(lr,os.path.join(self.config.root_dir, self.config.model_name))

        
        
    