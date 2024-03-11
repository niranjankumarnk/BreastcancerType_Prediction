import pandas as pd
import os
from mlProject import logger
from sklearn.linear_model import LogisticRegression
from lazypredict.Supervised import LazyClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
from mlProject.entity.config_entity import ModelTrainerConfig


class ModelTrainer:
    def __init__(self, config = ModelTrainerConfig):
        self.config = config
        
    def train_model_1(self):
        
        train_data = pd.read_csv(self.config.train_data_path)
        test_data = pd.read_csv(self.config.test_data_path)
        train_cols = list(train_data.columns)
        test_cols= list(test_data.columns)
    
        
        target_column_schema =list(self.config.target_column.keys())[0]
        print(target_column_schema)
        label_encoder = LabelEncoder()
        train_data[target_column_schema] = label_encoder.fit_transform(train_data[target_column_schema])
        
        train_x = train_data.drop(columns=[target_column_schema], axis = 1)
        test_x = test_data.drop(columns=[target_column_schema], axis = 1)
        train_y = train_data[target_column_schema]
        test_y = test_data[target_column_schema]
        
        lr = LogisticRegression()
        lr.fit(train_x, train_y)
        
        joblib.dump(lr,os.path.join(self.config.root_dir, self.config.model_name))
        
    def train_model_2(self):
        train_data = pd.read_csv(self.config.train_data_path)
        test_data = pd.read_csv(self.config.test_data_path)
        train_cols = list(train_data.columns)
        test_cols= list(test_data.columns)
    
        
        target_column_schema =list(self.config.target_column.keys())[0]
        print(target_column_schema)
        label_encoder = LabelEncoder()
        train_data[target_column_schema] = label_encoder.fit_transform(train_data[target_column_schema])
        
        train_x = train_data.drop(columns=[target_column_schema], axis = 1)
        test_x = test_data.drop(columns=[target_column_schema], axis = 1)
        train_y = train_data[target_column_schema]
        test_y = test_data[target_column_schema]
        
        
        clf = LazyClassifier()
        clf.fit(train_x, test_x, train_y, test_y)
        
        joblib.dump(clf,os.path.join(self.config.root_dir, self.config.model_name))