import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from src.exception import CustomException
from src.logger import logging
import os
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts',"preprocessor.pkl")

class DataTransformation:

    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_datatransformer(self):

        '''
        This function is to do the StandardScaling and Imputing Missing values 
        '''
        
        try:
            numerical_features = ['writing score', 'reading score']
            categorical_features = [
                'gender', 
                'race/ethnicity',
                'parental level of education', 
                'lunch', 
                'test preparation course']
            
            # Making a numerical pipeline to handle missing values and scaling values
            num_pipeline=Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]
            )

            # Categorical Pipeline to handle missing values and scaling values

            cat_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('one_hot_encoder', OneHotEncoder()),
                    ('scaler', StandardScaler(with_mean=False))
                ]
            )
            logging.info(f"Numerical Columns: {numerical_features}")
            logging.info(f"Categorical Columns: {categorical_features}")

            preprocessor = ColumnTransformer(
                [
                    ('num_pipeline', num_pipeline, numerical_features),
                    ('cat_pipeline', cat_pipeline, categorical_features)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)


    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Reading train and test data completed")


            logging.info("Obtaining preprocessing object")

            preprocessing_obj = self.get_datatransformer()

            target_column_name = 'math score'
            numerical_columns = ['writing score', 'reading score']

            #Getting input features and target features

            input_feature_train_df = train_df.drop(columns = [target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns = [target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info(
                f" Applying preprocessing on train and test dataframes"
            )

            input_feature_train = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train, np.array(target_feature_train_df)
            ]

            test_arr = np.c_[
                input_feature_test, np.array(target_feature_test_df)
            ]

            logging.info("Saving preprocessed data")

            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path, 
                obj = preprocessing_obj
            )
            
            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path

            )
        except Exception as e:
            raise CustomException(e, sys)
