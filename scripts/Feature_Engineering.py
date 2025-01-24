import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging

class Feature_Engineering:
    
    def __init__(self, file_path):
        self.data = self.load_data(file_path)
    
    def load_data(self, file_path):
        """Load the dataset."""
        data = pd.read_csv(file_path)
        return data