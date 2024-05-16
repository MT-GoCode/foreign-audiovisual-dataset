import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime
import ast

class Plotter:
    def __init__(self, **kwargs):
        self.__dict__ = dict(kwargs)

        self.feature_csv_pd = pd.read_csv(self.feature_csv)
    
    def create_pie_charts(self, df, output_folder):
        # Define attribute categories and possible values
        categories = {
            'Gender': ['Female', 'Male'],
            'Ethnicity': ['Asian', 'Indian', 'Black', 'White', 'Middle Eastern', 'Hispanic'],
            'Facial hair': ['Facial_Hair', 'Bald', 'Bangs', 'Bushy_Eyebrows'],
            'Glasses': ['Eyeglasses'],
            'Age group': ['Old', 'Young']
        }

        # Ensure output directory exists
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # Timestamp for file names
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        df['Attributes'] = df['Attributes'].astype(str)

        print(df['Attributes'])

        # Iterate over each category
        for category, values in categories.items():
            # Count occurrences for each value in the category
            counts = {value: 0 for value in values}
            counts['Unclassified'] = 0

            print(values)

            for attributes in df['Attributes']:
                found = False
                
                for value in values:
                    if value in attributes:
                        counts[value] += 1
                        found = True
                if not found:
                    counts['Unclassified'] += 1

            # Filter out classes with zero counts
            counts = {k: v for k, v in counts.items() if v > 0}

            # Generate pie chart
            fig, ax = plt.subplots()
            ax.pie(counts.values(), labels=counts.keys(), autopct='%1.1f%%', startangle=90)
            ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
            plt.title(f'{category} Distribution')

            # Save the pie chart as PNG
            file_name = f"{category.replace(' ', '_')}_distribution_{timestamp}.png"
            plt.savefig(os.path.join(output_folder, file_name))
            plt.close()


    def execute(self):
        self.create_pie_charts(self.feature_csv_pd, self.plot_folder)
