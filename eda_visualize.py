import matplotlib
matplotlib.use("Agg")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


def generate_visualizations():
    os.makedirs("static/plots", exist_ok=True)

    data = pd.read_csv("data.csv")
    data['BMI Category'] = data['BMI Category'].replace({'Normal Weight': 'Normal'})

    if 'Blood Pressure' in data.columns:
        data[['Systolic BP', 'Diastolic BP']] = data['Blood Pressure'].str.split('/', expand=True)
        data['Systolic BP'] = pd.to_numeric(data['Systolic BP'], errors='coerce')
        data['Diastolic BP'] = pd.to_numeric(data['Diastolic BP'], errors='coerce')
        data.drop(columns=['Blood Pressure'], inplace=True)

    data.drop(columns=['Person ID'], inplace=True, errors='ignore')

    sns.set(style="whitegrid")

    plt.figure(figsize=(8, 5))
    sns.heatmap(data.select_dtypes(include=['int64','float64']).corr(), annot=True)
    plt.savefig("static/plots/correlation_heatmap.png")
    plt.close()

    plt.figure(figsize=(6, 4))
    sns.countplot(x="Gender", hue="BMI Category", data=data)
    plt.savefig("static/plots/bmi_by_gender.png")
    plt.close()

    plt.figure(figsize=(6, 4))
    sns.scatterplot(x="Physical Activity Level", y="Stress Level", data=data)
    plt.savefig("static/plots/activity_vs_stress.png")
    plt.close()
