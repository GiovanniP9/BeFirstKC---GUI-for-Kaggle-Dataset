import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class Visualization:
    def __init__(self, df):
        self.df = df
        
    def split_numerical_categorical(self):
        df_num = self.df.select_dtypes(include=['int64', 'float64'])
        df_cat = self.df.select_dtypes(include=['object', 'category'])
        return df_num, df_cat
    
    def histplot(self, df):
        all_cols = df.columns
        n_cols = 3
        n_rows = (len(all_cols) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        axes = axes.flatten()
        
        for i, col in enumerate(all_cols):
            if df[col].dtype in ['int64', 'float64']:
                sns.histplot(df[col], ax=axes[i], kde=True)
            else:
                sns.histplot(df[col].astype(str), ax=axes[i], kde=False)
            axes[i].set_title(f'Histogram of {col}')
        
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])
        
        plt.tight_layout()
        plt.show()
        
    def boxplot(self, df):
        df_num, df_cat = self.split_numerical_categorical()
        
        n_cols = 3
        n_rows = (len(df_num.columns) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        axes = axes.flatten()
        
        for i, col in enumerate(df_num.columns):
            sns.boxplot(x=df_num[col], ax=axes[i])
            axes[i].set_title(f'Boxplot of {col}')
        
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])
        
        plt.tight_layout()
        plt.show()

    def correlation_matrix(self, df):
        df_num, df_cat = self.split_numerical_categorical()
        
        matrix = df_num.corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
        plt.title("Correlation Matrix")
        plt.show()
    
    def violin_plot(self, df):
        df_num, df_cat = self.split_numerical_categorical()
        
        n_cols = 3
        n_rows = (len(df_num.columns) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        axes = axes.flatten()
        
        for i, col in enumerate(df_num.columns):
            sns.violinplot(y=df_num[col], ax=axes[i])
            axes[i].set_title(f'Violin Plot of {col}')
        
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])
        
        plt.tight_layout()
        plt.show()
        
df = pd.read_csv(r'C:\Users\pasti\Documents\GitHub\Progetto_Finale_Corso_Python\train_clean.csv')
        
visual1 = Visualization(df)
#visual1.histplot(df)
#visual1.correlation_matrix(df)
#visual1.boxplot(df)
#visual1.violin_plot(df)
