import os
import pandas as pd
import product_model
from product_model import app_config
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from product_model.data_proc.embedding import TextEmbedding

random_state = 42

abs_dir_path = os.path.dirname(os.path.abspath(product_model.__file__))

path_of_dataset = os.path.join(abs_dir_path, app_config['main_dataset'])
row_df = pd.read_csv(path_of_dataset, sep='delimiter', header=None)
df = row_df[0].str.split(';', expand=True)
df.columns = df.iloc[0]
df = df[1:]
df = df.rename(columns={'productgroup': 'product_group'})
df["product_group"] = df["product_group"].astype('category')
df["product_group_cat"] = df["product_group"].cat.codes
cols = ["main_text", "add_text", "manufacturer"]
df['text'] = df[cols].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
df.head()

df["text_embedded"] = df["text"].apply(TextEmbedding().sent_embedding)
print(df.head())

path_of_embedded_dataset = os.path.join(abs_dir_path, app_config['main_embedded_dataset'])
df = pd.read_csv(path_of_embedded_dataset)
print(df)
df.to_csv(path_of_embedded_dataset, index=False)
train_embedded_df, test_embedded_df = train_test_split(df, test_size=0.1)
#
path_of_train_embedded_df = os.path.join(abs_dir_path, app_config['train_embedded_data_path'])
path_of_test_embedded_df = os.path.join(abs_dir_path, app_config['test_embedded_data_path'])

train_embedded_df.to_csv(path_of_train_embedded_df, index=False)
test_embedded_df.to_csv(path_of_test_embedded_df, index=False)
