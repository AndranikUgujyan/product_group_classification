import os
import product_model
import pandas as pd
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from product_model import app_config
from product_model.data_proc.normalizer import TextNormalizer

random_state = 42
ABS_DIR_PATH = os.path.dirname(os.path.abspath(product_model.__file__))


def read_and_normalize_dataset(dataset_path, main_dataset_plot_path):
    dataset_df = pd.read_csv(dataset_path, sep='delimiter', header=None)
    dataset_df = dataset_df[0].str.split(';', expand=True)
    dataset_df.columns = dataset_df.iloc[0]
    dataset_df = dataset_df[1:]
    dataset_df = dataset_df.rename(columns={'productgroup': 'product_group'})
    # print(dataset_df.product_group.value_counts())
    dataset_df["product_group"] = dataset_df["product_group"].astype('category')

    dataset_df["product_group_cat"] = dataset_df.product_group.replace(
        to_replace=['BICYCLES', 'CONTACT LENSES', 'USB MEMORY', 'WASHINGMACHINES'],
        value=[0, 1, 2, 3])

    # dataset_df["product_group_cat"] = dataset_df["product_group"].cat.codes
    cols = ["main_text", "add_text", "manufacturer"]
    dataset_df['text'] = dataset_df[cols].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)

    dataset_df['text'] = dataset_df['text'].apply(TextNormalizer().normalize)
    print(dataset_df.product_group.value_counts())
    dataset_df.product_group.value_counts().plot(kind='bar', title='Count (product)')
    plt.savefig(main_dataset_plot_path, dpi=300)
    return dataset_df


def split_train_val_test(normalized_df):
    train_df, test_df = train_test_split(normalized_df, test_size=0.1)
    return train_df, test_df


if __name__ == "__main__":
    abs_dir_path = os.path.dirname(os.path.abspath(product_model.__file__))

    path_of_dataset = os.path.join(abs_dir_path, app_config['main_dataset'])

    path_of_train = os.path.join(abs_dir_path, app_config['train_norm_data_path'])
    path_of_test = os.path.join(abs_dir_path, app_config['test_norm_data_path'])

    main_data_plot_abs_path = os.path.join(ABS_DIR_PATH, app_config["main_data_plot_path"])
    norm_df = read_and_normalize_dataset(path_of_dataset, main_data_plot_abs_path)
    print(norm_df)
    train_norm, test_norm = split_train_val_test(norm_df)

    train_norm.to_csv(path_of_train, index=False)
    test_norm.to_csv(path_of_test, index=False)
