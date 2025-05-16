import pandas as pd

def load_data(data_dir="data"):
    print("Loading datas...")
    big_matrix = pd.read_csv(f"{data_dir}/big_matrix.csv")
    small_matrix = pd.read_csv(f"{data_dir}/small_matrix.csv")
    social_network = pd.read_csv(f"{data_dir}/social_network.csv")
    social_network["friend_list"] = social_network["friend_list"].map(eval)
    item_categories = pd.read_csv(f"{data_dir}/item_categories.csv")
    item_categories["feat"] = item_categories["feat"].map(eval)
    user_features = pd.read_csv(f"{data_dir}/user_features.csv")
    item_daily_feat = pd.read_csv(f"{data_dir}/item_daily_features.csv")
    print("All data loaded.")
    

    return {
        "big_matrix": big_matrix,
        "small_matrix": small_matrix,
        "social_network": social_network,
        "item_categories": item_categories,
        "user_features": user_features,
        "item_daily_feat": item_daily_feat
    }

if __name__ == "__main__":
    dfs = load_data()
    for name, df in dfs.items():
        print(f"\n{name} ({df.shape}):")
        print(df.head())
