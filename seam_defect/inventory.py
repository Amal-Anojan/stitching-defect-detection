import pandas as pd


def inventory_views(path):
    inventory_df = pd.read_csv(path)
    # Recreate image dataframe
    image_df = inventory_df[inventory_df["is_image"]].copy()

    # Recreate class distribution
    class_distribution = (
        image_df.groupby("folder_2").size().reset_index(name="image_count")
    )

    class_distribution["class_id"] = (
        class_distribution["folder_2"].str.extract(r"^(\d+)").astype(int)
    )

    class_distribution = class_distribution.sort_values("class_id").reset_index(
        drop=True
    )

    # Recreate fabric and class order
    fabric_order = sorted(image_df["folder_1"].unique())

    class_order = class_distribution["folder_2"].tolist()

    # Select three images from each combination
    combination_counts = image_df.groupby(["folder_1", "folder_2"]).size()

    assert combination_counts.min() >= 3, "Some combinations have fewer than 3 images"

    stratified_sample = (
        image_df.groupby(["folder_1", "folder_2"], group_keys=False)
        .sample(n=3, random_state=42)
        .reset_index(drop=True)
    )

    return (
        image_df,
        class_distribution,
        fabric_order,
        class_order,
        combination_counts,
        stratified_sample,
    )
