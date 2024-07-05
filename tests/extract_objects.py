import cv2
import pandas as pd
import pathlib
import os
from tqdm import tqdm

def cut_image_and_return_df(
        path_to_dir, 
        path_to_image,
        path_to_labels = "dataset/dataset1/labels.csv",
        cut_and_labels = pd.DataFrame({"path" : [], "alpha" : []})
    ):
    img_path = pathlib.Path(path_to_image)
    image = cv2.imread(img_path)

    labels_path = pathlib.Path(path_to_labels)
    df = pd.read_csv(labels_path)

    image_labels = df[df["image"] == img_path.name.split("_")[-1]]

    if not os.path.isdir(f"{path_to_dir}/cut"):
        os.mkdir(f"{path_to_dir}/cut")

    i = 0
    for line in image_labels.iloc():
        idx, img, xtl, ytl, xbr, ybr, alpha = line.values
        cropped_image = image[ytl:ybr, xtl:xbr, :]

        img_cut = img.split(".")[0]
        new_name = f"{img_cut}_{i}.jpg"
        i += 1

        temp = pd.DataFrame({
                "path" : [new_name],
                "alpha" : [alpha]
        })

        cut_and_labels = pd.concat([cut_and_labels, temp], ignore_index = True)
        cv2.imwrite(path_to_dir + "/cut/" + new_name, cropped_image)
    
    return cut_and_labels

def cut_all(root_dir):
    image_names = os.listdir(root_dir)

    cut_and_labels = pd.DataFrame({"path" : [], "alpha" : []})

    print("Extracting object from images...")

    for img_name in tqdm(image_names):
        cut_and_labels = cut_image_and_return_df(
            path_to_dir = "dataset/dataset1",
            path_to_image = f"{root_dir}/{img_name}",
            cut_and_labels = cut_and_labels
        )

    cut_and_labels.to_csv("dataset/dataset1/cut_and_labels.csv", index = False)

    print("Extraction complete")


if __name__ == "__main__":
    cut_all("dataset/dataset1/images")