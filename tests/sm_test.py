from images_segmentation.models.simple_model import SimpleModel
from images_segmentation.models.shell import AnswerMatrix
from images_segmentation.data import ImagesDataModule
import numpy as np
import matplotlib.pyplot as plt

def prepare_dm():
    dm = ImagesDataModule(
            data_dir = "dataset/dataset1",
            batch_size = 4,
            height = 150,
            wigth = 120
        )
    
    dm.prepare_data()
    dm.setup(stage = "fit")

    return dm

def sm_t():
    dm = prepare_dm()

    model = SimpleModel()

    image, alpha = dm.val_dataset[15]
    print(image.shape)
    print(alpha)

    output = model(image)

    print(output.shape)
    print(output)

def answer_matrix_test():
    cb = AnswerMatrix()

    matrix = np.ones((11, 11), dtype = int)

    fig = cb.make_img_matrix(matrix)
    plt.show()

if __name__ == "__main__":
    answer_matrix_test()