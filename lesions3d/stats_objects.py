from datasets import *
from monai.data import box_area
import matplotlib.pyplot as plt



if __name__ == "__main__":
    pass
    batch_size=8
    dataset = LesionsDataModule(percentage=0.05, batch_size=8)
    dataset.setup("all")
    loader = dataset.all_dataloader()

    lengths = list()
    widths = list()
    depths = list()
    areas = list()

    for i, batch in enumerate(loader):
        for subj_boxes in batch["boxes"]:
            length = (subj_boxes[:,3] - subj_boxes[:,0])
            width = (subj_boxes[:,4] - subj_boxes[:,1])
            depth = (subj_boxes[:,5] - subj_boxes[:,2])
            area = (length*width*depth).detach().tolist()

            lengths += length.detach().tolist()
            widths += depth.detach().tolist()
            depths += length.detach().tolist()
            areas += area


    plt.hist(lengths, bins=50)
    plt.title("Lengths")
    plt.xlim((0,0.2))
    plt.show()
    plt.hist(widths, bins=50)
    plt.title("Widths")
    plt.xlim((0,0.2))
    plt.show()
    plt.hist(depths, bins=50)
    plt.title("Depths")
    plt.xlim((0,0.2))
    plt.show()
    plt.hist(areas, bins=50)
    plt.title("Areas")
    plt.xlim((0,0.005))
    plt.show()







