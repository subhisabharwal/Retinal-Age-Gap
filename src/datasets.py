import cv2
from torch.utils.data import Dataset
from src.preprocessing import preprocess_image


class RetinalDataset(Dataset):
    def __init__(self, df, transform=None):
        """
        df: pandas dataframe (train_df or val_df)
        transform: torchvision transforms (augmentation + normalization)
        """
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_path = row["image_path"]
        label = row["age"]

        # load image
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # preprocessing (fixed)
        img = preprocess_image(img)

        # transforms (augmentation + normalization)
        if self.transform:
            img = self.transform(img)

        return img, label
