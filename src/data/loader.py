import os
import urllib.request
import zipfile

import pandas as pd


DATA_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"


def load_dataset(path: str = "data/raw/SMSSpamCollection") -> pd.DataFrame:
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = "sms_spam.zip"
        urllib.request.urlretrieve(DATA_URL, tmp)
        with zipfile.ZipFile(tmp, "r") as z:
            z.extractall(os.path.dirname(path))
        os.remove(tmp)

    df = pd.read_csv(path, sep="\t", header=None, names=["label", "message"])
    df = df.drop_duplicates(subset="message").reset_index(drop=True)
    df["label_enc"] = (df["label"] == "spam").astype(int)
    return df
