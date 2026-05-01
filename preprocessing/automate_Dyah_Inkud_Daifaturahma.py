import os
import re
import kagglehub
import pandas as pd

from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory


def clean_text(text):
    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    # Remove user @ references and '#' from hashtags
    text = re.sub(r"\@\w+|\#", "", text)
    # Remove punctuations
    text = re.sub(r"[^\w\s]", "", text)
    # Remove emojis
    text = re.sub(
        r"["
        r"\U0001F600-\U0001F64F"  # emoticons
        r"\U0001F300-\U0001F5FF"  # symbols & pictographs
        r"\U0001F680-\U0001F6FF"  # transport & map symbols
        r"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        r"\U00002702-\U000027B0"
        r"\U000024C2-\U0001F251"
        "]+",
        "",
        text,
    )
    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()
    # Remove non-ASCII characters
    text = re.sub(r"[^\x00-\x7F]+", "", text)
    # Remove special characters
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    # Remove numbers
    text = re.sub(r"\d+", "", text)
    # Convert to lowercase
    text = text.lower()

    return text


def main():
    output_raw_dir = "./twitter-dataset-raw"
    os.makedirs(output_raw_dir, exist_ok=True)

    raw_source = kagglehub.dataset_download(
        "anggapurnama/twitter-dataset-ppkm",
        output_dir=output_raw_dir,
        force_download=True,
    )

    print("Path dataset:", raw_source)

    file_path = f"{raw_source}/INA_TweetsPPKM_Labeled_Pure.csv"
    data_load = pd.read_csv(
        file_path, sep="\t", engine="python", quotechar='"', on_bad_lines="skip"
    )

    factory = StopWordRemoverFactory()
    stopword = factory.create_stop_word_remover()

    data_load["cleaned_text"] = data_load["Tweet"].apply(clean_text)
    data_load["cleaned_text"] = data_load["cleaned_text"].apply(stopword.remove)

    na = data_load["cleaned_text"].isna().sum()
    print(f"Jumlah baris dengan cleaned_text NaN: {na}")

    if na > 0:
        print("Terdapat baris dengan cleaned_text NaN. Menghapus baris tersebut.")
        data_load = data_load.dropna(subset=["cleaned_text"])

    output_clean_dir = "./twitter-dataset-cleaned"
    os.makedirs(output_clean_dir, exist_ok=True)

    df_final = data_load[["cleaned_text", "sentiment"]].copy()
    df_final.to_csv(f"{output_clean_dir}/data_clean.csv", index=False)

    print(
        f"Preprocessing completed. Cleaned data saved to {output_clean_dir}/data_clean.csv"
    )


if __name__ == "__main__":
    main()
