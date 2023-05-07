# -*- coding: utf-8 -*-
import uvicorn ## ASGI
import pandas as pd
import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModel

from fastapi import FastAPI
from test_strings import TestString

app = FastAPI() #uvicorn main:app --reload

model_ckpt = "sentence-transformers/multi-qa-mpnet-base-dot-v1"

#load tokenizer for the specified model
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
model = AutoModel.from_pretrained(model_ckpt)

#loading the model to cuda device
device = torch.device("cuda")
model.to(device)

def concatenate_text(examples):
    """create a dictionary object with key "text" and store the concatenated string as value

    Parameters
    ----------
    examples : str
        The file location of the spreadsheet

    Returns
    -------
    dict
        a dictory object containing concatened row values in specified format
    """

    return {
        "text": examples["title"]
                + " \n "
                + examples["body"]
                + " \n "
                + examples["comments"]
    }

def cls_pooling(model_output):
    """a function to apply CLS pooling operation on-top of the contextualized word embeddings

    Parameters
    ----------
    model_output
      word embeddings

    Returns
    -------
      output from first token
    """
    return model_output.last_hidden_state[:, 0]


def get_embeddings(text_list):
    """a function to get the text embeddings

    Parameters
    ----------
    text_list : str
      text string

    Returns
    -------
      1x768 dimensional embedding
    """
    # encode input tokens and return as PyTorch tensors
    encoded_input = tokenizer(
        text_list, padding=True, truncation=True, return_tensors="pt"
    )
    encoded_input = {k: v.to(device) for k, v in encoded_input.items()}

    # computing token embeddeding by passing values from encoded dictionary
    model_output = model(**encoded_input)
    return cls_pooling(model_output)



issues_dataset = load_dataset("lewtun/github-issues", split="train")
issues_dataset = issues_dataset.filter(
    lambda x: (x["is_pull_request"] == False and len(x["comments"]) > 0)
)

columns = issues_dataset.column_names
columns_to_keep = ["title", "body", "html_url", "comments"]
columns_to_remove = set(columns_to_keep).symmetric_difference(columns)
issues_dataset = issues_dataset.remove_columns(columns_to_remove)

issues_dataset.set_format("pandas")
df = issues_dataset[:]

comments_df = df.explode("comments", ignore_index=True)

comments_dataset = Dataset.from_pandas(comments_df)

comments_dataset = comments_dataset.map(
    lambda x: {"comment_length": len(x["comments"].split())}
)

comments_dataset = comments_dataset.filter(lambda x: x["comment_length"] > 15)
comments_dataset = comments_dataset.map(concatenate_text)

embeddings_dataset = comments_dataset.map(
    lambda x: {"embeddings": get_embeddings(x["text"]).detach().cpu().numpy()[0]}
)

embeddings_dataset.add_faiss_index(column="embeddings")

@app.get("/")
async def root():
    return {"message": "testing the api call"}

@app.post('/predict')
async def query_dataset(data:TestString):
    data = data.dict()
    question=data['input']
    question_embedding = get_embeddings([question]).cpu().detach().numpy()
    scores, samples = embeddings_dataset.get_nearest_examples(
        "embeddings", question_embedding, k=5
    )

    # convert the matched samples to pandas dataframe
    samples_df = pd.DataFrame.from_dict(samples)
    samples_df["scores"] = scores
    samples_df.sort_values("scores", ascending=False, inplace=True)

    return samples_df.to_json(orient='records')

#if __name__ == '__main__':
#    uvicorn.run(app, host = '127.0.0.1', port=8000)
