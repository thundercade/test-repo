#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


@author: dupr0017@umn.edu
"""
#imports
import streamlit as st
import pandas as pd
# st.write(pd.__version__)

import pickle as pkl
from tqdm import tqdm

import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from spacy import displacy
from string import punctuation
from collections import Counter
from heapq import nlargest
from PIL import Image
import torch
import os
# nlp = spacy.load("en_core_web_sm")
# https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.2.0/en_core_web_sm-3.2.0-py3-none-any.whl

import sentence_transformers
from sentence_transformers import SentenceTransformer, util

with open("tokyo_corpus_embeddings.pkl" , "rb") as file_1, open("tokyo_df.pkl" , "rb") as file_2, open("tokyo_corpus.pkl" , "rb") as file_3:
    corpus_embeddings = pkl.load(file_1)
    df = pkl.load(file_2)
    corpus = pkl.load(file_3)

with open("tokyo_sum_df.pkl" , "rb") as file_4:
    sum_df = pkl.load(file_4)

with open("tokyo_df1.pkl" , "rb") as file_5:
    df1 = pkl.load(file_5)

embedder = SentenceTransformer('all-MiniLM-L6-v2')

# @st.cache(allow_output_mutation=True)
# 	    def load_model():
# 	        return SentenceTransformer('all-MiniLM-L6-v2')
# 	    embedder = load_model()


image = Image.open('tokyo_night.jpg')
st.image(image, use_column_width=True)

st.title("Find Your Hotel in Tokyo!")

# st.write(corpus_embeddings.shape)
# st.write(df.shape)

queries = st.text_input("Enter the kind of hotel you're looking for:")
queries = list([queries])

# Find the closest 5 sentences of the corpus for each query sentence based on cosine similarity
top_k = min(5, len(corpus))
for query in queries:
    query_embedding = embedder.encode(query, convert_to_tensor=True)

    # We use cosine-similarity and torch.topk to find the highest 5 scores
    cos_scores = util.pytorch_cos_sim(query_embedding, corpus_embeddings)[0]
    top_results = torch.topk(cos_scores, k=top_k)

    st.markdown("""---""")
    st.write("You searched for:   ", query, "\n")
    st.subheader("""**Here are our top 5 recommendations:**""")

    for score, idx in zip(top_results[0], top_results[1]):
        # st.write("(Score: {:.4f})".format(score))
        # st.write(corpus[idx], "(Score: {:.4f})".format(score))
        st.markdown("""---""")
        row_dict = df.loc[df['all_review']== corpus[idx]]
        row2_dict = sum_df.loc[sum_df['all_review']== corpus[idx]]
        row3_dict = df1.loc[df1['hotel']==row_dict['hotel'].values[0]]
        st.write("Hotel Name: " , row_dict['hotel'].values[0])
        st.write("Hotel Review Summary: " , row2_dict['summary'].values[0])
        st.write("Tripadvisor Link: [here](%s)" %row3_dict['url'].values[0], "\n")
        st.markdown("""---""")


st.write("All of your code ran this time!")
