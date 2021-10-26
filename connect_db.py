#!/usr/bin/env python
# coding: utf-8

# In[1]:

from pymongo import MongoClient

import numpy as np
from tqdm import tqdm



def insertInfo(df):

    client = MongoClient('mongodb://localhost:27017/')
    infodb = client.Infodb
    userInfo = infodb.userInfo

    for index, instance in tqdm(df.iterrows(), total=df.shape[0]):
        ID = instance["id"]
        name = instance["name"]
        birth = instance["birth"]
        embeddings = instance["embedding"].tobytes()
        user = {'_id': ID, 'name': name, 'birth': birth, 'embeddings': embeddings}
        try :
            userInfo.insert_one(user)
        except :
            print('ID already exists.')



def load_info(ID):

    client = MongoClient('mongodb://localhost:27017/')
    infodb = client.Infodb
    userInfo = infodb.userInfo


    results = userInfo.find({"_id": ID}, {'name': True ,'embeddings': True})

    embedding = []
    for result in results:
        #id = result["_id"]
        name = result['name']
        embedding_bytes = result["embeddings"]
        embedding = np.frombuffer(embedding_bytes, dtype='float32')

    return name, embedding

