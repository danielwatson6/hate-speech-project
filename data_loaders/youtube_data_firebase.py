import os
import re

import numpy as np
import pandas as pd
import tensorflow as tf

import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore

# initialise database locally
cred = credentials.Certificate("../secrets/online-extremism.json")
firebase_admin.initialize_app(cred)

db = firestore.client()

comments_ref = db.collection(u"comments")
docs = users_ref.stream()

for doc in docs:
    print(u"{} => {}".format(doc.id, doc.to_dict()))
