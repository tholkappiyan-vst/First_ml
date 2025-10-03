import zipfile
import os

with zipfile.ZipFile("achive.zip", 'r') as zip_ref:
    zip_ref.extractall("dataset")
