from openbabel import pybel
from numpy import asarray
import numpy as np
import os
import cv2
import glob
from tempfile import TemporaryFile
from rdkit.Chem.Draw import SimilarityMaps
from rdkit.Chem import AllChem
from rdkit import Chem
import matplotlib.pyplot as plt
import multiprocessing

def smiles_image_create_src_train(x):
    mol =(Chem.MolFromSmiles(x[1]))
    AllChem.ComputeGasteigerCharges(mol)
    contribs = [mol.GetAtomWithIdx(i).GetDoubleProp('_GasteigerCharge') for i in range(mol.GetNumAtoms())]
    fig = SimilarityMaps.GetSimilarityMapFromWeights(mol, contribs, colorMap=None,  contourLines=10)
    fig.savefig("USPTO-50K-IMAGES-SRC-TRAIN/mol-{0}.png".format(x[0]), bbox_inches='tight')
    plt.close()
    # fig.FinishDrawing()
    # fig.cla()
    # fig.

def smiles_image_create_src_test(x):
    mol =(Chem.MolFromSmiles(x[1]))
    AllChem.ComputeGasteigerCharges(mol)
    contribs = [mol.GetAtomWithIdx(i).GetDoubleProp('_GasteigerCharge') for i in range(mol.GetNumAtoms())]
    fig = SimilarityMaps.GetSimilarityMapFromWeights(mol, contribs, colorMap=None,  contourLines=10)
    fig.savefig("USPTO-50K-IMAGES-SRC-TEST/mol-{0}.png".format(x[0]), bbox_inches='tight')
    plt.close()
    # fig.FinishDrawing()
    # fig.cla()
    # fig.
    
def smiles_image_create_tgt_train(x):
    mol =(Chem.MolFromSmiles(x[1]))
    AllChem.ComputeGasteigerCharges(mol)
    contribs = [mol.GetAtomWithIdx(i).GetDoubleProp('_GasteigerCharge') for i in range(mol.GetNumAtoms())]
    fig = SimilarityMaps.GetSimilarityMapFromWeights(mol, contribs, colorMap=None,  contourLines=10)
    fig.savefig("USPTO-50K-IMAGES-TGT-TRAIN/mol-{0}.png".format(x[0]), bbox_inches='tight')
    plt.close()
    # fig.FinishDrawing()
    # fig.cla()
    # fig.

def smiles_image_create_tgt_test(x):
    mol =(Chem.MolFromSmiles(x[1]))
    AllChem.ComputeGasteigerCharges(mol)
    contribs = [mol.GetAtomWithIdx(i).GetDoubleProp('_GasteigerCharge') for i in range(mol.GetNumAtoms())]
    fig = SimilarityMaps.GetSimilarityMapFromWeights(mol, contribs, colorMap=None,  contourLines=10)
    fig.savefig("USPTO-50K-IMAGES-TGT-TEST/mol-{0}.png".format(x[0]), bbox_inches='tight')
    plt.close()
    # fig.FinishDrawing()
    # fig.cla()
    # fig.

tuple_src_train_arr = []
idx_src_train_arr = []
image_src_train_list = []
smiles = open('USPTO-50K/src-train.txt', 'r')
content = smiles.read()
chunks = content.split('\n')
chunks.remove('')
chunks2 = content.split('\n')
chunks2.remove('')
content_src_train = smiles.read()
smiles.close()
for idx in range(len(chunks)):
    chunks[idx] = chunks[idx].replace(" ", "").split('>',1)[1]
    chunks2[idx] = chunks2[idx].replace(" ", "")
    chunks[idx] = chunks[idx].replace(" ", "").split('>',1)[0].replace("<RX_","")
    if(chunks2[idx].split('>',1)[0].replace("<RX_","") == "1"):
        tuple_src_train_arr.append((idx, chunks[idx]))
        idx_src_train_arr.append(idx)

pool = multiprocessing.Pool()
pool = multiprocessing.Pool(multiprocessing.cpu_count())
pool.map(smiles_image_create_src_train, tuple_src_train_arr)
pool.close()

smiles = open('USPTO-50K/tgt-train.txt', 'r')
content = smiles.read()
chunks = content.split('\n')
chunks.remove('')
for idx in range(len(chunks)):
    chunks[idx] = chunks[idx].replace(" ", "")
smiles.close()
tuple_tgt_train_arr = []
for idx in idx_src_train_arr:
    tuple_tgt_train_arr.append((idx, chunks[idx]))

pool = multiprocessing.Pool()
pool = multiprocessing.Pool(multiprocessing.cpu_count())
pool.map(smiles_image_create_tgt_train, tuple_tgt_train_arr)
pool.close()

#get the list of images from our first type of reactions
for filename in glob.glob('USPTO-50K-IMAGES-SRC-TRAIN/*'):
    # print(filename)
    for idx in tuple_src_train_arr:
        # print("USPTO-50K-IMAGES-SRC-TRAIN/mol-{0}.png".format(idx))
        if(filename == "USPTO-50K-IMAGES-SRC-TRAIN/mol-{0}.png".format(idx)):
            img = cv2.imread(filename)
            resized = cv2.resize(img, (128, 128) , interpolation= cv2.INTER_AREA)
            # grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # resized = cv2.resize(grey_img, (128, 128) , interpolation= cv2.INTER_AREA)
            # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
            # resized = cv2.erode(resized, kernel, iterations=1)
            flatten = resized.flatten()
            image_src_train_list.append(flatten)
            np.save("USPTO-50K-IMAGES-SRC-TRAIN/mol-{0}.npy".format(idx), asarray(flatten))
            os.remove("USPTO-50K-IMAGES-SRC-TRAIN/mol-{0}.png".format(idx))


#get the matching reactant images
image_tgt_train_list = []
for filename in glob.glob('USPTO-50K-IMAGES-TGT-TRAIN/*'):
    for idx in tuple_src_train_arr:
        if(filename == "USPTO-50K-IMAGES-TGT-TRAIN/mol-{0}.png".format(idx)):
            img = cv2.imread(filename)
            resized = cv2.resize(img, (128, 128) , interpolation= cv2.INTER_AREA)
            # grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # resized = cv2.resize(grey_img, (128, 128) , interpolation= cv2.INTER_AREA)
            # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
            # resized = cv2.erode(resized, kernel, iterations=1)
            flatten = resized.flatten()
            image_tgt_train_list.append(flatten)
            np.save("USPTO-50K-IMAGES-TGT-TRAIN/mol-{0}.npy".format(idx), asarray(flatten))
            os.remove("USPTO-50K-IMAGES-TGT-TRAIN/mol-{0}.png".format(idx))

tuple_src_test_arr = []
idx_src_test_arr = []
image_src_test_list = []
smiles = open('USPTO-50K/src-test.txt', 'r')
content = smiles.read()
chunks = content.split('\n')
chunks.remove('')
chunks2 = content.split('\n')
chunks2.remove('')
content_src_test = smiles.read()
smiles.close()
for idx in range(len(chunks)):
    chunks[idx] = chunks[idx].replace(" ", "").split('>',1)[1]
    chunks2[idx] = chunks2[idx].replace(" ", "")
    chunks[idx] = chunks[idx].replace(" ", "").split('>',1)[0].replace("<RX_","")
    if(chunks2[idx].split('>',1)[0].replace("<RX_","") == "1"):
        # print(idx)
        tuple_src_test_arr.append((idx, chunks[idx]))

pool = multiprocessing.Pool()
pool = multiprocessing.Pool(multiprocessing.cpu_count())
pool.map(smiles_image_create_src_test, tuple_src_test_arr)
pool.close()

smiles = open('USPTO-50K/tgt-test.txt', 'r')
content = smiles.read()
chunks = content.split('\n')
chunks.remove('')
smiles.close()
for idx in range(len(chunks)):
    chunks[idx] = chunks[idx].replace(" ", "")

idx_tgt_test_arr = []
for idx in idx_src_test_arr:
    tuple_tgt_train_arr.append((idx, chunks[idx]))

pool = multiprocessing.Pool()
pool = multiprocessing.Pool(multiprocessing.cpu_count())
pool.map(smiles_image_create_tgt_test, idx_tgt_test_arr)
pool.close()

#get the list of images from our first type of reactions
for filename in glob.glob('USPTO-50K-IMAGES-SRC-TEST/*'):
    # print(filename)
    for idx in tuple_src_test_arr:
        # print("USPTO-50K-IMAGES-SRC-TEST/mol-{0}.png".format(idx))
        if(filename == "USPTO-50K-IMAGES-SRC-TEST/mol-{0}.png".format(idx)):
            img = cv2.imread(filename)
            grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(grey_img, (128, 128) , interpolation= cv2.INTER_AREA)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
            # resized = cv2.erode(resized, kernel, iterations=1)
            flatten = resized.flatten()
            image_src_test_list.append(flatten)
            np.save("USPTO-50K-IMAGES-SRC-TEST/mol-{0}.npy".format(idx), asarray(flatten))
            # print("shrunk {0}".format(idx))
            # f = open("USPTO-50K-IMAGES-SRC-TEST/mol-{0}.npy".format(idx), "w")
            # f.write(asarray(flatten))
            # f.close()
            os.remove("USPTO-50K-IMAGES-SRC-TEST/mol-{0}.png".format(idx))


#get the matching reactant images
image_tgt_test_list = []
for filename in glob.glob('USPTO-50K-IMAGES-TGT-TEST/*'):
    for idx in tuple_src_test_arr:
        if(filename == "USPTO-50K-IMAGES-TGT-TEST/mol-{0}.png".format(idx)):
            img = cv2.imread(filename)
            grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(grey_img, (128, 128) , interpolation= cv2.INTER_AREA)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
            # resized = cv2.erode(resized, kernel, iterations=1)
            flatten = resized.flatten()
            image_tgt_test_list.append(flatten)
            np.save("USPTO-50K-IMAGES-TGT-TEST/mol-{0}.npy".format(idx), asarray(flatten))
            # print("shrunk {0}".format(idx))
            # f = open("USPTO-50K-IMAGES-TGT-TEST/mol-{0}.npy".format(idx), "w")
            # f.write(asarray(flatten))
            # f.close()
            os.remove("USPTO-50K-IMAGES-TGT-TEST/mol-{0}.png".format(idx))

# fps = [x.calcfp() for x in mols]
# print(fps[0].bits, fps[1].bits) 
# print(fps[0] | fps[1])