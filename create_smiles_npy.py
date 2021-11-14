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

idx_src_train_arr = []
image_src_train_list = []
smiles = open('USPTO-50K/src-train.txt', 'r')
content = smiles.read()
chunks = content.split('\n')
chunks.remove('')
chunks2 = content.split('\n')
chunks2.remove('')
content_src_train = smiles.read()
for idx in range(len(chunks)):
    chunks[idx] = chunks[idx].replace(" ", "").split('>',1)[1]
    chunks2[idx] = chunks2[idx].replace(" ", "")
    chunks[idx] = chunks[idx].replace(" ", "").split('>',1)[0].replace("<RX_","")
    if(chunks2[idx].split('>',1)[0].replace("<RX_","") == "1"):
        # print(idx)
        idx_src_train_arr.append(idx)
smiles.close()
mols = [Chem.MolFromSmiles(x) for x in chunks]
# valid_mols = [i for i in mols if i != None]
for idx in idx_src_train_arr:
    AllChem.ComputeGasteigerCharges(mols[idx])
    contribs = [mols[idx].GetAtomWithIdx(i).GetDoubleProp('_GasteigerCharge') for i in range(mols[idx].GetNumAtoms())]
    fig = SimilarityMaps.GetSimilarityMapFromWeights(mols[idx], contribs, colorMap=None,  contourLines=10)
    fig.savefig("USPTO-50K-IMAGES-SRC-TRAIN/mol-{0}.png".format(idx), bbox_inches='tight')
# for idx in idx_src_train_arr:
#     mols[idx].draw(False, "USPTO-50K-IMAGES-SRC-TRAIN/mol-{0}.png".format(idx))
    # image_src_train_list.append(mols[idx].draw(show = False, filename = png))

smiles = open('USPTO-50K/tgt-train.txt', 'r')
content = smiles.read()
chunks = content.split('\n')
chunks.remove('')
for idx in range(len(chunks)):
    chunks[idx] = chunks[idx].replace(" ", "")
smiles.close()
mols = [Chem.MolFromSmiles(x) for x in chunks]
# for idx in idx_src_train_arr:
#     mols[idx].draw(False, "USPTO-50K-IMAGES-TGT-TRAIN/mol-{0}.png".format(idx))
for idx in idx_src_train_arr:
    AllChem.ComputeGasteigerCharges(mols[idx])
    contribs = [mols[idx].GetAtomWithIdx(i).GetDoubleProp('_GasteigerCharge') for i in range(mols[idx].GetNumAtoms())]
    fig = SimilarityMaps.GetSimilarityMapFromWeights(mols[idx], contribs, colorMap=None,  contourLines=10)
    fig.savefig("USPTO-50K-IMAGES-TGT-TRAIN/mol-{0}.png".format(idx), bbox_inches='tight')

#get the list of images from our first type of reactions
for filename in glob.glob('USPTO-50K-IMAGES-SRC-TRAIN/*'):
    # print(filename)
    for idx in idx_src_train_arr:
        # print("USPTO-50K-IMAGES-SRC-TRAIN/mol-{0}.png".format(idx))
        if(filename == "USPTO-50K-IMAGES-SRC-TRAIN/mol-{0}.png".format(idx)):
            img = cv2.imread(filename)
            grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(grey_img, (128, 128) , interpolation= cv2.INTER_AREA)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
            # resized = cv2.erode(resized, kernel, iterations=1)
            flatten = resized.flatten()
            image_src_train_list.append(flatten)
            np.save("USPTO-50K-IMAGES-SRC-TRAIN/mol-{0}.npy".format(idx), asarray(flatten))
            # print("shrunk {0}".format(idx))
            # f = open("USPTO-50K-IMAGES-SRC-TRAIN/mol-{0}.npy".format(idx), "w")
            # f.write(asarray(flatten))
            # f.close()
            os.remove("USPTO-50K-IMAGES-SRC-TRAIN/mol-{0}.png".format(idx))


#get the matching reactant images
image_tgt_train_list = []
for filename in glob.glob('USPTO-50K-IMAGES-TGT-TRAIN/*'):
    for idx in idx_src_train_arr:
        if(filename == "USPTO-50K-IMAGES-TGT-TRAIN/mol-{0}.png".format(idx)):
            img = cv2.imread(filename)
            grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(grey_img, (128, 128) , interpolation= cv2.INTER_AREA)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
            # resized = cv2.erode(resized, kernel, iterations=1)
            flatten = resized.flatten()
            image_tgt_train_list.append(flatten)
            np.save("USPTO-50K-IMAGES-TGT-TRAIN/mol-{0}.npy".format(idx), asarray(flatten))
            # print("shrunk {0}".format(idx))
            # f = open("USPTO-50K-IMAGES-TGT-TRAIN/mol-{0}.npy".format(idx), "w")
            # f.write(asarray(flatten))
            # f.close()
            os.remove("USPTO-50K-IMAGES-TGT-TRAIN/mol-{0}.png".format(idx))

idx_src_test_arr = []
image_src_test_list = []
smiles = open('USPTO-50K/src-test.txt', 'r')
content = smiles.read()
chunks = content.split('\n')
chunks.remove('')
chunks2 = content.split('\n')
chunks2.remove('')
content_src_test = smiles.read()
for idx in range(len(chunks)):
    chunks[idx] = chunks[idx].replace(" ", "").split('>',1)[1]
    chunks2[idx] = chunks2[idx].replace(" ", "")
    chunks[idx] = chunks[idx].replace(" ", "").split('>',1)[0].replace("<RX_","")
    if(chunks2[idx].split('>',1)[0].replace("<RX_","") == "1"):
        # print(idx)
        idx_src_test_arr.append(idx)
smiles.close()
mols = [Chem.MolFromSmiles(x) for x in chunks]
# for idx in idx_src_test_arr:
#     mols[idx].draw(False, "USPTO-50K-IMAGES-SRC-TEST/mol-{0}.png".format(idx))
    # image_src_test_list.append(mols[idx].draw(show = False, filename = png))
for idx in idx_src_test_arr:
    AllChem.ComputeGasteigerCharges(mols[idx])
    contribs = [mols[idx].GetAtomWithIdx(i).GetDoubleProp('_GasteigerCharge') for i in range(mols[idx].GetNumAtoms())]
    fig = SimilarityMaps.GetSimilarityMapFromWeights(mols[idx], contribs, colorMap=None,  contourLines=10)
    fig.savefig("USPTO-50K-IMAGES-SRC-TEST/mol-{0}.png".format(idx), bbox_inches='tight')

smiles = open('USPTO-50K/tgt-test.txt', 'r')
content = smiles.read()
chunks = content.split('\n')
chunks.remove('')
for idx in range(len(chunks)):
    chunks[idx] = chunks[idx].replace(" ", "")
smiles.close()
mols = [Chem.MolFromSmiles(x) for x in chunks]
# for idx in idx_src_test_arr:
#     mols[idx].draw(False, "USPTO-50K-IMAGES-TGT-TEST/mol-{0}.png".format(idx))
for idx in idx_src_test_arr:
    AllChem.ComputeGasteigerCharges(mols[idx])
    contribs = [mols[idx].GetAtomWithIdx(i).GetDoubleProp('_GasteigerCharge') for i in range(mols[idx].GetNumAtoms())]
    fig = SimilarityMaps.GetSimilarityMapFromWeights(mols[idx], contribs, colorMap=None,  contourLines=10)
    fig.savefig("USPTO-50K-IMAGES-TGT-TEST/mol-{0}.png".format(idx), bbox_inches='tight')

#get the list of images from our first type of reactions
for filename in glob.glob('USPTO-50K-IMAGES-SRC-TEST/*'):
    # print(filename)
    for idx in idx_src_test_arr:
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
    for idx in idx_src_test_arr:
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