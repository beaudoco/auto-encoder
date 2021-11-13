# from openbabel import pybel
from rdkit.Chem.Draw import SimilarityMaps
from rdkit.Chem import AllChem
import matplotlib
from rdkit import Chem
import cv2
import numpy as np
from numpy import asarray

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
# for idx in idx_src_train_arr:
#     mols[idx].draw(False, "USPTO-50K-IMAGES-SRC-TRAIN/mol-{0}.png".format(idx))
#     # image_src_train_list.append(mols[idx].draw(show = False, filename = png))


valid_mols = [i for i in mols if i != None]
for mol in valid_mols:
    AllChem.ComputeGasteigerCharges(mol)
    contribs = [mol.GetAtomWithIdx(i).GetDoubleProp('_GasteigerCharge') for i in range(mol.GetNumAtoms())]
    fig = SimilarityMaps.GetSimilarityMapFromWeights(mol, contribs, colorMap=None,  contourLines=10)
    grey_img = cv2.cvtColor(fig, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(grey_img, (128, 128) , interpolation= cv2.INTER_AREA)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    resized = cv2.erode(resized, kernel, iterations=1)
    flatten = resized.flatten()
    image_src_train_list.append(flatten)
    np.save("USPTO-50K-IMAGES/mol-{0}.npy".format(idx), asarray(flatten))