# from openbabel import pybel
from rdkit.Chem.Draw import SimilarityMaps
from rdkit.Chem import AllChem
import matplotlib
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
# for idx in idx_src_train_arr:
#     mols[idx].draw(False, "USPTO-50K-IMAGES-SRC-TRAIN/mol-{0}.png".format(idx))
#     # image_src_train_list.append(mols[idx].draw(show = False, filename = png))


valid_mols = [i for i in mols if i != None]
for mol in valid_mols:
    AllChem.ComputeGasteigerCharges(mol)
    contribs = [mol.GetAtomWithIdx(i).GetDoubleProp('_GasteigerCharge') for i in range(mol.GetNumAtoms())]
    fig = SimilarityMaps.GetSimilarityMapFromWeights(mol, contribs, colorMap=None,  contourLines=10)