import numpy as np
from openbabel import pybel
from rdkit import Chem

def _generate_AX(self):
    self.log('Creating features and adjacency matrices..')

    data = []
    smiles = []
    data_S = []
    data_A = []
    data_X = []
    data_D = []
    data_F = []
    data_Le = []
    data_Lv = []

    max_length = max(mol.GetNumAtoms() for mol in self.data)
    max_length_s = max(len(Chem.MolToSmiles(mol)) for mol in self.data)

    for i, mol in enumerate(self.data):
        A = self._genA(mol, connected=True, max_length=max_length)
        D = np.count_nonzero(A, -1)
        if A is not None:
            data.append(mol)
            smiles.append(Chem.MolToSmiles(mol))
            data_S.append(self._genS(mol, max_length=max_length_s))
            data_A.append(A)
            data_X.append(self._genX(mol, max_length=max_length))
            data_D.append(D)
            data_F.append(self._genF(mol, max_length=max_length))

            L = D - A
            Le, Lv = np.linalg.eigh(L)

            data_Le.append(Le)
            data_Lv.append(Lv)

    self.log(date=False)
    self.log('Created {} features and adjacency matrices  out of {} molecules!'.format(len(data),
                                                                                        len(self.data)))

    self.data = data
    self.smiles = smiles
    self.data_S = data_S
    self.data_A = data_A
    self.data_X = data_X
    self.data_D = data_D
    self.data_F = data_F
    self.data_Le = data_Le
    self.data_Lv = data_Lv
    self.__len = len(self.data)

if __name__ == '__main__':
    # data = SparseMolecularDataset()
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
    mols = [pybel.readstring("smi", x) for x in chunks]

    _generate_AX(mols)
    # data.generate('gdb9.sdf', filters=lambda x: x.GetNumAtoms() <= 9)
    # data.save('gdb9_9nodes.sparsedataset')