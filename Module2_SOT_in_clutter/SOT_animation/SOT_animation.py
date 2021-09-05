import os
import scipy.io
import matplotlib.pyplot as plt
import seaborn as sns


## constants for file paths
sotLinearWsSuccessFileName ='sot_linear_ws_success.mat'
dataDir = '/home/wabco/Downloads/sdxc/Git repos/Multi_Object_Tracking/Module2_SOT_in_clutter/SOT_animation/'


def readMatFile(filePath):
    """
    Function to read data from mat file

    Args:
        filePath (str): 

    Returns:
        [matFileData] (dict) : variable containing workspace variables
    """    
    matFileData = None
    if os.path.isfile(filePath):
        matFileData = scipy.io.loadmat(filePath)
    else:
        print(f"{filePath} Filepath not found")
    return matFileData


wsData = readMatFile(dataDir + sotLinearWsSuccessFileName)

if wsData is not None:
    # extract variables from file
    groundTruthStates = wsData['true_states']
    nnEstX  = wsData['nn_states_x']
    nnEstP  = wsData['nn_states_P']
    pdaEstX = wsData['pda_states_x']
    pdaEstP = wsData['pda_states_P']
    gsfEstX = wsData['gsf_states_x']
    gsfEstP = wsData['gsf_states_P']

    # print(groundTruthStates.shape, nnEstX.shape, nnEstP.shape, \
    # pdaEstX.shape, pdaEstP.shape, gsfEstX.shape, gsfEstP.shape)


fig, ax = plt.subplots(nrows=1, ncols=3)

## Nearest neighbour estimates
ax[0].plot(groundTruthStates[:,0], groundTruthStates[:,1], label='Ground Truth')
# ax[0].plot(nnEstX[:,0], nnEstX[:,1], '--', label='NN Estimates')

## Probabilistic Data Assocation estimates
ax[1].plot(groundTruthStates[:,0], groundTruthStates[:,1], label='Ground Truth')
# ax[1].plot(pdaEstX[:,0], pdaEstX[:,1], '--', label='PDA Estimates')

## Gaussian Sum filter estimates
ax[2].plot(groundTruthStates[:,0], groundTruthStates[:,1], label='Ground Truth')
# ax[2].plot(gsfEstX[:,0], gsfEstX[:,1], '--', label='GSF Estimates')

plt.grid(True)
plt.legend()
plt.show()

