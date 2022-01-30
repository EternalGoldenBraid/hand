import numpy as np  # Numpy for arrays

def dual_frame(L=10, N=4, s=0.2):

    # L window has size LxL
    # N polynomial order
    # s relative spread of Gaussian window (as stdev of Gaussian)
    # s1, s2 block indices
    
    # Independent variables (horizontal and vertical)
    t_2, t_1 = np.meshgrid(np.linspace(-1, 1, L), np.linspace(-1, 1, L))  
    
    M = []
    for n in range(0, N + 1):  # loop over homogenous bivariate polynomials of degree n
        for k_1 in range(0, n + 1):  # degree on t_1
            k_2 = n - k_1  # degree on t_2
            M.append(np.ravel(t_1) ** k_1 * np.ravel(t_2) ** k_2)
            #M.append(np.ravel(t_1) * np.ravel(t_2))
    
    M = np.array(M).T  # Convert from list to numpy array and transpose for column vec
    
    w = np.exp(-((0.5 / s ** 2) * t_2 ** 2 + (0.5 / s ** 2) * t_1 ** 2))  # Gaussian window
    Mtilde = M @ np.linalg.pinv(M.T @ np.diag(np.ravel(w)) @ M)  # THE DUAL FRAME
    #Mtilde = M @ (M.T @ np.diag(np.ravel(w)) @ M)  # No pinv
    #Mtilde = M
    return M, Mtilde, w
    
    #buffer1 = np.zeros(z.shape)
    #buffer2 = np.zeros(z.shape)
    
    # coefficients from inner product with the dual frame
def approx(block, M, Mtilde, w, L=10, N=4, s=0.2):
    c = (Mtilde.T * (np.ravel(w)).T) @ np.ravel(block)      
    # synthesis with frame and coeffs from inner product with the dual frame
    hatBlock = np.reshape(M @ c, [L, L])
    return hatBlock
    
    #buffer1[s1:s1+L, s2:s2+L] = buffer1[s1:s1+L, s2:s2+L] + np.reshape(hatBlock, [L, L])
    #buffer2[s1:s1+L, s2:s2+L] = buffer2[s1:s1+L, s2:s2+L] + 1
    #
    #hatz = buffer1 / buffer2
    #return hatz

def init_blocks(size, step, L):
    #block_height = int(size[0]/step)
    #block_width  = int(size[1]/step)
    #block_height = step
    #block_width  = step
    #block_idxs = np.empty((block_width, block_height))
    block_idxs = []
    S1 = np.unique(
            np.concatenate( ([i for i in range(0, (size[0] - L), step)], [size[0] - L])))
    S2 = np.unique(
            np.concatenate( ([i for i in range(0, (size[1] - L), step)], [size[1] - L])))
    for s1 in S1:
        for s2 in S2:
            #block_idxs[s1,s2]=np.array([s1,s2])
            block_idxs.append(np.array([s1,s2]))
    
    blocks_row = int(size[1]/L)
    n_blocks = blocks_row**2
    
    vstack_idxs=[]

    for idx in range(0,n_blocks,blocks_row):
        vstack_idxs.append(np.arange(idx,idx+blocks_row))

    return block_idxs, vstack_idxs
    
def stack_blocks(blocks, image, stack_idxs, L):
    row_idx = 0
    for idx in stack_idxs:
        if row_idx == image.shape[0]: break
        row = np.hstack(blocks[idx])
        image[row_idx:row_idx+L,:] = row
        row_idx += L
    
