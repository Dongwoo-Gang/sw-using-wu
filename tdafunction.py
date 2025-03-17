from itertools import permutations, combinations
import numpy as np
import dionysus as d
rng = np.random.default_rng(seed=42) #optional

def random_subarray(array, d):
    '''For given array n x k, return subarray d x k'''
    m, n = array.shape
    
    if d > m:
        raise ValueError("d cannot be greater than m")
    
    # Randomly choose d rows without replacement
    random_rows = np.random.choice(m, size=d, replace=False)
    
    # Sort the row indices to maintain original order
    random_rows.sort()
    
    # Return the subarray
    return array[random_rows, :]

def farthest_subarray(arr: np.ndarray, m: int) -> np.ndarray:
    """
    Given an (n, d) numpy array and an integer m <= n, 
    selects an (m, d) subarray based on the following rule:
    
    1. Pick the first row randomly.
    2. Select the next row that is farthest from the first row (L2 norm).
    3. Continue selecting rows that maximize the minimum distance to the already selected rows.
    
    Parameters:
        arr (np.ndarray): Input (n, d) array.
        m (int): Number of rows to select.
    
    Returns:
        np.ndarray: Selected (m, d) subarray.
    """
    n, d = arr.shape
    assert m <= n, "m should be less than or equal to n"
    
    first_idx = np.random.randint(0, n)
    selected_indices = [first_idx]
    
    for _ in range(m - 1):
        # Compute distances from selected points
        min_distances = np.full(n, np.inf)
        
        for idx in selected_indices:
            distances = np.linalg.norm(arr - arr[idx], axis=1)  # L2 norm
            min_distances = np.minimum(min_distances, distances)  # Track min distance to selected set
        
        # Choose the index with the max of the minimum distances
        farthest_idx = np.argmax(min_distances)
        selected_indices.append(farthest_idx)
    
    return arr[selected_indices]

def find_kth_largest_index(nums, k):
    # Create a list of tuples (value, index)
    indexed_nums = list(enumerate(nums))
    
    # Sort the list based on values in descending order
    sorted_nums = sorted(indexed_nums, key=lambda x: x[1], reverse=True)
    
    # Return the index of the k-th largest element
    return sorted_nums[k-1][0]

def steenrod_diagonal(i, spx):
    '''Returns the image of applying the ith Steenrod coproduct to spx.'''
    answer = []
    n = len(spx)-1
    for U in combinations(range(n+1), n-i):
        left, right = list(spx), list(spx)
        for u in U:
            if (U.index(u) + u) % 2 == 1:
                left.remove(spx[u])
            else:
                right.remove(spx[u])
        answer.append((left, right))
    return answer

def steenord_square(cocycle, f, n, k):
    '''compute the image of Sq^k([alpha]) = [(alpha tensor alpha) cup_{|alpha|-k} (....)]. Here |alpha|=n'''
    if n < k:
        return []
    if cocycle != [] and len(cocycle[0]) != n+1:
        raise Exception("cocycle should be n-dim")
    if k == 0:
        return cocycle
    simp_lr = []
    out = []
    for simplex in f:
        simplex = list(simplex)
        if len(simplex) == n+k+1:
            for pair in steenrod_diagonal(n-k,simplex):
                if len(pair[0]) == len(pair[1]):
                    simp_lr.append(pair)
    for pair in permutations(cocycle,2):
        if len(set(pair[0]+pair[1])) != n+k+1:
            continue
        if pair in simp_lr:
            square = list(set(pair[0]+pair[1]))
            square.sort()
            out.append(square)
    return out

def cupprod(x,y, f): 
    '''compute x cup y over coef=z/2.'''
    out = []
    simplices = []
    for simplex in f:
        simplex = list(simplex)
        simplices.append(simplex)
    for i in range(len(x)):
        for j in range(len(y)):
            #if x[i][1] == y[j][0]:
            if x[i][-1] == y[j][0] and x[i][:-1]+y[j] in simplices:
                element = x[i][:-1]+y[j] #+[1]
                if element in out:
                    out.remove(element)
                else:
                    out.append(element)
            '''
            elif y[j][-1] == x[i][0] and y[j][:-1]+x[i] in simplices:
                element = y[j][:-1]+x[i] #+[1]
                if element in out:
                    out.remove(element)
                else:
                    out.append(element)
            '''
    return out

def is_cycle(chain,n): #n : dimension
    if chain != [] and len(chain[0]) != n+1:
        raise Exception("chain should be n-dim")
    boundary = []
    for i in range(len(chain)):
        for spx in combinations(chain[i], n):
            print(spx)
            if spx in boundary:
                boundary.remove(spx)
            else:
                boundary.append(spx)
    print(boundary)
    if boundary == []:
        return True
    else:
        return False
def is_cocycle(f, cochain,n): #n : dimension
    if cochain != [] and len(cochain[0]) != n+1:
        raise Exception("chain should be n-dim")
    for simplex in f:
        simplex = list(simplex)
        if len(simplex) == n+2:  # ndim-simplices
            coboundary = sum(list(face) in cochain for face in combinations(simplex, n+1))
            if coboundary % 2 != 0:  # For Z2 coefficients
                return False
    return True

def row_echelon_form(matrix):
    """Transform the given binary matrix to its row echelon form."""
    rows, cols = matrix.shape
    lead = 0  # The lead column index

    for r in range(rows):
        if lead >= cols:
            return matrix
        i = r
        while matrix[i, lead] == 0:
            i += 1
            if i == rows:
                i = r
                lead += 1
                if cols == lead:
                    return matrix
        # Swap rows i and r
        matrix[[i, r]] = matrix[[r, i]]
        
        # Make the leading entry 1 and eliminate below
        for i in range(r + 1, rows):
            if matrix[i, lead] == 1:
                matrix[i] ^= matrix[r]
        
        lead += 1
    return matrix

def rank_of_binary_matrix(matrix):
    """Calculate the rank of a binary matrix over GF(2)."""
    # Convert the input to a NumPy array of integers
    binary_matrix = np.array(matrix, dtype=int)
    
    # Get the row echelon form of the binary matrix
    ref_matrix = row_echelon_form(binary_matrix)
    
    # Count non-zero rows to determine the rank
    rank = np.sum(np.any(ref_matrix, axis=1))
    
    return rank

def coboundary_matrix(f, n):
    '''compute nsimp x (n-1)simp coboundary matrix A over z/2 '''
    rowsimp = [] #n-simplex
    colsimp = [] # n-1-simplex
    for simplex in f:
        simplex = list(simplex)
        if len(simplex)== n+1:
            rowsimp.append(simplex)
        elif len(simplex) == n:
            colsimp.append(simplex)
    coboundary = np.zeros([len(rowsimp),len(colsimp)])
    for i in range(len(rowsimp)):
        rsp = rowsimp[i]
        for csp in combinations(rsp, n):
            j = colsimp.index(list(csp))
            coboundary[i][j] = ( coboundary[i][j] + 1 ) % 2
    return [rowsimp, colsimp, coboundary]

def filtered_coboundary_matrix(f, n, t):
    '''compute nsimp x (n-1)simp coboundary matrix A over z/2, at filtration t'''
    rowsimp = [] #n-simplex
    colsimp = [] # n-1-simplex
    for simplex in f:
        if simplex.data <= t:
            simplex = list(simplex)
            if len(simplex)== n+1:
                rowsimp.append(simplex)
            elif len(simplex) == n:
                colsimp.append(simplex)
    coboundary = np.zeros([len(rowsimp),len(colsimp)])
    for i in range(len(rowsimp)):
        rsp = rowsimp[i]
        for csp in combinations(rsp, n):
            j = colsimp.index(list(csp))
            coboundary[i][j] = ( coboundary[i][j] + 1 ) % 2
    return [rowsimp, colsimp, coboundary]


def has_binary_solution(A, b):
    m, n = A.shape
    Ab = np.column_stack((A, b))
    
    # Perform Gaussian elimination in GF(2)
    rank = 0
    for j in range(n):
        pivot_row = None
        for i in range(rank, m):
            if Ab[i, j] == 1:
                pivot_row = i
                break
        
        if pivot_row is not None:
            # Swap rows to move pivot row up
            Ab[[rank, pivot_row]] = Ab[[pivot_row, rank]]
            # Eliminate below
            for i in range(rank + 1, m):
                if Ab[i, j] == 1:
                    Ab[i] ^= Ab[rank]
            rank += 1
    
    # Check for inconsistency: any row of form [0 ... 0 | 1]
    for i in range(rank, m):
        if np.all(Ab[i, :n] == 0) and Ab[i, n] == 1:
            return False
    
    return True

def is_cohomologous(coboundary_matrix, cocycle1, cocycle2):
    [rowsimp, colsimp, coboundary] = coboundary_matrix
    # coumpute s1 - s2 as the vector b
    if ( (cocycle1 != []) and (cocycle2 != []) ) and (len(cocycle1[0]) != len(cocycle2[0])):
        raise Exception("dimension should be the same!")
    df = np.zeros([len(rowsimp)])
    for spx in cocycle1:
        if spx in rowsimp:
            i = rowsimp.index(spx)
            df[i] = (df[i] + 1) % 2
    for spx in cocycle2:
        if spx in rowsimp:
            i = rowsimp.index(spx)
            df[i] = (df[i] + 1) % 2
    # check whether rank(A|b) == rank(A)
    #return (rank_of_binary_matrix(coboundary) == rank_of_binary_matrix(np.concatenate((coboundary, df.reshape(-1, 1)), axis=1)))
    coboundary = coboundary.astype(int)
    df = df.astype(int)
    return (has_binary_solution(coboundary,df.reshape(-1, 1)))

def sum_cocycles(cocycles):
    '''input : [cocycle1,cocycle2,...]. out : cocycle1+cocycle2+...'''
    out = []
    for cocycle in cocycles:
        for spx in cocycle:
            if spx in out:
                out.remove(spx)
            else:
                out.append(spx)
    return out

def kth_longest_index(cdgms, n, k):
    '''return index of n-dim cocycles having k-th longest persistence'''
    length = []
    for barcode in cdgms[n]:
        length.append(barcode.death - barcode.birth)
    idx = find_kth_largest_index(length,k)
    return idx
    
def get_cocycles(cm,f,cdgms, n, idx):
    '''return representing n-dim cocycles of index idx'''
    cpt = cdgms[n][idx]
    cocycle = []
    for sei in cm.cocycle(cpt.data):
        cs= f[sei.index]
        cocycle.append([])
        for i in range(len(cs)):
            cocycle[-1].append(cs[i])
    return cocycle


def find_wu_class(f,n,m,t): #n : dimension. m : total dimension(dim of manifold)
    '''find n-th Wu class, return cocycle'''
    cm = d.cohomology_persistence(f,prime=2, keep_cocycles = True)
    cdgms = d.init_diagrams(cm,f)
    cdgms_t_idx = [] # indicies of barcodes that are nontrivial at t
    wu_candidate = []
    for i in range(len(cdgms)):
        cdgms_t_idx.append([])
        for j in range(len(cdgms[i])):
            if cdgms[i][j].death >= t:
                cdgms_t_idx[i].append(j)
    for idx in cdgms_t_idx[n]:
        cpt = cdgms[n][idx] 
        cocycle = []
        for sei in cm.cocycle(cpt.data):
            cs= f[sei.index]  # (co)simplex
            cocycle.append([])
            for i in range(len(cs)):
                cocycle[-1].append(cs[i])        
        wu_candidate.append(cocycle)
    combined_wu_candidate = []
    if len(wu_candidate) >1:        
        for i in range(2,len(wu_candidate)+1):
            for cocycles in combinations(wu_candidate,i):
                combined_wu_candidate.append(sum_cocycles(list(cocycles)))
    wu_candidate = wu_candidate + combined_wu_candidate
    wu_candidate.append([])
    j = m-n
    basis_for_nmink = []
    for idxx in cdgms_t_idx[j]:
        cptx = cdgms[j][idxx]
        cocyclex = []
        for sei in cm.cocycle(cptx.data):
            cs= f[sei.index]
            cocyclex.append([])
            for i in range(len(cs)):
                cocyclex[-1].append(cs[i])
        basis_for_nmink.append(cocyclex)
    cob_mat = coboundary_matrix(f, m)
    for cocyclex in basis_for_nmink:
        for cocycle in wu_candidate[:]:
            if is_cohomologous(cob_mat,cupprod(cocycle,cocyclex,f), steenord_square(cocyclex, f, j, n)) == False:
                wu_candidate.remove(cocycle)
    if len(wu_candidate) == 1:
        return wu_candidate[0]
    else:
        print('no wu class')
        
def sw_class(f,n,m,t): #(m:dim of manifold)
    '''find n-th Stiefel-Whitney cocycles at filtration t'''
    cocycles = []
    try:
        for i in range(1,n+1):
            cocycles.append(steenord_square((find_wu_class(f,i,m,t)),f,i,n-i))
        cocycle = sum_cocycles(cocycles)
        return cocycle
    except:
        print("There is no Wu class")
    
def zero_cohomologous(coboundary_matrix,cocycle):
    return is_cohomologous(coboundary_matrix,cocycle,[])

def pers_wu_class(f,n,m,s,t):
    '''return n-th wu class if it exists, and return False o.w.'''
    wu_class = find_wu_class(f,n,m,t)
    is_pers_wu = True
    cm = d.cohomology_persistence(f,prime=2, keep_cocycles = True)
    cdgms = d.init_diagrams(cm,f)
    cdgms_t_idx = [] # indicies of barcodes that are nontrivial at t
    for i in range(len(cdgms)):
        cdgms_t_idx.append([])
        for j in range(len(cdgms[i])):
            if cdgms[i][j].death < t and cdgms[i][j].death >= s:
                cdgms_t_idx[i].append(j)
    j = m-n
    for idxx in cdgms_t_idx[j]:
        cptx = cdgms[j][idxx]
        birth = cptx.death #comological birth  for (d,b]
        cocyclex = []
        for sei in cm.cocycle(cptx.data):
            cs= f[sei.index]
            cocyclex.append([])
            for i in range(len(cs)):
                cocyclex[-1].append(cs[i])
        cob_mat = filtered_coboundary_matrix(f,n,birth)
        is_pers_wu == is_cohomologous(cob_mat, cupprod(wu_class,cocyclex,f), steenord_square(cocyclex, f, j, n))
        if is_pers_wu == False:
            break
    if is_pers_wu == True:
        return wu_class
    else:
        return False
    
def pers_sw_class(f,n,m,s,t):
    '''return n-th sw class if it exists, and return False o.w.'''
    pers_sw = True
    cocycles = []
    for i in range(1,n+1):
        wu_class = pers_wu_class(f,i,m,s,t)
        if wu_class == False:
            pers_sw == False
            break
        cocycles.append(steenord_square(wu_class,f,i,n-i))
    if pers_sw == False:
        return False
    cocycle = sum_cocycles(cocycles)
    return cocycle