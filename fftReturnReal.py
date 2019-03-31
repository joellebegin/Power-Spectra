import numpy as np 

'''creates transformation matrix for endomorphism that reflects the rows of a matrix along the middle axis'''
def transformation(n):
    matrix = [0]*n
    j = n-1
    matrix[j] = 1
    matrix = np.array(matrix)
    j -=1

    for i in range(1,n):
        row = [0]*n
        row[j] +=1
        row = np.array(row)
        matrix = np.vstack((matrix,row))
        j-=1
        
    return matrix.astype(float)

'''reflects a grid along the middle axis'''
def reflect(grid):
    trans_matrix = transformation(grid.shape[1])
    return np.matmul(grid,trans_matrix)

'''reflects an array about right end'''
def reflect_row(row):
    reflected = []
    for entry in row[::-1]:
        reflected.append(entry)
        
    return reflected

'''returns an array of random gaussian complex distribution of specified shape'''
def gaussian_complex(shape):
    return np.random.normal(size =shape) + np.random.normal(size = shape)*1j

'''flips grid with required symmetry'''
def flip(conjugate_grid):
    first = reflect(conjugate_grid.transpose()).transpose()
    second = []
    for row in first: 
        second.append(reflect_row(row))
    return np.array(second)

''' returns the rows with required symmetries to complete grid'''
def rows(shape):
    first = np.random.normal(size = shape) + np.random.normal(size = shape)*1j
    first_prime = np.insert(first, len(first),1)
    first_conj = np.conj(first)
    
    return np.insert(first_prime, len(first_prime), reflect_row(first_conj))
 
'''puts it all together'''
def final(grid1, grid2, rows):
    first = np.vstack((rows, grid1, rows, grid2))
    side = np.array([np.insert(rows, 0, 1)]).T
    
    return np.hstack((side, first))


'''returns a grid of random gaussian distribution with the required symmetries in order for fft to return a real matrix. 
     n - shape of desired grid (square only at this point)'''
def real(n):
    a = gaussian_complex( (n//2 -1, n-1) ) 
    a_conj = np.conj(a)
    b = flip(a_conj)
    c = rows(n//2 -1)
    
    return final(a,b,c)
    
    