import numpy as np 

'''returns a grid of random (complex) gaussian distribution whose 2dfft is real 
valued only (approximately)'''


def transformation(n):
    '''creates transformation matrix for endomorphism that reflects the rows of 
    a matrix along the middle vertical axis
    -n: the number of rows of the matrix to be transformed'''
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

def reflect(grid):
    '''reflects a grid along the middle axis'''
    trans_matrix = transformation(grid.shape[1])
    return np.matmul(grid,trans_matrix)

def reflect_row(row):
    '''reflects an array about right end. That is, [a,b,c] --> [c,b,a]'''
    reflected = []
    for entry in row[::-1]:
        reflected.append(entry)
        
    return reflected

def gaussian_complex(shape):
    '''returns an array of random gaussian complex distribution of specified shape
    -shape: tuple or int. Returns either a grid or an array'''
    return np.random.normal(size =shape) + np.random.normal(size = shape)*1j

def flip(conjugate_grid):
    '''flips grid with required symmetry'''
    first = reflect(conjugate_grid.transpose()).transpose()
    second = []
    for row in first: 
        second.append(reflect_row(row))
    return np.array(second)

def rows(shape):
    ''' returns the rows with required symmetries to complete grid'''
    first = np.random.normal(size = shape) + np.random.normal(size = shape)*1j
    first_prime = np.insert(first, len(first),1)
    first_conj = np.conj(first)
    
    return np.insert(first_prime, len(first_prime), reflect_row(first_conj))

def final(grid1, grid2, rows):
    '''puts it all together'''
    first = np.vstack((rows, grid1, rows, grid2))
    side = np.array([np.insert(rows, 0, 1)]).T
    
    return np.hstack((side, first))

'''returns a grid of random gaussian distribution with the required symmetries 
in order for fft to return a real matrix. 
     n - shape of desired grid (square only at this point)'''
def real(n):
    a = gaussian_complex( (n//2 -1, n-1) ) 
    a_conj = np.conj(a)
    b = flip(a_conj)
    c = rows(n//2 -1)
    
    return final(a,b,c)
    
    
