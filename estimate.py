""" Find the estimated jacobian from each timeseries in a directory. Each file in teh directory must be a timeseries with the parameters in tbe filename. Uses method from Following Steuer et al. 2003 (https://academic.oup.com/bioinformatics/article/19/8/1019/235343) -jacobian*covariance+covariance*jacobian.T=-Fluctiation matrix. This can rearragned to [kron(covariance,I)+kron(I,covariance)CommutationMatrix]vec(J)=vec(D) - of the form Ax=b and with A and b known. This is combined with zeros in the jacobian that are known due to network structure, leading to an overdetermined system solved using linear least squares:
-params - directory- str - name of directory containing data files. each data file must contain a single time series, at a particular combination of variables. 


-outputs - results/eigenvalues/${directory} (with directory.replace('/','_')) - file - csv file with headers [phi,analytic_eigenvalue,estimate_eigenvalue,complex] that contain [phi from filename, analytic_eigenvalue calculated using parameters in filename, estimate_eigenvalue calculated using Steur method above, complex - bool - 1=complex eigenvalue- 0=real]
        - ${directory.split('/')[-1]}_by_phi.png - image - plot of eigenvalue by phi for analytic and etsimated eigenvalues.
 """

###ADD FUNCTIONS TO EXTRACT INFO ON LOCALISATION #####

from __future__ import division
import numpy as np
from scipy.stats import linregress
import sys
import matplotlib
matplotlib.use('Agg')
import numpy as np
import os
from copy import copy
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.linalg as sl



def load_time_series(fname):
    return(np.genfromtxt(fname,delimiter=','))

def find_covariance(data):
    return(np.cov(data.T))

def jacobian(phi,gamma):
    J_l=np.matrix([[10*phi-2-9*gamma,-9],[3*gamma,-3]])
    J_d=np.matrix([[-3,0],[0,-10]])
    L=np.matrix([[3,-1,-1,0,0,-1],[-1,3,-1,0,-1,0],[-1,-1,2,0,0,0],[0,0,0,1,0,-1],[0,-1,0,0,1,0],[-1,0,0,-1,0,2]])
    jac=np.kron(np.diag(np.ones(6)),J_l)+np.kron(L,J_d)
    return(jac[:,[0,2,4,6,8,10,1,3,5,7,9,11]][[0,2,4,6,8,10,1,3,5,7,9,11],:])

def commutation_matrix(n,m=None):
    if not m:
        m=n
    x=np.zeros([n*m,n*m])
    for i in range(n*m):
        x[i,((i%m)*n+np.floor(i/m)).astype(int)]=1
    return(x)

def A(covarianceMatrix):
    """A=[kron(covariance,I)+kron(I,covariance)CommutationMatrix]"""
    I=np.identity(np.shape(covarianceMatrix)[0])
    return(np.kron(covarianceMatrix,I)+np.dot(np.kron(I,covarianceMatrix),commutation_matrix(np.shape(covarianceMatrix)[0])))

def gaussian_elimination(A,b):
    """ Perform Gaussian elimination on augmented matrix (A|b) """
    nRows,nCols=np.shape(A)
    nCols=nCols-1
    A=np.append(A,np.expand_dims(b,1),axis=1).astype('float')
    for i in range(nCols):
        #Find maximum row in this column
        maxEl=np.max(abs(A[i:,i]))
        maxRow=np.argsort(abs(A[i:,i]))[-1]+i
        #Swap maximum row with current row
        tmp=copy(A[maxRow,:])
        A[maxRow,:]=copy(A[i,:])
        A[i,:]=copy(tmp)
        #Make all rows below this one zero in this column
        for j in range(nRows)[i+1:]:
            if A[i,i]!=0:
                c=-A[j,i]/A[i,i]
            else:
                c=0
            A[j,:]=A[j,:]+c*A[i,:]
            A[j,i]=0
    return(A)

def known_elements_from_matrix(knownJacobian):
    """get dict of known jacobian elements from given jacobian of known elements and unknown elements (nan)"""
    kJacobian={}
    size=np.shape(knownJacobian)[0]
    for i in range(size):
        for j in range(size):
            if not np.isnan(knownJacobian[i,j]):
                kJacobian[(i,j)]=knownJacobian[i,j]
    return(kJacobian)

def known_zeros(knownJacobian):
    for i in range(np.shape(knownJacobian)[0]):
        for j in range(np.shape(knownJacobian)[0]):
            if knownJacobian[i,j]!=0:
                knownJacobian[i,j]=np.nan
    return(known_elements_from_matrix(knownJacobian))

def add_known_jacobian_elements_for_least_squares(knownElements,A,numJacobianRows):
    """Add the equations to the augmented matrix A that describe the known elements of J"""
    rowLength=np.shape(A)[1]
    A=A[~np.all(A==0,axis=1)]
    for entry in knownElements.iteritems():
        tmpRow=np.zeros(rowLength)
        tmpRow[entry[0][0]+entry[0][1]*numJacobianRows]=1
        tmpRow[-1]=entry[1]
        A=np.vstack([A,tmpRow])
    return(A)

def least_squares(A,b):
    x=np.dot(np.dot(sl.inv(np.dot(A.T,A)),A.T),b)
    return(x)

def estimate_jacobian_least_squares(fname,knownJacobian,trim=None,noise=1):
    """Estimate jacobian when the fluctiation matrix D is assumed to be constant but proportional to mean biomass for each species and give analytic jacobian using least squares"""
    if trim:
        data=load_time_series(fname)[-trim:,:]
    else:
        data=load_time_series(fname)
    D=-np.diag(np.mean(data,0)).flatten('F')*noise**2
    data=data-np.mean(data,0)
    numVariables=np.shape(data)[1]
    covarianceMatrix=find_covariance(data)
    AForFirstElimination=A(covarianceMatrix)
    gauss_1=gaussian_elimination(AForFirstElimination,D)
    AForLeastSquares=add_known_jacobian_elements_for_least_squares(known_zeros(knownJacobian),np.hstack([AForFirstElimination,np.expand_dims(D,1)]),len(knownJacobian))
    x=least_squares(AForLeastSquares[:,:-1],AForLeastSquares[:,-1])
    return(np.reshape(x,[numVariables,numVariables],'F'))

def analytic_and_estimate_jacobian_least_squares(fname,trim=None):
    """ Find analytic and estimated jacobian from file using least squares - when the parameters are in the filename
    
    args:   fname - string - data file name
            trim - int - number of data points to use, backwards from end of time series

    outputs: analyticJacobian - array - calculated jacobian
             estimateJacobian - array - estimated jacobian from time series"""
    dataFilename=fname.split("/")[-1]
    phi = float(dataFilename.split("=")[1].split("g")[0][:-1].replace('_','.'))
    gamma = float(dataFilename.split("=")[2].split("n")[0][:-1].replace('_','.'))
    noise = float(dataFilename.split("=")[3].split(".")[0].replace('_','.'))
    print(phi,gamma,noise)
    analyticJacobian=jacobian(phi,gamma)
    estimateJacobian=estimate_jacobian_least_squares(fname,copy(analyticJacobian),trim,noise)
    return(phi,analyticJacobian,estimateJacobian)

def localisation(eigenvector):
    if len(np.unique(np.round(np.real(eigenvector),decimals=8)))==2:
        return(0)
    else:
        return(1)

def max_eigenvalues(analyticJacobian,estimateJacobian):
    estimateJacobianWithZeros=np.copy(estimateJacobian)
    for row,val1 in enumerate(np.array(analyticJacobian)):
        for col,val2 in enumerate(val1):
            if val2==0:
                estimateJacobianWithZeros[row,col]=0

    analyticEigValVec=sl.eig(analyticJacobian)
    maxEigs=[np.max(analyticEigValVec[0]),np.max(sl.eig(estimateJacobianWithZeros)[0])]
    localised=localisation(analyticEigValVec[1][:,np.argmax(analyticEigValVec[0])])
    if np.imag(maxEigs[0]) == 0:
        im=0
    else:
        im=1
    return(np.real(maxEigs[0]),np.real(maxEigs[1]),im,localised)




def analytic_and_estimate_max_eigenvalues_from_directory(directory,trim=None):
    allMaxEigenvalues=[]
    for fname in os.listdir(directory):
        phi,analyticJacobian,estimateJacobian=analytic_and_estimate_jacobian_least_squares("".join(directory+'/'+fname),trim)
        maxEigs=max_eigenvalues(analyticJacobian,estimateJacobian)
        allMaxEigenvalues.append([phi,maxEigs[0],maxEigs[1],maxEigs[2],maxEigs[3]])
    return(allMaxEigenvalues)

def plot_by_phi(data,outFilename='tmp.png'):
    plt.figure('f')
    plt.close('f')
    plt.figure('f',figsize=(8,8))
    data=np.asarray(data)
    plt.plot(np.sort(data[:,0]),data[np.argsort(data[:,0]),1],marker='x',label='Analytic',color='g')
    plt.scatter(data[:,0],data[:,2],marker='o',label='Estimate')
    plt.legend()
    plt.xlabel(r'\phi')
    plt.ylabel('Max Eigenvalue')
    plt.savefig(outFilename)
    return()


def from_directory(directory,trim=None):
    maxEigenvalues=analytic_and_estimate_max_eigenvalues_from_directory(directory,trim=trim)
    splitDirectoryName=directory.split('/')
    if len(splitDirectoryName[-1]) == 0:
       title=splitDirectoryName[-2]
    else:
       title=splitDirectoryName[-1]
    plot_by_phi(maxEigenvalues, "".join(title + '_by_phi.png'))
    return(maxEigenvalues)

def results_file_name(directory):
    return("".join('results/eigenvalues/'+directory.replace('/','_')+'.csv'))

def write_eigenvalues(maxEigenvalues,outfile='tmp.csv'):
    with open(outfile, 'wb') as outf:
        outf.write('phi,analytic_eigenvalue,estimate_eigenvalue,complex,localised\n')
        np.savetxt(outf, maxEigenvalues, delimiter=",")
    return()

if __name__ == '__main__':
    if len(sys.argv)== 3:
        write_eigenvalues(from_directory(sys.argv[1],int(sys.argv[2])),results_file_name(sys.argv[1]))
    elif len(sys.argv)==2:
        write_eigenvalues(from_directory(sys.argv[1]),results_file_name(sys.argv[1]))
    else:
        raise ValueError('critical_point_estimator.py expects at most 2 argument but received {0}.'.format(len(sys.argv)-1))
