# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 08:28:37 2020

@author: Kwaku
"""
import random
import math
import statistics as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import seaborn as sns




# settings for seaborn plotting style
sns.set(color_codes=True)
# settings for seaborn plot sizes
sns.set(rc={'figure.figsize':(5,5)})
np.random.seed(20)#just the to add a little bit of repeatbilty in the randomnes
#np.random.seed()
def theorwhichman(size):
    rand=[]
    for i in range(0,size):
        rand.append(random.uniform(0, 1))
    return rand
size=100

a=theorwhichman(size)

mu = st.mean(a)
sigma = st.stdev(a)

x = np.linspace(-1, 1, size)
a.sort()

print("Sigma:", sigma)
print("Mu:", mu)


#Creating bits to be transmitted 
# _____________________________________________________________________________
#                               Bit generator
# _____________________________________________________________________________
# random bits generator
def bits_gen(values): # function takes a list of values between 0 and 1
    data = []
    
    for i in values:
        #if the values is less than 0.5 append a 0
        if i < 0.5: 
            data.append(0)
        else:
            #if the value found in the list is greater than 0.5 append 1
            data.append(1)

    return data

# ________________________________________________________________________________
#               mapping of bits to symbol using constellation maps
# ______________________________________________________________________________
def BPSK(bits):
    bpsk = []
    for k in bits:
        if k == 1:
            bpsk.append(1)
        else:
            bpsk.append(-1)
    bpsk.append(1)
    return bpsk

def fourQAM(bits):
    FQAM = []
    M = 2
    subList = [bits[n:n + M] for n in range(0, len(bits), M)]
    for k in subList:
        if k == [0, 0]:
            FQAM.append(complex(1 / np.sqrt(2), 1 / np.sqrt(2)))
        elif k == [0, 1]:
            FQAM.append(complex(-1 / np.sqrt(2), 1 / np.sqrt(2)))
        elif k == [1, 1]:
            FQAM.append(complex(-1 / np.sqrt(2), -1 / np.sqrt(2)))
        # elif(k==[1,0]):
        elif k == [1, 0]:
            FQAM.append(complex(1 / np.sqrt(2), -1 / np.sqrt(2)))
    FQAM.append((1+1j)/sqrt2)
    return FQAM

def eight_PSK(bits):
    EPSK = []
    M = 3
    subList = [bits[n:n + M] for n in range(0, len(bits), M)]
    for k in subList:
        if k == [0, 0, 0]:
            EPSK.append(complex(1 , 0))
        elif k == [0, 0, 1]:
            EPSK.append((1+1j)/np.sqrt(2))
        elif k == [0, 1, 1]:
            EPSK.append( 1j)
        elif k == [0, 1, 0]:
            EPSK.append((-1+1j)/np.sqrt(2))
        elif k == [1, 1, 0]:
            EPSK.append(-1)
        elif k == [1, 1, 1]:
            EPSK.append((-1-1j)/np.sqrt(2))
        elif k == [1, 0, 1]:
            EPSK.append(-1j)
        elif k == [1, 0, 0]:
            EPSK.append((1-1j)/np.sqrt(2))
    EPSK.append(1)
    return EPSK

#________________________________________________________________
#                       Sigma calculation 
#__________________________________________________________________

def sigma(domain,M):# M is the number of symbols
    sigma=[]
    for i in domain:
        sigma.append(1/np.sqrt(math.pow(10, (i/ 10)) * 2 * math.log2(M)))
    return sigma


#______________________________________________________________________________
#                       create noise 
#______________________________________________________________________________

def noise(size,sigma):
    noiseList = np.random.normal(0,sigma,size)
    return noiseList

#__________________________________________________________________________
#                       add noise 
#______________________________________________________________________________

def addnoise(transmitted,channels,L,m,snr, sigma_):# assuming transmited comes with the memory symbols padded
    recieved=[]
    M=m
    k=snr
    
    
    #print(sigma_)
    
    
    for i in range(L-1,len(transmitted)):
        sample=np.random.normal(0,1,1)[0]
        recieved.append(transmitted[i]*channels[0]+transmitted[i-1]*channels[0+1]
                        +transmitted[i-2]*channels[2]+sigma_*(sample+(sample)*1j))
    return recieved #without the padding

#__________________________________________________________________________
#                       DFE function 
#______________________________________________________________________________

def DFE(recieved,channels,L,options,memory):
    Options =options# [1,-1]# these are the option available for bpsk
    symbols = memory#[1]*(L-1) #the first 1 is the memory symbols
    s=0 # symbol mover
    for i in range(0,len(recieved)):
        guess=[]
        n=len(channels)-1# length of the chanel L-1
        #calculating the product but from second position
        sumof=0
        for j in range(1,n):
           sumof+= symbols[n-1+s]*channels[j]
           n-=1
           
        for k in Options:
            guess.append(np.abs(recieved[i]-((k)*channels[0]+sumof))**2)
        estimate=Options[guess.index(min(guess))]
        symbols.append(estimate)
        s+=1
    return symbols[L-1:] #final 
#__________________________________________________________________________
#                       Options and memory generator
#______________________________________________________________________________
def OptMemGen(i,L):
    #BPSK=1 #4QAM=2 8PSK=3
    if i==1:
        Options = [1,-1]# these are the option available for bpsk
        memory= [1]*(L-1) #the first 1 is the memory symbols
        return Options,memory
    elif(i==2):
        Options = [(1+1j)/np.sqrt(2), (-1+1j)/np.sqrt(2), (-1-1j)/np.sqrt(2), (1-1j)/np.sqrt(2)]
        memory=[(1+1j)/np.sqrt(2)]*(L-1)
        return Options,memory
    elif(i==3):
         Options=[1, (1+1j)/np.sqrt(2), 1j, (-1+1j)/np.sqrt(2), -1, (-1-1j)/np.sqrt(2), -1j, (1-1j)/np.sqrt(2)]
         memory=[(1+1j)/np.sqrt(2)]*(L-1)
         return Options,memory 

#__________________________________________________________________________
#                      Bit Error calculation
#______________________________________________________________________________

def bit_errors(sent, recieved):
    error = 0
    for k in range(0,len(recieved)):
        if sent[k] != recieved[k]:
            error += 1
    BER = error / len(recieved)*100
    return BER

#transmitted=[1,1,1,-1,1,-1,1]
#channels = [0.89+0.92j,0.42-0.37j,0.19+0.12j]    
#Recieved=addnoise(transmitted,channels,3)#[1.5,1.2,1,-1.2,-1.5,0.2]
#print(Recieved)


###############################################################################
"""

Lefa's graph function














"""
###############################################################################



sqrt2 = np.sqrt(2)
CIR = [0.89 + 0.92j, 0.42 - 0.37j, 0.19 + 0.12j]
N = 200


def Reverse(x):
  return x [::-1]
    
# generate deltas for BPSK    
def findDeltaBPSK(recieved, Symbols =[[]],i=0, c = CIR):
    Options = [-1,1]
    delta = []
    for s in Symbols:
        for j in Options:
            delta.append(np.abs(recieved[i] - (j*c[0] + s[1]*c[1] + s[0]*c[2]) )**2)
    return delta

# generate deltas for 4QAM
def findDelta4QAM(recieved, Symbols =[[]],i=0, c = CIR):
    Options = [(1+1j)/sqrt2, (-1+1j)/sqrt2, (-1-1j)/sqrt2, (1-1j)/sqrt2]
    delta = []
    for s in Symbols:
        for j in Options:
            delta.append(np.abs(recieved[i] - (j*c[0] + s[1]*c[1] + s[0]*c[2]) )**2)
            
    return delta

# generate deltas for 8PSK
def findDelta8PSK(recieved, Symbols =[[]],i=0, c = CIR):
    Options = [1, (1+1j)/sqrt2, 1j, (-1+1j)/sqrt2, -1, (-1-1j)/sqrt2, -1j, (1-1j)/sqrt2]
    delta = []
    for s in Symbols:
        for j in Options:
            delta.append(np.abs(recieved[i] - (j*c[0] + s[1]*c[1] + s[0]*c[2]) )**2)
    return delta


def BPSK_MLSE(recieved, N, c):
    # Generate the bits, then get their symbols from the constellation map    
    Bits = []
    for i in range(8):
        Bits.append(format(8+i, 'b'))    
    Symbols = []
    # print(Bits)
    for i in range(8):
        Sym = []
        for j in range(3):
            if Bits[i][j+1] == "1":
                Sym.append(1)
            else:
                Sym.append(-1)
        Symbols.append(Sym)
    # print(Symbols) 
    # Get all the deltas
    deltas = []
    for i in range(N):
        deltas.append(findDeltaBPSK(recieved, Symbols ,i, c))

    # Using the deltas, work backwards and determine the transmitted sequence
    transmitted = []
    for i in range(N):
        cost = min(deltas[N-1-i])
        bit = deltas[N-1-i].index(cost)
            
        # print(bit)
        if bit % 2 == 0:
            transmitted.append(-1)
        else:
            transmitted.append(1) 
    transmitted = Reverse(transmitted)
    return transmitted

def MLSE_4QAM(recieved, N, c):
    # Generate the bits, then get their symbols from the constellation map
    Bits = []
    for i in range(64):
        Bits.append(format(64+i, 'b'))    
    Symbols = []
    i = 0
    while i < 64:
        Sym = []
        for j in range(6):
            if Bits[i][j+1:j+3] == "00":
                Sym.append((1+1j)/sqrt2)
            elif Bits[i][j+1:j+3] == "01":
                Sym.append((-1+1j)/sqrt2)
            elif Bits[i][j+1:j+3] == "11":
                Sym.append((-1-1j)/sqrt2)
            elif Bits[i][j+1:j+3] == "10":
                Sym.append((1-1j)/sqrt2)
        Symbols.append(Sym)
        i += 2
    
    # Get all the deltas
    deltas = []
    for i in range(N):
        deltas.append(findDelta4QAM(recieved, Symbols ,i, c))
     
    # print(deltas)
    # Using the deltas, work backwards and determine the transmitted sequence
    transmitted = []
    for i in range(N):
        cost = min(deltas[N-1-i])
        bit = deltas[N-1-i].index(cost)
        
        if bit % 4 == 0: #recieved (1+1j)/sqrt2
            transmitted.append((1+1j)/sqrt2)
        elif bit % 4 == 1: #recieved (-1+1j)/sqrt2
            transmitted.append((-1+1j)/sqrt2)
        elif bit % 4 == 2: #recieved (-1-1j)/sqrt2
            transmitted.append((-1-1j)/sqrt2)
        elif bit % 4 == 3: #recieved (1-1j)/sqrt2
            transmitted.append((1-1j)/sqrt2)
            
    transmitted = Reverse(transmitted)
    
    return transmitted

def MLSE_8PSK(recieved, N, c):
    # Generate the bits, then get their symbols from the constellation map 
    L=3
    Bits = []
    
    for i in range(512):
        Bits.append(format(512+i, 'b'))    
    Symbols = []
    i = 0
    Sym = []
    while i < 8**L:
        Sym = []
        for j in range(9):
            if Bits[i][j+1:j+4]  == "111":
                Sym.append(1)
            elif Bits[i][j+1:j+4] == "110":
                Sym.append((1+1j)/sqrt2)
            elif Bits[i][j+1:j+4] == "010":
                Sym.append(1j)
            elif Bits[i][j+1:j+4] == "011":
                Sym.append((-1+1j)/sqrt2)
            elif Bits[i][j+1:j+4] == "001":
                Sym.append(-1)
            elif Bits[i][j+1:j+4] == "000":
                Sym.append((-1-1j)/sqrt2)
            elif Bits[i][j+1:j+4] == "100":
                Sym.append(-1j)
            elif Bits[i][j+1:j+4] == "101":
                Sym.append((1-1j)/sqrt2)
        Symbols.append(Sym)
        i += 3
        
    
    # Get all the deltas
    deltas = []
    for i in range(N):
        deltas.append(findDelta8PSK(recieved, Symbols ,i, c))
     
    # print(deltas)
    # Using the deltas, work backwards and determine the transmitted sequence
    transmitted = []
    for i in range(N):
        cost = min(deltas[N-1-i])
        bit = deltas[N-1-i].index(cost)            
            
        if bit % 8 == 0:
            transmitted.append(1)
        elif bit % 8 == 1:
            transmitted.append((1+1j)/sqrt2)
        elif bit % 8 == 2:
            transmitted.append(1j)
        elif bit % 8 == 3:
            transmitted.append((-1+1j)/sqrt2)
        elif bit % 8 == 4:
            transmitted.append(-1)
        elif bit % 8 == 5:
            transmitted.append((-1-1j)/sqrt2)
        elif bit % 8 == 6:
            transmitted.append(-1j)
        elif bit % 8 == 7:
            transmitted.append((1-1j)/sqrt2)
     
    transmitted = Reverse(transmitted)   
    return transmitted

def Graph():
    size=N
    size2=5
    randomValues= theorwhichman(size)
    bits=bits_gen(randomValues)
    BPSK_bits=BPSK(bits)
    FourQAM_bits=fourQAM(bits)
    EBPSK_bits=eight_PSK(bits)
    channels = [0.89+0.92j,0.42-0.37j,0.19+0.12j]
    transmitted=[]
    xValues = np.linspace(-4, 15, 38*size2)
    yvalues=[]
    """#bpsk
    
    L=3
    M=2
    a,transmitted=OptMemGen(1,3)
    transmitted.extend(BPSK_bits)
    k=-4
    while (k<15):
        
        for i in range(size2):
            sigma_=1/np.sqrt(math.pow(10, (k/ 10)) * 2 * math.log2(M))
            c = [(random.gauss(0,sigma_)+random.gauss(0,sigma_)*1j)/np.sqrt(6),
                 (random.gauss(0,sigma_)+random.gauss(0,sigma_)*1j)/np.sqrt(6),
                 (random.gauss(0,sigma_)+random.gauss(0,sigma_)*1j)/np.sqrt(6)]
            Recieved=addnoise(transmitted,channels,L,M,k,sigma_)#[1.5,1.2,1,-1.2,-1.5,0.2]
            Options,memory= OptMemGen(1,L)#bpsk 1, 4Qam,8psk
            Detected=BPSK_MLSE(Recieved,size,c)
            yvalues.append(bit_errors(transmitted[L-1:],Detected))
        k+= 0.5
    plt.semilogy(xValues,yvalues, label="BPSK")
    plt.ylabel('BER')
    plt.xlabel('SNR')
    
    
    yvalues=[]
     
    #4Qam
    L=3
    M=4
    a,transmitted=OptMemGen(2,3)
    transmitted.extend(FourQAM_bits)
    k=-4
    while (k<15):
        for i in range(size2):
            sigma_=1/np.sqrt(math.pow(10, (k/ 10)) * 2 * math.log2(M))
            c = [(random.gauss(0,sigma_)+random.gauss(0,sigma_)*1j)/np.sqrt(6),
                 (random.gauss(0,sigma_)+random.gauss(0,sigma_)*1j)/np.sqrt(6),
                 (random.gauss(0,sigma_)+random.gauss(0,sigma_)*1j)/np.sqrt(6)]
            Recieved=addnoise(transmitted,channels,L,M,k,sigma_)#[1.5,1.2,1,-1.2,-1.5,0.2]
            Options,memory= OptMemGen(2,L)#bpsk 1, 4Qam,8psk
            Detected=MLSE_4QAM(Recieved,int(size/2),c)
            yvalues.append(bit_errors(transmitted[L-1:],Detected))
        k+=0.5
    
    
    plt.semilogy(xValues,yvalues, label="4QAM")
    plt.ylabel('BER')
    plt.xlabel('SNR')
    yvalues=[]
    """
    L=3
    M=8 #BPSK=2 4Qam=4 8psk=8
    #8psk
    a,transmitted=OptMemGen(3,3)
    transmitted.extend(EBPSK_bits)
    
    k=-4
    while (k<15):
        for i in range(size2):
            sigma_=1/np.sqrt(math.pow(10, (k/ 10)) * 2 * math.log2(M))
            c = [(random.gauss(0,sigma_)+random.gauss(0,sigma_)*1j)/np.sqrt(6),
                 (random.gauss(0,sigma_)+random.gauss(0,sigma_)*1j)/np.sqrt(6),
                 (random.gauss(0,sigma_)+random.gauss(0,sigma_)*1j)/np.sqrt(6)]
            Recieved=addnoise(transmitted,channels,L,M,k, sigma_)#[1.5,1.2,1,-1.2,-1.5,0.2]
            Options,memory= OptMemGen(3,L)#bpsk 1, 4Qam,8psk
            Detected=MLSE_8PSK(Recieved,int(size/3),c)
            yvalues.append(bit_errors(transmitted[L-1:],Detected))
        k+=0.5
        
    plt.semilogy(xValues,yvalues, label="8PSK")
    plt.ylabel('BER')
    plt.xlabel('SNR')
    plt.title(" BER vs SNR")
    plt.legend()
# print(BPSK_MLSE(r,N))
#
#print(MLSE_4QAM(r,N))
#
Graph()







































