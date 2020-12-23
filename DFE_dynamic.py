# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 10:52:03 2020

@author: user
"""


# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 22:11:23 2020

@author: user
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



"""ax = sns.distplot(a,
                  bins=100,
                  kde=True,
                  color='skyblue',
                  hist_kws={"linewidth": 15,'alpha':1})
ax.set(xlabel='Normal Distribution', ylabel='Frequency')
#[Text(0,0.5,u'Frequency'), Text(0.5,0,u'Normal Distribution')]
"""
#plt.plot(a, norm(0.5, 1).pdf(a))
#plt.ylabel('Probability Density')
#plt.xlabel('Randomly Generated Numbers')
#plt.show()

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

def addnoise(transmitted,channels,L,m,snr):# assuming transmited comes with the memory symbols padded
    recieved=[]
    M=m
    k=snr
    
    sigma_=1/np.sqrt(math.pow(10, (k/ 10)) * 2 * math.log2(M))
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
    #print(recieved)
    for i in range(0,len(recieved)):
        guess=[]
        n=len(channels)-1# length of the chanel L-1
        #calculating the product but from second position
        sumof=0
        for j in range(1,n):
           sumof+= symbols[n-1+s]*channels[j]
           n-=1
           
        for k in Options:
            #guess.append(np.abs(recieved[i]-((k)*channels[0]+symbols[1]*channels[1]+symbols[0]*channels[2]))**2)
            guess.append(np.abs(recieved[i]-((k)*channels[0]+sumof))**2)
        #print(guess[0])
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

def yvalueCal(transmitted):
    channels = [0.89+0.92j,0.42-0.37j,0.19+0.12j]
    L=3
    M=2 #for bpsk=2
    #SNR=15
    yvalues=[]
    #print(transmitted)
    
    for k in range (-4,16):
        
       
        Recieved=addnoise(transmitted,channels,L,M,k)#[1.5,1.2,1,-1.2,-1.5,0.2]
        #print(Recieved)
        Options,memory= OptMemGen(1,L)
        Detected=DFE(Recieved,channels,L,Options,memory)
        yvalues.append(bit_errors(transmitted[L-1:],Detected))
    return yvalues
#print(yvalueCal(transmitted))
 
def newchannel(v1,v2,v3):
    c=[]
    b=[v1+v2*1j,(v2+v3*1j),(v3+v1*1j)]/np.sqrt(2.3)
    c.extend(b)
    return c

def grapghs():
    size=1000000
    randomValues= theorwhichman(size)
    bits=bits_gen(randomValues)
    BPSK_bits=BPSK(bits)
    FourQAM_bits=fourQAM(bits)
    EBPSK_bits=eight_PSK(bits)
    channels = [0.89+0.92j,0.42-0.37j,0.19+0.12j]
    
    transmitted=[]
    xValues = np.linspace(-4, 15, 38)
    yvalues=[]
    #bpsk
    
    L=3
    M=2
    blocks=[BPSK_bits[n:n + 200] for n in range(0, len(BPSK_bits), 200)]
    a,transmitted=OptMemGen(1,3)
    #transmitted.extend(BPSK_bits)
    a,transmitter=OptMemGen(1,3)
    transmitter.extend(BPSK_bits)
  
    k=-4
    while (k<15):
        Recieved=[]
        Detected=[]
        
        for block in blocks:
            v1=np.random.normal(0,1,1)[0]
            v2= np.random.normal(0,1,1)[0]
            v3 =np.random.normal(0,1,1)[0]
            channels= newchannel(v1,v2,v3)
            a,transmitted=OptMemGen(1,3)
            transmitted.extend(block)
            
            Recieved=(addnoise(transmitted,channels,L,M,k))
            Options,memory= OptMemGen(1,L)#bpsk 1, 4Qam,8psk
            Detected.extend(DFE(Recieved,channels,L,Options,memory))
        
        yvalues.append(bit_errors(transmitter[L-1:],Detected))
        k+=0.5
    plt.semilogy(xValues,yvalues, label="BPSK")
    plt.ylabel('BER')
    plt.xlabel('SNR')
    
    
    yvalues=[]
    #4Qam
    L=3
    M=4
    blocks=[FourQAM_bits[n:n + 200] for n in range(0, len(FourQAM_bits), 200)]
    a,transmitter=OptMemGen(2,3)
    transmitter.extend(FourQAM_bits)
  
    k=-4
    while (k<15):
        Recieved=[]
        Detected=[]
        
        for block in blocks:
            v1=np.random.normal(0,1,1)[0]
            v2= np.random.normal(0,1,1)[0]
            v3 =np.random.normal(0,1,1)[0]
            channels= newchannel(v1,v2,v3)
            a,transmitted=OptMemGen(2,3)
            transmitted.extend(block)
            
            Recieved=(addnoise(transmitted,channels,L,M,k))
            Options,memory= OptMemGen(2,L)#bpsk 1, 4Qam,8psk
            Detected.extend(DFE(Recieved,channels,L,Options,memory))
        
        yvalues.append(bit_errors(transmitter[L-1:],Detected))
        k+=0.5
    plt.semilogy(xValues,yvalues, label="4QAM")
    plt.ylabel('BER')
    plt.xlabel('SNR')
    
    yvalues=[]
    
    L=3
    M=8 #BPSK=2 4Qam=4 8psk=8
    #8psk
    blocks=[EBPSK_bits[n:n + 200] for n in range(0, len(EBPSK_bits), 200)]
    a,transmitter=OptMemGen(3,3)
    transmitter.extend(EBPSK_bits)
  
    k=-4
    while (k<15):
        Recieved=[]
        Detected=[]
        
        for block in blocks:
            v1=np.random.normal(0,1,1)[0]
            v2= np.random.normal(0,1,1)[0]
            v3 =np.random.normal(0,1,1)[0]
            channels= newchannel(v1,v2,v3)
            a,transmitted=OptMemGen(3,3)
            transmitted.extend(block)
            
            Recieved=(addnoise(transmitted,channels,L,M,k))
            Options,memory= OptMemGen(3,L)#bpsk 1, 4Qam,8psk
            Detected.extend(DFE(Recieved,channels,L,Options,memory))
        
        yvalues.append(bit_errors(transmitter[L-1:],Detected))
        k+=0.5
    plt.semilogy(xValues,yvalues, label="8PSK")
    plt.ylabel('BER')
    plt.xlabel('SNR')
    plt.title(" BER vs SNR")
    plt.legend()

grapghs()



def tester():
    size=10000
    #np.random.seed(420)
    print("Generating the random number generator of size 200")
    randomValues= theorwhichman(size)
    
    
    print("\nExpected sigma =0.29 and expected mu=0.50")
    print("Sigma",st.mean(randomValues))
    print("mu",st.stdev(randomValues))

    print("\nTesting the bits_gen function of size 200")
    bits=bits_gen(randomValues)
    print("Size:",len(bits))
    
    print("\nNow testing the mapping of sysmbols for different modulation schemes")
    print("BPSK expected length 200")
    BPSK_bits=BPSK(bits)
    print("BPSK length:",len(BPSK_bits))
    print("4QAM expected length =100")
    #FourQAM_bits=fourQAM(bits)
    #print("4QAM length:",len(FourQAM_bits))
    print("8BPSK expected length 66")
    #EBPSK_bits=eight_PSK(bits)
    #print("8BPSK length:",len(EBPSK_bits))
    #SNR = np.linspace(0, 15, 16)
    #print(SNR)
    #print("\nGenerating noise for 8psk")
    #noiseList=noise(len(EBPSK_bits),sigma(SNR,8)[0])
    #print(noiseList)
    #print("Length of noise:",len(noiseList))
    #print("variance:" ,st.variance(noiseList))
    
    print("\nTesting the DFE function")
    channels = [0.89+0.92j,0.42-0.37j,0.19+0.12j]
    #bpsk
    transmitted=[1,1]
    transmitted.extend(BPSK_bits)
    #QPSK
    
    L=3
    M=2 #for bpsk=2
    SNR=-1
    
    print("Adding Noise")
    #Recieved=addnoise(transmitted,channels,L,M,SNR)#[1.5,1.2,1,-1.2,-1.5,0.2]
    #Options,memory= OptMemGen(1,L)
    #print("Received symbols:",Recieved)
    #Detected=DFE(Recieved,channels,L,Options,memory)
    #print("Detected Symbols",Detected)
    xValues = np.linspace(-4, 15, 20)
    yvalues=yvalueCal(transmitted)
    print(yvalues)
    plt.semilogy(xValues,yvalues)
   
    

#tester()
    
