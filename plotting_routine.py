import numpy as np
import matplotlib.pyplot as plt
 
# default setting
plt.rcParams['figure.figsize'] = [7, 5.25]
 
# This plots the N(z)
# Here data is a 1D numpy array with spec-z (or any redshift data)
def plotN(z_data):
    plt.rc('font', size=15)
    reds=np.arange(0,4,.1)
    n, edges, patches = plt.hist(z_data, bins=reds, density=False)
    plt.xlabel("redshift")
    plt.ylabel("number")
    return 
 
 
 
# For the rest of the routines the input is a numpy array with the spec-z in columm 8, the phot-z in column 6,
# the actual CO flag in column 7, and the prediction from the binary classifier in column 10. Change the indexes to
# match what your array columns are

# col-n
# 6: phot-z
# 7: CO flag
# 8: spec-z
# 9: 
# 10: prediction from the binary classifier
 
# This plots a spec-z vs phot-z scatter plot with the lines
def plotpzsz(specz_data, photz_data, alpha=1):
    plt.rc('font', size=15)

    plt.scatter(specz_data, photz_data, s=0.1, alpha=alpha)
    
    plt.xlabel("spectroscopic z")
    plt.ylabel("photo z")
    plt.xticks(np.arange(0, 4.1, step=1)) # 4.1 since it doesnt include 4 when it's 4.0 for some reason
    plt.yticks(np.arange(0, 4.1, step=1)) # 4.1 since it doesnt include 4 when it's 4.0 for some reason
    plt.xlim(-.2,4.1)
    plt.ylim(-.2,4.1)
    
    ax = plt.gca()

    reds=np.arange(-.2, 5,.1)
    kwargs={'linestyle':'--'}
    
    plt.plot(reds, reds+0.15*(1+reds), color='black', **kwargs)
    plt.plot(reds, reds-0.15*(1+reds), color='black', **kwargs)
    plt.plot(reds, reds+1, color='black')
    plt.plot(reds, reds-1, color='black')
    
    plt.plot()
    return

# expected structure of data, the paramter
# data = np.array([
#     np.array(results['Phot z']),
#     np.array(results['CO?']),
#     np.array(results['Spec z']),
#     np.array(results['Predicted CO?']) # predicted CO value (either 0 or 1)
# ])
# CO_threshold is a value in [0, 1.0]
# if predicted CO value is > CO_threshold, the obejct is considered as CO
def plotvsz(data, CO_threshold, evaluation_ratio = 1.0):
    assert (0 <= evaluation_ratio) and (evaluation_ratio <= 1.0), "evaluation_ratio should be in [0, 1]"

    plt.rc('font', size=15)
    
    # Plots
    
    # percent of NCO or CO
    pctnosarr=np.zeros(40)
    pctcosarr=np.zeros(40)
    
    # Percent of bad NCO or CO
    pctbadnosarr=np.zeros(40)
    pctgoodcosarr=np.zeros(40)
    
    # Number of CO or NCO
    numcosarr=np.zeros(40)
    numnosarr=np.zeros(40)
    
    eval_indices = np.random.choice(data.shape[0], int(evaluation_ratio * len(data)), replace = False)
    data = data[eval_indices]

    z_range = np.arange(0,4,.1) # returns evenly spaced values within the interval
    deek=0 #1 for PZ
    nutz = 2
    if deek == 1:
            nutz = 0
    for n, i in enumerate(z_range):
        bins = np.asarray(np.where((data[:,nutz] >= i) & (data[:,nutz] < i+0.1))) # select data that belong in a bin
        num=bins.size
        
        # 1. NO
        nos=np.asarray(np.where( abs(data[bins[0,:], 2] - data[bins[0,:], 0])/(1+data[bins[0,:], 2]) < 0.15  ))
        numnos=nos.size
        
        badnos=np.asarray(np.where(data[bins[0,nos[0,:]], 3] > CO_threshold))
        numbadnos=badnos.size

#         print(nos, numnos, '\n', badnos, numbadnos)
        if (numnos != 0):
            pctbadnos=float(numbadnos)/float(numnos)
        else:
            pctbadnos=0
        if (num != 0):
            pctnos=float(numnos)/float(num)
        else:
            pctnos=50
        
#         print(i)
#         print(f'Percent bad NOs: {pctbadnos: .3f}\nPercent NOs{pctnos: .3f}')
        pctnosarr[n]=pctnos
        pctbadnosarr[n]=pctbadnos
        numnosarr[n]=numnos
        
        # 2. CO
        cos=np.asarray(np.where( (data[bins[0,:],1] == 1)))
        numcos=cos.size
        
        goodcos=np.array(np.where(data[bins[0,cos[0,:]], 3] > CO_threshold))
        numgoodcos=goodcos.size
        
        if (numcos != 0):
            pctgoodcos=float(numgoodcos)/float(numcos)
        else:
            pctgoodcos=0
        if (num != 0):
            pctcos=float(numcos)/float(num)
        else:
            pctcos=50
        
#         print(f'Percent good COs: {pctgoodcos: .3f}\nPercent COs{pctcos: .3f}')
#         print()
        pctcosarr[n]=pctcos
        pctgoodcosarr[n]=pctgoodcos
        numcosarr[n]=numcos
        
#     print('good CO arr\n', pctgoodcosarr)
#     print('bad NO arr\n', pctbadnosarr)
    #print(pctnosarr)
    #print(pctcosarr)
    #print(z_range)
    plt.scatter(z_range, pctgoodcosarr)
    plt.scatter(z_range, pctbadnosarr, marker='x')
    plt.xticks(np.arange(0, 4.1, step=1)) # 4.1 since it doesnt include 4 when it's 4.0 for some reason
    plt.xlabel("spectroscopic z")
    plt.ylabel("fraction")
    plt.ylim(-.1,1.1)
   
    
    fig, ax1 = plt.subplots()
    plt.rcParams['figure.figsize'] = [7, 5.25]
    ax1.set_xlabel("spectroscopic z")
    plt.xticks(np.arange(0, 4.1, step=1)) # 4.1 since it doesnt include 4 when it's 4.0 for some reason
    ax1.set_ylabel('# COs')
    ax1.plot(z_range, numcosarr, color = 'blue')
    ax2 = ax1.twinx()
    ax2.set_ylabel('# NOs')
    ax2.plot(z_range, numnosarr, color = 'orange')
    plt.show()
   
    #plt.scatter(z_range,numcosarr)
    #plt.scatter(z_range,numnosarr)
   
    text = ''
    
    # Print fractions
    
    #  Fraction of incorrectly identified NOs > 2
    bins = np.asarray(np.where((data[:,nutz] >= 2)))
    nos=np.asarray(np.where( abs(data[bins[0,:],2] - data[bins[0,:],0])/(1+data[bins[0,:],nutz]) < 0.15  ))
    badnos=np.array(np.where(data[bins[0,nos[0,:]], 3] > CO_threshold))
    numnos=nos.size
    numbadnos=badnos.size
    frc=numbadnos/numnos
    print(f'flagged nos > 2:\t {frc*100:.4f}%')
    text += f'\nflagged nos > 2: {frc*100:.4f}%'
    
    # Fraction of incorrectly identified NOs > 1
    bins=np.asarray(np.where((data[:,nutz] >= 1)))
    nos=np.asarray(np.where( abs(data[bins[0,:],2] - data[bins[0,:],0])/(1+data[bins[0,:],nutz]) < 0.15  ))
    badnos=np.array(np.where(data[bins[0,nos[0,:]], 3] > CO_threshold))
    numnos=nos.size
    numbadnos=badnos.size
    frc=numbadnos/numnos
    print(f'flagged nos > 1:\t {frc*100:.4f}%')
    text += f'\nflagged nos > 1: {frc*100:.4f}%'
    
    # Fraction of incorrectly identified NOs
    nos=np.asarray(np.where( abs(data[:,2] - data[:,0])/(1+data[:,2]) < 0.15  )).squeeze()
    badnos=np.array(np.where(data[nos, 3] > CO_threshold))
    numnos=nos.size
    numbadnos=badnos.size
    frc=numbadnos/numnos
    print(f'flagged nos:\t\t {frc*100:.4f}%')
    text += f'\nflagged nos: {frc*100:.4f}%'
    
    s=len(data[:,2])
    
    cos=np.asarray(np.where( (data[:,1] == 1 ))).squeeze()
    goodcos=np.array(np.where(data[cos, 3] > CO_threshold))
    
    numcos=cos.size
    numgoodcos=goodcos.size
    
    frc=numgoodcos/numcos
    print(f'flagged COs:\t\t {frc*100:.4f}%')
    text += f'\nflagged COs: {frc*100:.4f}%'
    
    frc=numnos/s
    print(f'NOs:\t\t\t {frc*100:.4f}%')
    text += f'\nNOs: {frc*100:.4f}%'
    
    frc=numcos/s
    print(f'COs:\t\t\t {frc*100:.4f}%')
    text += f'\nCOs: {frc*100:.4f}%'
    
    return text