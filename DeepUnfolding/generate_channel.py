import numpy.matlib
import numpy as np
import pickle 
import msgpack
import datetime


User = 30
Layer= 5
Nt = 64
Nr = 2
d = 2
snr_dB = 20; #in dB
Pt = 100
sigma = 1
number_of_sample = 10**(4)
starttime = datetime.datetime.now()

fw = open('data_channel\\data_channel.pkl','wb') 
for i in range(number_of_sample):
    H=[]
    for k in range(User):
        H.append( np.matrix( np.random.randn(Nr,Nt) + np.random.randn(Nr,Nt)*1j,dtype=np.complex ) ) 
        
    pickle.dump(H, fw)     
fw.close() 

endtime = datetime.datetime.now()


