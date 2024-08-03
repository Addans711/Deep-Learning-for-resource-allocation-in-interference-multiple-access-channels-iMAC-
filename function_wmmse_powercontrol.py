import numpy as np
import math
import time
import scipy.io as sio
import matplotlib.pyplot as plt
import seaborn as sns
import mat73

# Functions for objective (sum-rate) calculation

def obj_IA_sum_rate(H, p, var_noise, N, K):
    y = 0.0
    for n in range(N):
        #The current user i belongs to cell n
        for i in range(K):
            # 
            idx_user = n * K + i  #The global index of the current user
            s = var_noise  
            for m in range(N):
                for j in range(K):
                    idx_other_user = m * K + j  # Global index of other users
                    if idx_other_user != idx_user:
                        s += H[idx_user, idx_other_user]**2 * p[idx_other_user]  #Cumulative interference power 
            y += math.log2(1 + H[idx_user, idx_user]**2 * p[idx_user] / s)  # Calculate the rate and add it up to the total rate
    return y

# Functions for WMMSE algorithm
def WMMSE_sum_rate(p_int, H, Pmax, var_noise):
    K = np.size(p_int)
    vnew = 0
    b = np.sqrt(p_int)
    f = np.zeros(K)
    w = np.zeros(K)
    for i in range(K):
        f[i] = H[i, i] * b[i] / (np.square(H[i, :]) @ np.square(b) + var_noise)
        w[i] = 1 / (1 - f[i] * b[i] * H[i, i])
        vnew = vnew + math.log2(w[i])

    VV = np.zeros(100)
    for iter in range(100):
        vold = vnew
        for i in range(K):
            btmp = w[i] * f[i] * H[i, i] / sum(w * np.square(f) * np.square(H[:, i]))
            b[i] = min(btmp, np.sqrt(Pmax)) + max(btmp, 0) - btmp

        vnew = 0
        for i in range(K):
            f[i] = H[i, i] * b[i] / ((np.square(H[i, :])) @ (np.square(b)) + var_noise)
            w[i] = 1 / (1 - f[i] * b[i] * H[i, i])
            vnew = vnew + math.log2(w[i])

        VV[iter] = vnew
        if vnew - vold <= 1e-3:
            break

    p_opt = np.square(b)
    return p_opt

# Functions for performance evaluation
def perf_eval(H, Py_p, NN_p, N,K,var_noise=1):
    num_sample = H.shape[2]
    pyrate = np.zeros(num_sample)
    nnrate = np.zeros(num_sample)
    mprate = np.zeros(num_sample)
    rdrate = np.zeros(num_sample)
    for i in range(num_sample):
        pyrate[i] = obj_IA_sum_rate(H[:, :, i], Py_p[:, i], var_noise, N,K)
        nnrate[i] = obj_IA_sum_rate(H[:, :, i], NN_p[i, :], var_noise, N,K)
        mprate[i] = obj_IA_sum_rate(H[:, :, i], np.ones(N*K), var_noise, N,K)
        rdrate[i] = obj_IA_sum_rate(H[:, :, i], np.random.rand(N*K,1), var_noise, N,K)
    print('Sum-rate: WMMSE: %0.3f, DNN: %0.3f, Max Power: %0.3f, Random Power: %0.3f'%(sum(pyrate)/num_sample, sum(nnrate)/num_sample, sum(mprate)/num_sample, sum(rdrate)/num_sample))
    print('Ratio: DNN: %0.3f%%, Max Power: %0.3f%%, Random Power: %0.3f%%\n' % (sum(nnrate) / sum(pyrate)* 100, sum(mprate) / sum(pyrate) * 100, sum(rdrate) / sum(pyrate) * 100))

    plt.figure('%d'%K)
    sns.set_style('dark')
    data = np.vstack([pyrate, nnrate]).T
    bins = np.linspace(0, max(pyrate), 50)
    plt.hist(data, bins, alpha=0.7, label=['WMMSE', 'DNN'])
    plt.legend(loc='upper right')
    plt.xlim([0, 40])
    plt.xlabel('sum-rate')
    plt.ylabel('number of samples')
    plt.savefig('Histogram_%d.eps'%K, format='eps', dpi=1000)
    plt.show()
    return 0




# Functions for data generation, IMAC case
import numpy as np
import time
import scipy.io as sio

def generate_IMAC(num_BS, num_User, num_H, Pmax=1, var_noise = 1):
    # Load Channel Data
    CH = sio.loadmat('IMAC_%d_%d_%d' % (num_BS, num_User, num_H))['X']
    Temp = np.reshape(CH, (num_BS, num_User * num_BS, num_H), order="F")
    H = np.zeros((num_User * num_BS, num_User * num_BS, num_H))
    for iter in range(num_BS):
        H[iter * num_User:(iter + 1) * num_User, :, :] = Temp[iter, :, :]

    # Compute WMMSE output
    Y = np.zeros((num_User * num_BS, num_H))
    Pini = Pmax * np.ones(num_User * num_BS)
    start_time = time.time()
    for loop in range(num_H):
        Y[:, loop] = WMMSE_sum_rate(Pini, H[:, :, loop], Pmax, var_noise)
    wmmsetime=(time.time() - start_time)
    H = np.reshape(H,(num_User * num_BS*num_User * num_BS,num_H), order="F")
    # print("wmmse time: %0.2f s" % wmmsetime)
    return H, Y, wmmsetime

def generate_IMAC_tst(num_BS, num_User, num_H,R,minR_ratio, Pmax=1, var_noise = 1):
    # Load Channel Data
    
    H = sio.loadmat('IMAC_%d_%d_%d_%d_%d' % (num_BS, num_User, num_H,R,minR_ratio))['H']
    Y = sio.loadmat('IMAC_%d_%d_%d_%d_%d' % (num_BS, num_User, num_H,R,minR_ratio))['Y']
    wmmsetime = sio.loadmat('IMAC_%d_%d_%d_%d_%d' % (num_BS, num_User, num_H,R,minR_ratio))['WmmseTime']
    return H, Y, wmmsetime

def generate_IMAC_tst1(num_BS, num_User, num_H, Pmax=1, var_noise = 1):
    # Load Channel Data
    
    H = sio.loadmat('IMAC_%d_%d_%d' % (num_BS, num_User, num_H))['H']
    Y = sio.loadmat('IMAC_%d_%d_%d' % (num_BS, num_User, num_H))['Y']
    wmmsetime = sio.loadmat('IMAC_%d_%d_%d' % (num_BS, num_User, num_H))['WmmseTime']
    return H, Y, wmmsetime
  






