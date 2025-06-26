# -*- coding: utf-8 -*-
"""Methods for Echo-Planar Imaging (EPI) acquisition
with an focus on diffusion tensor/kurtosis imaging.

Author:
    Zhengguo Tan <zhengguo.tan@gmail.com>
    Julius Glaser <julius-glaser@gmx.de>
"""
import numpy as np
from sigpy import fourier, util, block
import sigpy as sp
from sigpy.mri import sms, app
import h5py

MIN_POSITIVE_SIGNAL = 0.0001


def phase_corr(kdat, pcor, topup_dim=-11, splitted=False):
    """perform phase correction.

    Args:
        kdat (array): k-space data to be corrected
        pcor (array): phase-correction reference data
        topup_dim (int): dimension of the top-up and top-down measurements

    Output:
        phase-corrected k-space data

    Reference:
        Ehses P. https://github.com/pehses/twixtools
        Ahn, Cho A New Phase Correction Method in NMR Imaging Based on Autocorrelation and Histogram Analysis
    """
    col_dim = -1

    ncol = pcor.shape[col_dim]
    npcl = pcor.shape[topup_dim]

    # if three lines are present,
    # average the 1st and the 3rd line.
    # "Robust EPI Phase Correction O. Heid"
    if npcl == 3:
        pcors = np.swapaxes(pcor, 0, topup_dim)
        pcor1 = pcors[[0, 2], ...]
        pcor_odd = np.mean(pcor1, axis=0, keepdims=True)
        pcor_eve = pcors[[1], ...]
        pcor = np.concatenate((pcor_odd, pcor_eve))
        pcor = np.swapaxes(pcor, 0, topup_dim)

    oshape = list(kdat.shape)
    oshape[topup_dim] = 1

    output = np.zeros(oshape)

    pcor_img = fourier.ifft(pcor, axes=[col_dim])
    kdat_img = fourier.ifft(kdat, axes=[col_dim])

    slope = np.angle((np.conj(pcor_img[..., 1:]) * pcor_img[..., :-1])
                     .sum(col_dim, keepdims=True).sum(-2, keepdims=True))
    x = np.arange(ncol) - ncol//2

    pcor_fac = np.exp(1j * slope * x)
    kdat_img *= pcor_fac
    if splitted:
        pass
    else:
        kdat_img = kdat_img.sum(topup_dim, keepdims=True)
    output = fourier.fft(kdat_img, axes=[-1])

    return output

def SAKE_ref_correction(kdat_ref, calib_shape, 
                        kSize=[3,3], 
                        SAKE_beta_rel=0.001,#600000
                        SAKE_p = 0.5, 
                        nIter=30, 
                        threshold_PSI_calc = 0.0015, 
                        radius_PSI_calc = 0.25):
    """perform phase correction using the SAKE method proposed by Lyu.

    Args:
        kdat_ref (array): k-space data of the already simply corrected references
        calib_shape (list or int): shape of the calibration area
        kSize (list): shape of the SAKE kernel
        SAKE_beta_rel (float):  TODO
        SAKE_p (float):        TODO
        nIter (int): number of iterations to run SAKE
        threshold_PSI_calc (float): threshold for masking of phase correlation matrix in Psi calculation in phase alignment
        radius_PSI_calc (float): frequency limit from k-space center to be used in estimation of Psi in phase alignment

    Output:
        phase-corrected, resized references in k-space

    Author:
        Julius Glaser <julius-glaser@gmx.de>

    Reference:
        Lyu, Barth, Xie, Liu, Ma, Deng, Wu
        Robust SENSE reconstruction of simultaneous multislice EPI with low-rank 
        enhanced coil sensitivity calibration and slice-dependent 
        2D Nyquist ghost correction
        DOI: https://doi.org/10.1002/mrm.27120
        Code: https://github.com/mylyu/SMS-EPI-Ghost-Correction

        TODO: multishot support
    """

    assert (len(kdat_ref.shape) == 4 or len(kdat_ref.shape) == 3), \
    "kdat_ref must be either 3 dimensional (Nx, Ny, Nch) or 4 dimensional (Nx, Ny, Nsli, Nch)"

    if len(kdat_ref.shape) == 4:
         Nx, Ny, Nsli, Ncha = kdat_ref.shape
    else:
         Nx, Ny, Ncha = kdat_ref.shape
         Nsli = 1

    if isinstance(calib_shape, list) and len(calib_shape)==2:
        n_calib_shape = calib_shape + [Ncha]
    elif isinstance(calib_shape, int):
        n_calib_shape = [calib_shape]*2 + [Ncha]
    else:
        raise TypeError("calib_width must be either a 2 entry list or an integer")

    calib_SAKE = []
    # run SAKE seperatly for each slice
    for i in range(Nsli):
        print('>> slice number ', i)
        if Nsli > 1:
            data = kdat_ref[:, :, i, :]
        else:
            data = kdat_ref

        # resize data to get only calib region in the center
        calib = util.resize(data, n_calib_shape)

        # join calib region twice as virtual channels and shift the higher channels to 
        # use the same mask for positive and negative echos
        calib_vc = np.concatenate((calib, np.roll(calib, shift=-1, axis=-2)), axis=-1)

        # mask for positive and negative echos is the same due to shifting before
        mask = np.zeros_like(calib_vc)
        mask[:,0::2,:] += 1


        res = SAKEwithInitialValue(calib_vc, mask, kSize, SAKE_beta_rel, SAKE_p, nIter)

        # reverse shifting from before
        res[:,:, Ncha:] = np.roll(res[:,:, Ncha:], shift=1, axis=-2)

        # add positive and negative estimated echos together without signal cancellation
        calib_SAKE.append(pos_neg_add(res[:,:, 0:Ncha], res[:,:, Ncha:], threshold_PSI_calc, radius_PSI_calc))

    # bring calculated references in shape Ncha, Nsli, Ny, Nx and zero fill
    calib_SAKE = np.array(calib_SAKE)
    calib_SAKE = np.transpose(calib_SAKE, (-1, 0, 2, 1))
    calib_SAKE_zf = util.resize(calib_SAKE, [Ncha, Nsli, Ny, Nx])

    return calib_SAKE_zf 

def SAKEwithInitialValue(DATA, mask, kSize, beta_relative=0.01, p=0.01, nIter=50):
    """perform SAKE for already initial corrected EPI data split in negative and positvie echos.
       This is effectively the python implementation of the matlab function SAKEwithInitialValue by Lyu

    Args:
        DATA (array): k-space data of the calibration area with pos and neg echos as virtual channels
        mask (array): mask of pos and neg echos (same mask for both as neg are shifted before)
        kSize (list): shape of the SAKE kernel
        beta (float): TODO
        p (float): TODO
        nIter (int): number of iterations to run SAKE

    Output:
        estimated and filled virtual channels of pos and neg echos

    Author:
        Julius Glaser <julius-glaser@gmx.de>

    Reference:
        Lyu, Barth, Xie, Liu, Ma, Deng, Wu
        Robust SENSE reconstruction of simultaneous multislice EPI with low-rank 
        enhanced coil sensitivity calibration and slice-dependent 
        2D Nyquist ghost correction
        DOI: https://doi.org/10.1002/mrm.27120

        Shin, Larson, Ohliger, Elad, Pauly, Vigneron, Lustig
        Calibrationless parallel imaging reconstruction based on structured 
        low-rank matrix completion
        DOI: https://doi.org/10.1002/mrm.24997

        Code: https://github.com/mylyu/SMS-EPI-Ghost-Correction

        TODO: multishot support
    """
    sx, sy, sc = DATA.shape
    res = DATA

    for i in range(nIter):
        tmp = block.array_to_blocks(res, kSize + [1],[1]*3)
        block_1, block_2, block_3, _,_,_ = tmp.shape
        tmp = np.reshape(tmp, (block_1*block_2, block_3, np.prod(kSize)), order='F')
        tmp = np.swapaxes(tmp, -2, -1)
        tsx, tsy, tsz = tmp.shape
        A = np.reshape(tmp, (tsx, tsy*tsz), order='F')

        # Perform SVD on calibration matrix and keep up to number specified by Schatten norm
        U, S, VH = np.linalg.svd(A, full_matrices=False)
        beta = S[0]*beta_relative
        S_new = S-S**(p-1)*beta         # times beta, because S[0] is really small, therefore beta_relative would need to be extremely large if divided by
        # print(S[0])
        # print(S[0]**(p-1)/beta)
        S_new = S_new[S_new>0]
        rank_new = len(S_new)
        S_keep = np.diag(S_new)
        
        print(rank_new)
        if rank_new < 15:
            raise ValueError(f"Rank must be greater than 15, was {rank_new}")
        A = U[:,0:rank_new] @ S_keep @ VH[0:rank_new, :]

        # Enforce Hankel structure
        A = np.reshape(A,(tsx,tsy,tsz), order='F')

        A = np.swapaxes(A, -2, -1)
        A = np.reshape(A, (block_1, block_2, block_3, kSize[0], kSize[1], 1), order='F')
        # # print(A.shape)
        tmp = block.blocks_to_array(A, (sx, sy, sc), kSize + [1],[1]*3)

        # calculate weight map for normalization as in matlab code
        ones_blocks = np.ones_like(A)
        weight_map = block.blocks_to_array(ones_blocks, (sx, sy, sc), kSize + [1],[1]*3)

        normalized_tmp = np.divide(
            tmp,
            weight_map,
            out=np.zeros_like(tmp),
            where=weight_map != 0
        )

        #IFFT of tmp
        # assume only one shot
        N_virtual_coils = sc

        img_1 = fourier.ifft(normalized_tmp[:,:,0:N_virtual_coils//2], axes=(0,1), norm=None)
        img_1 *= np.sqrt(sx*sy)
        img_2 = fourier.ifft(normalized_tmp[:,:,N_virtual_coils//2::], axes=(0,1), norm=None)
        img_2 *= np.sqrt(sx*sy)

        #mean of two virtual channels
        mean_abs = (abs(img_1) + abs(img_2))/2

        #FFT
        tmp = fourier.fft(np.concatenate((mean_abs*np.exp(1j*np.angle(img_1)), mean_abs*np.exp(1j*np.angle(img_2))), axis=-1), axes=(0,1))
        
        # enforce data consistency
        res = tmp*(1-mask) + DATA*mask
        
    return res

def pos_neg_add(a, b, threshold = 0.0015, radius = 0.25):
    """Add SAKE filled positive and negative echo images without signal cancellation.

    Args:
        a (array): positive echo image
        b (array): negative echo image
        threshold (float): threshold for masking of phase correlation matrix in Psi calculation
        radius (float): frequency limit from k-space center to be used in estimation of Psi

    Output:
        combined echo image

    Author:
        Julius Glaser <julius-glaser@gmx.de>

    Reference:
        Hoge, Kraft
        Robust EPI Nyquist ghost elimination via spatial and temporal encoding
        DOI: https://doi.org/10.1002/mrm.22564
    """

    # Calculation of Psi and application has to be done in image space 
    # (shift in k-space is linear phase in image space)
    coil_imgs_a = fourier.ifft(a, axes=(0,1))
    coil_imgs_b = fourier.ifft(b, axes=(0,1))

    out = np.zeros_like(a)
    Ncha = out.shape[-1]

    for iCha in range(Ncha):
        coil_img_a = coil_imgs_a[:,:,iCha]
        coil_img_b = coil_imgs_b[:,:,iCha]

        # calculate Psi for phase alignment as described in reference
        Psi = calc_phase_align_matrix(coil_img_a, coil_img_b, threshold, radius)

        # align phase of b to image a
        coil_img_b_corr = coil_img_b*Psi

        # combine images and backtransform
        out[:,:,iCha] = fourier.fft((coil_img_a + coil_img_b_corr), axes=(0,1))/2

    return out

def calc_PEM(refs_zf_split, coil, calib_width, sp_device=-1):

    device = sp.Device(sp_device)
    xp = device.xp

    Necho, Ncha, Nsli, Ny, Nx = refs_zf_split.shape
    MB = 1

    estimated_echo_images = np.zeros_like(refs_zf_split[:,0,...])

    print('> device: ', device)



    for s in range(Nsli):

        slice_str = str(s).zfill(3)
        print('> slice idx: ', slice_str)

        # map from collapsed slice index to interleaved uncollapsed slice list
        # slice_mb_idx = sms.get_uncollap_slice_idx(N_slices, MB, s)
        slice_mb_idx = sms.map_acquire_to_ordered_slice_idx(s, Nsli, MB)
        # slice_mb_idx = [46, 93]


        # read in k-space data
        echo_pos = refs_zf_split[0,:,slice_mb_idx,...].squeeze()
        
        echo_pos = echo_pos[:,np.newaxis, ...]
        echo_neg = refs_zf_split[1,:,slice_mb_idx,...].squeeze()
        echo_neg = echo_neg[:,np.newaxis, ...]
        print('>> slice_mb_idx: ', slice_mb_idx)

        coil2 = coil[:, slice_mb_idx, :, :]

        kdat_pos_echo = sp.to_device(echo_pos, device=device)
        kdat_neg_echo = sp.to_device(echo_neg, device=device)
        coil2 = sp.to_device(coil2, device=device)

        with device:
            img_shape = [1,1,coil2.shape[-2], coil2.shape[-1]]

            S = sp.linop.Multiply(img_shape, coil2)
            F = sp.linop.FFT(S.oshape, axes=[-2, -1])
            
            weights_pos = (sp.rss(kdat_pos_echo, axes=(0, ), keepdims=True) > 0).astype(echo_pos.dtype)
            weights_neg = (sp.rss(kdat_neg_echo, axes=(0, ), keepdims=True) > 0).astype(echo_neg.dtype)
            W_pos = sp.linop.Multiply(F.oshape, weights_pos)
            W_neg = sp.linop.Multiply(F.oshape, weights_neg)

            A_pos = W_pos * F * S
            A_neg = W_neg * F * S

            AHA_pos = lambda x: A_pos.N(x) + 0.01 * x
            AHA_neg = lambda x: A_neg.N(x) + 0.01 * x
            AHy_pos = A_pos.H(kdat_pos_echo)
            AHy_neg = A_neg.H(kdat_neg_echo)

            img_pos = xp.zeros(img_shape, dtype=kdat_pos_echo.dtype)
            img_neg = xp.zeros(img_shape, dtype=kdat_pos_echo.dtype)
            alg_method_pos = sp.alg.ConjugateGradient(AHA_pos, AHy_pos, img_pos, max_iter=30, verbose=False)
            alg_method_neg = sp.alg.ConjugateGradient(AHA_neg, AHy_neg, img_neg, max_iter=30, verbose=False)

            while (not alg_method_pos.done()) and (not alg_method_neg.done()):
                if (not alg_method_pos.done()):
                    alg_method_pos.update()
                if (not alg_method_neg.done()):
                    alg_method_neg.update()

            estimated_echo_images[0,slice_mb_idx,...] = img_pos
            estimated_echo_images[1,slice_mb_idx,...] = img_neg

    f = h5py.File('estimated_echos.h5', 'w')
    f.create_dataset('estimated_echos', data=estimated_echo_images)
    f.close()
    # pem_mps = np.zeros_like(estimated_echo_images)
    # for s in range(Nsli):
    #     print('  ' + str(s).zfill(3))

    #     c = app.EspiritCalib(sp.fft(estimated_echo_images[:, s, :, :],axes=(-2,-1)),
    #                         calib_width = calib_width,
    #                         crop=0.,
    #                         device=device, show_pbar=False).run()
    #     pem_mps[:,s,...] = sp.to_device(c)

    PEM_filtered = estimated_echo_images[0,...]*np.conj(estimated_echo_images[1,...])
    # PEM_filtered = PEM_filtered/abs(PEM_filtered)

    PEM_filtered = PEM_filtered[np.newaxis,...]
    # mps_reord = sms.reorder_slices_mb1(PEM_filtered, Nsli)
    return PEM_filtered

def SB_PEM_recon(kdat, coil, PEM, s, N_slices, sp_device = -1):  

    MB = 1

    kdat = np.squeeze(kdat)  # 5 dim
    kdat = np.swapaxes(kdat, -2, -3)
    Necho, Ndiff, Ncha, Ny, Nx = kdat.shape
    kdat = kdat[0,...] + kdat[1,...] # 4dim

    device = sp.Device(sp_device)

    print('> device: ', device)

    xp = device.xp

    slice_mb_idx = sms.map_acquire_to_ordered_slice_idx(s, N_slices, MB)

    coil2 = np.stack((coil[:, slice_mb_idx, :, :]*np.exp(1j*np.angle(PEM[:,slice_mb_idx,...]))/2, coil[:, slice_mb_idx, :, :]/2), axis=0)

    with device:
        img_shape = [1,1,1,coil2.shape[-2], coil2.shape[-1]]

        S = sp.linop.Multiply(img_shape, coil2)
        F = sp.linop.FFT(S.oshape, axes=[-2, -1])
        D = sp.linop.Sum(F.oshape, axes=(0, ), keepdims=False)
        print(img_shape)
        print(S.oshape)
        print(F.oshape)
        print(D.oshape)

        dwi_pi_comb = []
        for d in range(Ndiff):

            k = kdat[d,...]
            weights = (sp.rss(k[...], axes=(0, ), keepdims=True) > 0).astype(kdat.dtype)
            W = sp.linop.Multiply(D.oshape, weights)
            k = k[:,np.newaxis,...]
            print(W.oshape)
            print(k.shape)
            A = W * D * F * S

            AHA = lambda x: A.N(x) + 0.01 * x
            AHy = A.H(k)

            img = xp.zeros(img_shape, dtype=k.dtype)
            alg_method = sp.alg.ConjugateGradient(AHA, AHy, img, max_iter=30, verbose=False)

            while (not alg_method.done()):
                alg_method.update()

            dwi_pi_comb.append(sp.to_device(img))

        dwi_pi_comb = np.array(dwi_pi_comb)
        return dwi_pi_comb
    
def MB_PEM_recon(kdat, coil, PEM, s, N_slices, MB, pat, sp_device = -1):  

    

    kdat = np.squeeze(kdat)  # 5 dim
    kdat = np.swapaxes(kdat, -2, -3)
    Necho, Ndiff, Ncha, Ny, Nx = kdat.shape
    kdat = np.concatenate((kdat[0,...], kdat[1,...]), axis = 1)  # create virtual channels for pos and neg echos

    # number of collapsed slices
    N_slices_collap = N_slices // MB

    # SMS phase shift
    yshift = []
    for b in range(MB):
        yshift.append(b / pat)

    sms_phase = sms.get_sms_phase_shift([MB, Ny, Nx], MB=MB, yshift=yshift)

    device = sp.Device(sp_device)

    print('> device: ', device)

    xp = device.xp

    slice_mb_idx = sms.map_acquire_to_ordered_slice_idx(s, N_slices, MB)

    coil2 = np.concatenate((coil[:, slice_mb_idx, :, :]/2, coil[:, slice_mb_idx, :, :]/2), axis=0)

    with device:
        img_shape = [1,MB,coil2.shape[-2], coil2.shape[-1]]

        S = sp.linop.Multiply(img_shape, coil2)
        F = sp.linop.FFT(S.oshape, axes=[-2, -1])
        P = sp.linop.Multiply(F.oshape, sms_phase)
        M = sp.linop.Sum(P.oshape, axes=(-3, ), keepdims=True)

        dwi_pi_comb = []
        for d in range(Ndiff):

            k = kdat[d,...]
            weights = (sp.rss(k[...], axes=(0, ), keepdims=True) > 0).astype(kdat.dtype)
            W = sp.linop.Multiply(M.oshape, weights)
            k = k[:,np.newaxis,...]

            A = W * M * P * F * S

            AHA = lambda x: A.N(x) + 0.01 * x
            AHy = A.H(k)

            img = xp.zeros(img_shape, dtype=k.dtype)
            alg_method = sp.alg.ConjugateGradient(AHA, AHy, img, max_iter=100, verbose=False)

            while (not alg_method.done()):
                alg_method.update()

            dwi_pi_comb.append(sp.to_device(img))

        dwi_pi_comb = np.array(dwi_pi_comb)
        return dwi_pi_comb
    
def calc_phase_align_matrix(a, b, threshold = 0.0015, radius = 0.25):
    """Calculate PSI to align the phase of a and b such that no signal cancellation happens.

    Args:
        a (array): k-space data to be corrected
        b (array): phase-correction reference data
        threshold (float): threshold for masking of phase correlation matrix
        radius (float): frequency limit from k-space center to be used in estimation of Psi

    Output:
        Psi: noise free correlation matrix

    Author:
        Julius Glaser <julius-glaser@gmx.de>

    Reference:
        Hoge
        A subspace identification extension to the phase correlation method
        DOI: 10.1109/TMI.2002.808359
        Code: https://sigprocexperts.com/demo_code/sie_pcm/
    """

    assert len(a.shape) == 2, 'A should have a dimensionality of 2'
    assert len(b.shape) == 2, 'B should have a dimensionality of 2'
    assert radius < 0.5, f"radius can't be larger than 0.5, radius = {radius} "

    m,n = a.shape

    # calculate original phase correlation matrix (PCM)
    Q = (a * np.conj(b))/ abs(a * np.conj(a) + 1e-10)

    mask = np.ones_like(Q)
    #TODO: fix mask
    t = np.max(abs(a))*threshold

    mask[ abs(a) > t ] = 1
    # mask = medfilt2d(mask, 10 );       # 2D median filter of mask

    U, S, VH = np.linalg.svd(Q*mask, full_matrices=False)

    # noise free PCM has rank one so only first components are relevant
    U = U[:,0]
    V = VH.T.conj()
    V = V[:,0]
    
    n2 = len(V)

    # take centralized window of vector
    t_n = np.arange(np.ceil((0.5-radius)*n2)-1, np.floor((0.5+radius)*n2)).astype(int) 
    
    unwrapped_angle_v = np.unwrap(np.angle(V[t_n]))
    A = np.column_stack(((t_n).T, np.ones_like(t_n.T)))

    # Perform linear least squares fit
    sys_y, _, _, _ = np.linalg.lstsq(A, unwrapped_angle_v, rcond=None)

    # mu is the slope of the fitted line
    mu_1 = sys_y[0]

    # equals mag of vertical shift of b to a
    y = mu_1

    m2 = len(U)
    t_m = np.arange(np.ceil((0.5-radius)*m2)-1, np.floor((0.5+radius)*m2)).astype(int) # take centralized window of vector
    
    unwrapped_angle_u = np.unwrap(np.angle(U[t_m]))
    A = np.column_stack(((t_m).T, np.ones_like(t_m.T)))

    # Perform linear least squares fit
    sys_x, _, _, _ = np.linalg.lstsq(A, unwrapped_angle_u, rcond=None)

    # mu is the slope of the fitted line
    mu_2 = sys_x[0]

    # equals mag of horizontal shift of b to a
    x = mu_2

    # calculate Psi using slopes of hor and ver shift over complete length and hight
    Psi = np.exp( 1j* x * np.arange(m).reshape(-1, 1) ) * np.exp( -1j* y * np.arange(n).reshape(-1, 1).T )
    mc = m//2-1
    nc = n//2-1

    # This rotates Psi in the complex plane to align its phase with Q at the center pixel
    Psi = Psi * np.exp(np.angle(np.conj(Psi[mc,nc])*Q[mc,nc])*1j)

    return Psi

def get_B(b, g):
    """Compute B matrix from b values and g vectors

    Args:
        b (1D array): b values
        g (2D array): g vectors

    Output:
        B (array): [gx**2, gy**2, gz**2,
                    2*gx*gy, 2*gx*gz, 2*gy*gz] of every pixel
    """
    num_g, num_axis = g.shape

    assert num_axis == 3
    assert num_g == len(b)

    gx = g[:, 0]
    gy = g[:, 1]
    gz = g[:, 2]

    return - b * np.array([gx**2, 2*gx*gy, gy**2,
                           2*gx*gz, 2*gy*gz, gz**2]).transpose()

def get_B2(b, g):
    """For Diffusion Kurtosis:
        Compute B2 matrix from b values and g vectors

    Args:
        b (1D array): b values
        g (2D array): g vectors

    Output:
        B (array)
    """
    num_g, num_axis = g.shape

    assert num_axis == 3
    assert num_g == len(b)

    gx = g[:, 0]
    gy = g[:, 1]
    gz = g[:, 2]

    BT = get_B(b, g)

    BK = b * b * np.array([
                     gx**4 / 6,
                     gy**4 / 6,
                     gz**4 / 6,
                     4 * gx**3 * gy / 6,
                     4 * gx**3 * gz / 6,
                     4 * gy**3 * gx / 6,
                     4 * gy**3 * gz / 6,
                     4 * gz**3 * gx / 6,
                     4 * gz**3 * gy / 6,
                     gx**2 * gy**2,
                     gx**2 * gz**2,
                     gy**2 * gz**2,
                     2 * gx**2 * gy * gz,
                     2 * gy**2 * gx * gz,
                     2 * gz**2 * gx * gy]).transpose()

    return np.concatenate((BT, BK), axis=1)


def get_D(B, sig, fit_method='wls', fit_only_tensor=False,
          min_signal=0, fit_kt=False):
    """Compute D matrix (diffusion tensor)

    Args:
        B (array): see above.
        sig (array): b0 image and diffusion-weighted images.
        fit_method (string): [default: 'wls']
            - 'wls' weighted least square
            - 'ols' ordinary least square
        fit_only_tensor (boolean): excluding b0 [default: False]
        min_signal (float): minimal signal intensity in DWI
            [Default: MIN_POSITIVE_SIGNAL]
            better set to 0.
        fit_kt (boolean): fit kurtosis tensor directly [default: False]

    Output:
        D (array): [Dxx, Dxy, Dyy, Dxz, Dyz, Dzz] of every pixel.
        Please refer to get_B() and get_B2() for the actual order
        of the D array.

    References:
        Chung S. W., Lu Y., Henry R. G. (2006).
        Comparison of bootstrap approaches for
        estimation of uncertainties of DTI parameters.
        NeuroImage 33, 531-541.

        DiPy. https://github.com/dipy/dipy
    """
    sig = np.abs(sig)
    sig = np.maximum(sig, min_signal)
    S = np.log(sig, out=np.zeros_like(sig), where=(sig != 0))

    ndiff = S.shape[0]
    image_shape = S.shape[1:]

    if fit_only_tensor is True:
        y = S[0, ...] - S
    else:
        y = S
        dummy = np.ones((B.shape[0], 1))
        B = np.concatenate((B, dummy), axis=1)

    nparam = B.shape[1]
    yr = y.reshape(ndiff, -1)

    # print('> OLS Fitting')
    xr = np.dot(np.linalg.pinv(B), yr)
    D_fit = xr.reshape([nparam] + list(image_shape))

    if fit_method == 'wls':
        # print('> WLS Fitting')

        if fit_kt is True:
            eigvals, eigvecs = get_eig(D_fit, B)
            MD2 = get_MD(eigvals)**2
            scale = np.tile(MD2[None, ...], [nparam] + [1] * len(image_shape))
            scale = np.reshape(scale, (nparam, -1))
            scale[:6, ...] = 1
            scale[-1, ...] = 1
        else:
            scale = np.ones_like(xr)

        scale = np.expand_dims(scale.T, axis=1)

        w = np.exp(np.dot(B, xr)).T  # weight

        lhs = np.linalg.pinv(B * w[..., None] * scale, rcond=1e-15)
        lhs = np.swapaxes(lhs, 0, 1)

        rhs = (w.T * yr).T

        xr = np.sum(lhs * rhs, axis=-1)

    return xr.reshape([nparam] + list(image_shape))


_lt_indices = np.array([[0, 1, 3],
                        [1, 2, 4],
                        [3, 4, 5]])


def DT_vec2mat(Dvec):
    """Convert the 6 elements of diffusion tensor (DT)
    to a 3x3 symmetric matrix
    """
    assert 6 == Dvec.shape[0]

    return Dvec[_lt_indices, ...]


def get_eig(D, B=None):
    """Compute eigenvalues and eigenvectors of the D matrix

    Args:
        D (array): output from get_D(B, sig)

    Output:
        eigvals: eigenvalues
        eigvecs: eigenvectors
    """
    image_shape = D.shape[1:]
    image_size = np.prod(image_shape)

    Dmat = DT_vec2mat(D[:6, ...])
    temp = np.rollaxis(Dmat, 0, len(Dmat.shape))
    Dmat = np.rollaxis(temp, 0, len(Dmat.shape))
    eigvals, eigvecs = np.linalg.eigh(Dmat)

    # flatten eigvals and eigenvecs
    eigvals = eigvals.reshape(-1, 3)
    eigvecs = eigvecs.reshape(-1, 3, 3)

    order = eigvals.argsort()[:, ::-1]

    xi = np.ogrid[:image_size, :3][0]
    eigvals = eigvals[xi, order]

    xi, yi = np.ogrid[:image_size, :3, :3][:2]
    eigvecs = eigvecs[xi, yi, order[:, None, :]]

    eigvals = eigvals.reshape(image_shape + (3, ))
    eigvecs = eigvecs.reshape(image_shape + (3, 3))

    eigvals = np.rollaxis(eigvals, -1, 0)

    eigvecs = np.rollaxis(eigvecs, -1, 0)
    eigvecs = np.rollaxis(eigvecs, -1, 0)

    if B is not None:
        min_diffusivity = 1e-6 / -B.min()
        eigvals = eigvals.clip(min=min_diffusivity)

    return eigvals, eigvecs


def get_FA(eigvals):
    """Compute Fractional Anisotropy (FA) map

    Args:
        eigvals (array): output from get_eig(D)

    Output:
        FA (array): FA map
    """
    l1 = eigvals[0, ...]
    l2 = eigvals[1, ...]
    l3 = eigvals[2, ...]

    nomi = 0.5 * ((l1-l2)**2 + (l2-l3)**2 + (l3-l1)**2)
    deno = l1**2 + l2**2 + l3**2

    FA = np.sqrt(np.divide(nomi, deno,
                           out=np.zeros_like(nomi),
                           where=deno != 0))

    return FA


def get_cFA(FA, eigvecs):
    """Compute color-coded Fractional Anisotropy (cFA) map

    Args:
        FA (array): FA map
        eigvecs (array): eigen vectors

    Output:
        cFA (array): cFA map
    """
    return np.abs(eigvecs[:, 0, ...]) * FA


def get_MD(eigvals):
    """Compute Mean Diffusivity (MD) map

    Args:
        eigvals (array): output from get_eig(D)

    Output:
        MD (array): MD map
    """
    assert 3 == eigvals.shape[0]

    return np.mean(eigvals, axis=0)


def get_KT(D, B=None):
    """Compute Kurtosis Tensor (KT) map

    Args:
        D (array): output from get_D(B, sig)

    Output:
        KT (array): KT map
    """
    assert 21 <= D.shape[0]
    DT = D[:6, ...]
    DK = D[6:21, ...]

    eigvals, eigvecs = get_eig(DT, B=B)
    MD = get_MD(eigvals)

    return DK / (MD**2)


def get_ADC(D):
    """Compute the apparent diffusion coefficient (ADC) map

    Args:
        D (array): diffusion tensor

    Output:
        ADC (array): ADC map
    """
    Dxx = D[0, ...]
    Dyy = D[2, ...]
    Dzz = D[5, ...]

    return (Dxx + Dyy + Dzz) / 3