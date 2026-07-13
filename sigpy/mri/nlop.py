# -*- coding: utf-8 -*-
"""MRI non-linear operators.

This module contains these non-linear operators:

    * Nlinv,
        joint coil sensitivity maps and image content.

    * Diffusion,
        exponential diffusion modelling and parallel imaging sampling.

Author: Zhengguo Tan <zhengguo.tan@gmail.com>
"""
import sigpy as sp

from sigpy import nlop
from sigpy.mri import linop
import numpy as np


class Nlinv(sp.nlop.Nlop):
    """
    Construction of the non-linear parallel imaging (nlinv) operator.

    Given the unknown x = (rho, c_1, ..., c_N)^T
    , where
        rho: image content, and
        c_1, ..., c_N: N coil sensitivity maps,

    the forward operation is,

        F(x) = ( ..., FT{rho * c_n}, ... )^T

    , where
        n in [1, N], and
        FT is either masked FFT or NUFFT.

    Args:
        image_shape (tuple): shape of image.
        coil_shape (tuple): shape of coils.
        coord (None or array): coordinates, i.e. trajectories
        coil (None or array): coil sensitivity maps.
        W_coil (boolean): apply Sobolev weight on coil or not.
        upd_coil (boolean): update coil sensitivity maps or not.

    Reference:
        Bauer F., Kannengiesser S. (2007).
        An alternative approach to the image reconstruction
        for parallel data acquisition in MRI.
        Math. Methods Appl. Sci., 30, 1437-1451.

        Uecker M., Hohage T., Block K. T., Frahm J. (2008).
        Image reconstruction by regularized nonlinear inversion -
        joint estimation of coil sensitivities and image content.
        Magn. Reson. Med., 60, 674-682.
    """

    def __init__(self, image_shape, coil_shape,
                 coord=None, coil=None,
                 W_coil=True, upd_coil=True,
                 repr_str=None):
        self.image_shape = image_shape
        self.coil_shape = coil_shape

        ishape = self._get_xshape()

        self.coord = coord
        self.coil = coil
        self.upd_coil = upd_coil

        # Sobolev linear operator on coils
        if W_coil:
            self.W = sp.linop.Sobolev(self.coil_shape)
        else:
            self.W = sp.linop.Identity(self.coil_shape)

        # FFT or NUFFT operator
        x_ndim = len(ishape)
        if coord is None:
            self.F = sp.linop.FFT(self.coil_shape, axes=range(-x_ndim+1, 0))
        else:
            self.F = sp.linop.NUFFT(self.coil_shape, coord)

        oshape = self.F.oshape

        super().__init__(oshape, ishape, repr_str)

    def _get_xshape(self):

        image_ndim = len(self.image_shape)

        if image_ndim == 2:
            num_coilimg = 1 + self.coil_shape[0]
        else:
            num_coilimg = self.image_shape[0] + self.coil_shape[0]

        xshape = []   # empty list
        xshape.append(num_coilimg)

        return xshape + list(self.image_shape[-2:])

    def _forward(self, input):
        with sp.backend.get_device(input):

            # store the current estimate into class
            self.x = input

            image = self.x[0, :, :]   # extract image
            coil_ksp = self.x[1:, :, :]   # extract coils
            coil_img = self.W * coil_ksp

            return self.F(image * coil_img)

    def _get_Jacobian(self, x):
        return None

    def _derivative(self, x, dx):
        device = sp.backend.get_device(dx)

        self.x = x

        with device:
            image = self.x[0, :, :]
            coil_ksp = self.x[1:, :, :]
            coil_img = self.W * coil_ksp

            dimage = dx[0, :, :]
            dcoil_ksp = dx[1:, :, :]
            dcoil_img = self.W * dcoil_ksp

            return self.F * (dimage * coil_img + image * dcoil_img)

    def _adjoint(self, x, dy):
        device = sp.backend.get_device(dy)
        xp = device.xp

        self.x = x

        output = xp.zeros_like(self.x)

        with device:
            image = self.x[0, :, :]
            coil_ksp = self.x[1:, :, :]
            coil_img = self.W * coil_ksp

            dcoilimg = self.F.H * dy

            output[0, :, :] = xp.sum(xp.conj(coil_img) * dcoilimg, axis=0)

            if self.upd_coil:
                output[1:, :, :] = self.W.H(xp.conj(image) * dcoilimg)

            return output


# class Diffusion(sp.nlop.Nlop):
def Diffusion(input_shape, diff_enc, coil,
              scale=None, rvc=False, dwi_phase=None,
              coord=None, weights=None):
    """
    Construction of the non-linear Diffusion operator.

    Given the unknown x = D
    , where
        D: diffusion tensor,

    the forward operation is,

        A(x) = E * P, and E = F * S

    , where
        E (linop): the SENSE linear operator,
        F (linop): k-space sampling operator,
        S (linop): multiply with coil sensitivity maps, and
        P (nlop): exponential diffusion model.

    Args:
        input_shape (tuple): shape of input images.
        diff_enc (array): diffusion encoding matrix,
        i.e. the output matrix from sp.mri.epi.get_B().
        coil (array): coil sensitivity maps.
        coord (None or array): coordinates, i.e. trajectories.
    """
    const_b0 = True if input_shape[0] == 6 or input_shape[0] == 21 else False

    P = nlop.Exponential(input_shape, diff_enc,
                         const_a=const_b0, rvc=rvc, scale=scale)

    # phase correction for every diffusion-weighted image
    if dwi_phase is not None:
        I = sp.linop.Multiply(P.oshape, dwi_phase)
    else:
        I = sp.linop.Identity(P.oshape)

    # parallel imaging forward model (Sense)
    E = linop.Sense(coil, ishape=P.oshape, coord=coord, weights=weights)

    # Compose (i.e. Chain) Sense linop with Diffusion nlop
    A = E * I * P
    A.repr_str = 'Diffusion'

    return A


class FatWaterReco(sp.nlop.Nlop):
    """
    Construction of the non-linear parallel imaging (nlinv) operator.

    Given the unknown x = (rho, c_1, ..., c_N)^T
    , where
        rho: image content, and
        c_1, ..., c_N: N coil sensitivity maps,

    the forward operation is,

        F(x) = ( ..., FT{rho * c_n}, ... )^T

    , where
        n in [1, N], and
        FT is either masked FFT or NUFFT.

    Args:
        image_shape (tuple): shape of image (len=3, contains Fat, Water and Phi, each shape of N x N x N).
        coil (array): coil sensitivity maps (C x N x N x N).
        coord (None or array): coordinates, i.e. trajectories
        fat_peaks (list(tuple)): chemical shift of fat peaks in Hz and relative amplitude
        echo_times (array): echo times in milliseconds
        gradient_times (array): exact readout gradient times in milliseconds   
        field_strength (float): field strength in Tesla    

    """

    def __init__(self, image_shape, coil,
                 fat_peaks: list[tuple], 
                 echo_times: np.array, 
                 gradient_times: np.array,
                 field_strength: float = 7.0,
                 coord=None, 
                 repr_str=None):
        
        print("Initializing FatWaterReco with image_shape:", image_shape)
        coil =coil[:,np.newaxis, :, :, :]  # add a new axis for coil dimension
        self.image_shape = image_shape
        self.coil_shape = list(coil.shape)
        self.coil = coil
        self.fat_peaks = fat_peaks
        self.echo_times = echo_times
        self.gradient_times = gradient_times
        self.field_strength = field_strength

        assert len(echo_times.shape) == 2, "echo_times must be a 2D array, was shape {}".format(echo_times.shape)

        ishape = self._get_xshape()

        self.coord = coord
        self.coil = coil

        # Sobolev linear operator on coils

        # FFT or NUFFT operator
        x_ndim = len(ishape)

        

        coil_ishape = [1, echo_times.shape[0]] + image_shape[-3:] 
        self.S = sp.linop.Multiply(coil_ishape,self.coil)
        print('coil shapes')
        print(self.S.oshape)
        print(self.S.ishape)
        print(coil_ishape)
        print(self.coil.shape)

        if coord is None:
            self.F = sp.linop.FFT(self.coil_shape, axes=range(-x_ndim+1, 0))
        else:
            print([self.S.oshape[0], 1] + self.S.oshape[2:])

            self.F = sp.linop.Diag(
                                    [
                                    sp.linop.ToDevice(shape=[self.S.oshape[0], 1] + list(coord.shape[1:3]), odevice=sp.Device(-1), idevice=sp.Device(0)) * 
                                    sp.linop.NUFFT([self.S.oshape[0], 1] + self.S.oshape[2:] , coord[TE,...]) *
                                    sp.linop.ToDevice(shape=[self.S.oshape[0], 1] + self.S.oshape[2:], odevice=sp.Device(0), idevice=sp.Device(-1))
                                    for TE in range(self.echo_times.shape[0])], 
                        iaxis=1,
                        oaxis=1  
            )

        print(ishape[2:])
        self.exp_op = nlop.Exponential(ishape[2:], self.echo_times*1j*np.pi*2,
                         const_a=True, rvc=False, scale=None)
        
        self.D_t = self._get_D_t()
        self.D_t_adj = self._get_D_t(adjoint=True)

        oshape = self.F.oshape


        super().__init__(oshape, ishape, repr_str)

    def _get_xshape(self):
        #TODO: check if this is correct for Fat, Water, Phi reconstruction

        image_ndim = len(self.image_shape)

        if image_ndim == 1:
            num_coilimg = self.image_shape[0] + self.coil_shape[0]
        else:
            num_coilimg = 1 + self.coil_shape[0]

        xshape = [3] + [1] + [1] + self.image_shape[-3:]
    
        return xshape
    

    def _get_D_t(self, adjoint: bool =False)-> np.array: 
        gamma = 42.577478518e6  # gyromagnetic ratio in Hz/T
        CF = gamma * self.field_strength  # Larmor frequency in Hz
        f_off_factor = CF * 1e-6  # Hz

        print(self.gradient_times.shape)

        for m in range(len(self.fat_peaks)):
            delta_f = self.fat_peaks[m][0] * f_off_factor  # chemical shift in Hz
            rel_amp = self.fat_peaks[m][1]  # relative amplitude

            # Calculate the phase evolution for each echo time
            if adjoint:
                phase_evolution = np.exp(1j * 2 * np.pi * delta_f * self.gradient_times)
            else:
                phase_evolution = np.exp(-1j * 2 * np.pi * delta_f * self.gradient_times)

            # Calculate the signal contribution for this fat peak
            signal_contribution = rel_amp * phase_evolution

            if m == 0:
                D_t = signal_contribution
            else:
                D_t += signal_contribution

        return D_t
    

    def _forward(self, input):
        with sp.backend.get_device(input):

            # store the current estimate into class
            self.x = input
            print("Forward operation with input shape:", input.shape)

            Water = self.x[0, :, :]   # extract Water image
            Fat = self.x[1, :, :]   # extract Fat image
            Phi = self.x[2,0, :, :]   # extract Phi image

            expo_Phi = self.exp_op(Phi)

            Water_decay = expo_Phi * Water
            Water_decay_coil = self.S(Water_decay)

            Fat_decay = expo_Phi * Fat
            Fat_decay_coil = self.S(Fat_decay)

            ksp_wat = np.zeros(self.oshape, dtype=Water_decay_coil.dtype)
            ksp_fat = np.zeros(self.oshape, dtype=Fat_decay_coil.dtype)

            # print("F.ishape:", self.F.ishape)
            # print("F.oshape:", self.F.oshape)
            # print("Water_decay_coil shape:", Water_decay_coil.shape)
            # print("Fat_decay_coil shape:", Fat_decay_coil.shape)
            # print("D_t shape:", self.D_t.shape)
            
            ksp_wat = self.F(Water_decay_coil)
            ksp_fat = self.D_t[np.newaxis, ..., np.newaxis] *self.F(Fat_decay_coil)
                # out += self.D_t * sp.to_device(self.F(Fat_decay_coil[c, :, :, :]), device=sp.backend.cpu_device) + sp.to_device(self.F(Water_decay_coil[c, :, :, :]), device=sp.backend.cpu_device)

            return ksp_wat + ksp_fat

    def _get_Jacobian(self, x):
        return None

    def _derivative(self, x, dx):
        device = sp.backend.get_device(dx)
        xp = device.xp

        self.x = x

        with device:
            Water = self.x[0, :, :]
            Fat = self.x[1, :, :]
            Phi = self.x[2,0, :, :]

            dWater = dx[0, :, :]
            dFat = dx[1, :, :]
            dPhi = dx[2,0, :, :]

            factor_phi = self.echo_times[...,np.newaxis, np.newaxis] * 1j*np.pi*2 * dPhi

            fat_der = self.D_t[np.newaxis, ..., np.newaxis] * self.F(self.S(self.exp_op(Phi) * (dFat + Fat * factor_phi)))
            water_der = self.F(self.S(self.exp_op(Phi) * (dWater + Water * factor_phi)))
            return  water_der + fat_der

    def _adjoint(self, x, dy):
        # print('adjoint')
        device = sp.backend.get_device(dy)
        xp = device.xp

        self.x = x

        output = np.zeros_like(self.x)

        # print(output.shape)

        with device:
            Water = self.x[0, :, :]
            Fat = self.x[1, :, :]
            Phi = self.x[2,0, :, :]

            wat_out = np.sum(self.S.H * (self.exp_op(-Phi)[np.newaxis, ...] * self.F.H(dy)), axis=1)
            output[0, :, :] = wat_out
            # print('wat_out shape:', wat_out.shape)
            output[1, :, :] = np.sum((self.S.H * (self.exp_op(-Phi)[np.newaxis, ...] * self.F.H(self.D_t_adj[np.newaxis, ..., np.newaxis] * dy))), axis=1)

            out_1 = np.conj(Fat[...]) * self.F.H(self.D_t_adj[np.newaxis, ..., np.newaxis] * dy)
            # print('out_1 shape:', out_1.shape)

            out_2 = np.conj(Water[...])*self.F.H(dy)
            # print('out_2 shape:', out_2.shape)
            # print(self.exp_op(-Phi)[np.newaxis, ...].shape)

            out_3 = (-1j*np.pi*2*self.echo_times[np.newaxis, ..., np.newaxis, np.newaxis]) * self.exp_op(-Phi)[np.newaxis, ...] * (out_2 + out_1)
            # print('out_3 shape:', out_3.shape)

            out_4 = np.real(np.sum(self.S.H * out_3,axis=1))
            # print('out_4 shape:', out_4.shape)

            output[2, 0, :, :] = out_4

            return output