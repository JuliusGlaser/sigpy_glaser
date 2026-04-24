# -*- coding: utf-8 -*-
"""This module contains an abstraction class Prox for proximal operators,
and provides commonly used proximal operators, including soft-thresholding,
l1 ball projection, and box constraints.
"""
from re import S
import numpy as np
import random
import torch
import torch.nn as nn

from sigpy import backend, util, thresh, linop
import time
from multiprocessing import Pool, shared_memory


class Prox(object):
    r"""Abstraction for proximal operator.

    Prox can be called on a float (:math:`\alpha`) and
    a NumPy or CuPy array (:math:`x`) to perform a proximal operation.

    .. math::
        \text{prox}_{\alpha g} (y) =
        \text{argmin}_x \frac{1}{2} || x - y ||_2^2 + \alpha g(x)

    Prox can be stacked, and conjugated.

    Args:
        shape: Input/output shape.
        repr_str (string or None): default: class name.

    Attributes:
        shape: Input/output shape.

    """

    def __init__(self, shape, repr_str=None):
        self.shape = list(shape)

        if repr_str is None:
            self.repr_str = self.__class__.__name__
        else:
            self.repr_str = repr_str

    def _check_shape(self, input):
        for i1, i2 in zip(input.shape, self.shape):
            if i2 != -1 and i1 != i2:
                raise ValueError(
                    "shape mismatch for {s}, got {input_shape}.".format(
                        s=self, input_shape=input.shape
                    )
                )

    def __call__(self, alpha, input):
        try:
            self._check_shape(input)
            output = self._prox(alpha, input)
            self._check_shape(output)
        except Exception as e:
            raise RuntimeError("Exceptions from {}.".format(self)) from e

        return output

    def __repr__(self):
        return "<{shape} {repr_str} Prox>.".format(
            shape=self.shape, repr_str=self.repr_str
        )


class Conj(Prox):
    r"""Returns the proximal operator for the convex conjugate function.

    The proximal operator of the convex conjugate function
    :math:`g^*` is defined as:

    .. math::
        \text{prox}_{\alpha g^*} (x) =
        x - \alpha \text{prox}_{\frac{1}{\alpha} g} (\frac{1}{\alpha} x)

    """

    def __init__(self, prox):
        self.prox = prox
        super().__init__(prox.shape)

    def _prox(self, alpha, input):
        with backend.get_device(input):
            return input - alpha * self.prox(1 / alpha, input / alpha)


class NoOp(Prox):
    r"""Proximal operator for empty function.
    Equivalant to an identity function.

    Args:
       shape (tuple of ints): Input shape

    """

    def __init__(self, shape):
        super().__init__(shape)

    def _prox(self, alpha, input):
        return input


class Stack(Prox):
    r"""Stack outputs of proximal operators.

    Args:
       proxs (list of proxs): Prox of the same shape.

    """

    def __init__(self, proxs):
        self.nops = len(proxs)

        assert (self.nops > 0)

        self.proxs = proxs
        self.shapes = [prox.shape for prox in proxs]
        shape = [sum(util.prod(prox.shape) for prox in proxs)]

        super().__init__(shape)

    def _prox(self, alpha, input):
        with backend.get_device(input):
            if np.isscalar(alpha):
                alphas = [alpha] * self.nops
            else:
                alphas = util.split(alpha, self.shapes)

            inputs = util.split(input, self.shapes)
            outputs = [
                prox(alpha, input)
                for prox, input, alpha in zip(self.proxs, inputs, alphas)
            ]
            output = util.vec(outputs)

            return output


class UnitaryTransform(Prox):
    r"""Unitary transform input space.

    Returns a proximal operator that does

    .. math::
        A^H \text{prox}_{\alpha g}(A x)

    Args:
        prox (Prox): Proximal operator.
        A (Linop): Unitary linear operator.

    """

    def __init__(self, prox, A):
        self.prox = prox
        self.A = A

        super().__init__(A.ishape)

    def _prox(self, alpha, input):
        return self.A.H(self.prox(alpha, self.A(input)))


class L2Reg(Prox):
    r"""Proximal operator for l2 regularization.

    .. math::
        \min_x \frac{1}{2} \|x - y\|_2^2 + \frac{\lambda}{2}\|x-z\|_2^2 + h(x)

    Args:
        shape (tuple of ints): Input shape.
        lamda (float): Regularization parameter.
        y (scalar or array): Bias term.
        proxh (Prox): optional additional proximal operator.

    """

    def __init__(self, shape, lamda, y=None, proxh=None):
        self.lamda = lamda
        self.y = y
        self.proxh = proxh

        super().__init__(shape)

    def _prox(self, alpha, input):
        with backend.get_device(input):
            output = input.copy()
            if self.y is not None:
                output += (self.lamda * alpha) * self.y

            output /= 1 + self.lamda * alpha

            if self.proxh is not None:
                return self.proxh(alpha / (1 + self.lamda * alpha), output)

        return output


class L2Proj(Prox):
    r"""Proximal operator for l2 norm projection.

    .. math::
        \min_x \frac{1}{2} \| x - y \|_2^2 + 1\{\| x \|_2 < \epsilon\}

    Args:
        shape (tuple of ints): Input shape.
        epsilon (float): Regularization parameter.
        y (scalar or array): Bias term.

    """

    def __init__(self, shape, epsilon, y=0, axes=None):
        self.epsilon = epsilon
        self.y = y
        self.axes = axes

        super().__init__(shape)

    def _prox(self, alpha, input):
        with backend.get_device(input):
            return (
                thresh.l2_proj(self.epsilon, input - self.y, self.axes)
                + self.y
            )


class LInfProj(Prox):
    r"""Proximal operator for l-infinity ball projection.

    .. math::
        \min_x \frac{1}{2} \| x - y \|_2^2 + 1\{\| x \|_\infty < \epsilon\}

    Args:
        shape (tuple of ints): Input shape.
        epsilon (float): Regularization parameter.
        y (scalar or array): Bias term.

    """

    def __init__(self, shape, epsilon, bias=None, axes=None):
        self.epsilon = epsilon
        self.bias = bias
        self.axes = axes

        super().__init__(shape)

    def _prox(self, alpha, input):
        with backend.get_device(input):
            return thresh.linf_proj(self.epsilon, input, bias=self.bias)


class PsdProj(Prox):
    r"""Proximal operator for positive semi-definite matrices.

    .. math::
        \min_x \frac{1}{2} \| X - Y \|_2^2 + 1\{\| X \succeq 0\}

    Args:
        shape (tuple of ints): Input shape.

    """

    def _prox(self, alpha, input):
        with backend.get_device(input):
            return thresh.psd_proj(input)


class L1Reg(Prox):
    r"""Proximal operator for l1 regularization.

    .. math::
        \min_x \frac{1}{2} \| x - y \|_2^2 + \lambda \| x \|_1

    Args:
        shape (tuple of ints): input shape
        lamda (float): regularization parameter

    """

    def __init__(self, shape, lamda):
        self.lamda = lamda

        super().__init__(shape)

    def _prox(self, alpha, input):
        with backend.get_device(input):
            return thresh.soft_thresh(self.lamda * alpha, input)


class L1Proj(Prox):
    r"""Proximal operator for l1 norm projection.

    .. math::
        \min_x \frac{1}{2} \| x - y \|_2^2 + 1\{\| x \|_1 < \epsilon\}

    Args:
        shape (tuple of ints): input shape.
        epsilon (float): regularization parameter.

    """

    def __init__(self, shape, epsilon):
        self.epsilon = epsilon

        super().__init__(shape)

    def _prox(self, alpha, input):
        with backend.get_device(input):
            return thresh.l1_proj(self.epsilon, input)


class BoxConstraint(Prox):
    r"""Box constraint proximal operator.

    .. math::
        \min_{x : l \leq x \leq u} \frac{1}{2} \| x - y \|_2^2

    Args:
        shape (tuple of ints): input shape.
        lower (scalar or array): lower limit.
        upper (scalar or array): upper limit.

    """

    def __init__(self, shape, lower, upper):
        self.lower = lower
        self.upper = upper
        super().__init__(shape)

    def _prox(self, alpha, input):
        device = backend.get_device(input)
        xp = device.xp

        with device:
            return xp.clip(input, self.lower, self.upper)

class LLRL1Reg(Prox):
    r"""Local Low Rank L1 Regularization

    Args:
        shape (tuple of int): input shapes.
        lamda (float): regularization parameter.
        randshift (boolean): switch on random shift or not.
        blk_shape (tuple of int): block shape [default: (8, 8)].
        blk_strides (tuple of int): block strides [default: (8, 8)].

    References:
        * Cai JF, Candes EJ, Shen Z.
          A singular value thresholding algorithm
          for matrix completion.
          SIAM J Optim 20:1956-1982 (2010).

        * Trzasko J, Manduca A.
          Local versus global low-rank promotion
          in dynamic MRI series reconstruction.
          Proc. ISMRM 19:4371 (2011).

        * Zhang T, Pauly J, Levesque I.
          Accelerating parameter mapping with a locally low rank constraint.
          Magn Reson Med 73:655-661 (2015).

        * Saucedo A, Lefkimmiatis S, Rangwala N, Sung K.
          Improved computational efficiency of locally low rank
          MRI reconstruction using iterative random patch adjustments.
          IEEE Trans Med Imaging 36:1209-1220 (2017).

        * Hu Y, Wang X, Tian Q, Yang G, Daniel B, McNab J, Hargreaves B.
          Multi-shot diffusion-weighted MRI reconstruction
          with magnitude-based
          spatial-angular locally low-rank regularization (SPA-LLR).
          Magn Reson Med 83:1596-1607 (2020).

    Author:
        Zhengguo Tan <zhengguo.tan@gmail.com>
    """

    def __init__(self, shape, lamda, randshift=True,
                 blk_shape=(8, 8), blk_strides=(8, 8),
                 reg_magnitude=False,
                 normalization=False,
                 verbose=False):
        self.lamda = lamda
        self.randshift = randshift
        self.reg_magnitude = reg_magnitude
        self.normalization = normalization

        assert len(blk_shape) == len(blk_strides)
        self.blk_shape = blk_shape
        self.blk_strides = blk_strides
        self.verbose = verbose

        # construct forward linops
        self.RandShift = self._linop_randshift(shape, blk_shape, randshift)
        self.A = linop.ArrayToBlocks(shape, blk_shape, blk_strides)
        self.Reshape = self._linop_reshape()

        self.Fwd = self.Reshape * self.A * self.RandShift

        super().__init__(shape)

    def _check_blk(self):
        assert len(self.blk_shape) == len(self.blk_strides)

    def _prox(self, alpha, input):
        device = backend.get_device(input)
        xp = device.xp

        with device:
            

            if self.reg_magnitude:
                mag = xp.abs(input)
                phs = xp.exp(1j * xp.angle(input))

            else:
                mag = input.copy()
                phs = xp.ones_like(mag)

            output = self.Fwd(mag)

            # t = time.time()
            output = backend.to_device(output, device=-1)
            # print("Time to move data to device: ", time.time()-t)
            # print("Output device: ", output.device)
            # t = time.time()
            u, s, vh = np.linalg.svd(output, full_matrices=False)
            # print("Time for SVD: ", t-time.time())

            if self.normalization is True:
                s = s / self.blk_shape[-1]

            s_thresh = thresh.soft_thresh(self.lamda * alpha, s)

            if self.normalization is True:
                s_thresh = s_thresh * self.blk_shape[-1]

            output = (u * s_thresh[..., None, :]) @ vh

            output = backend.to_device(output, device=device)

            output = self.Fwd.H(output)

            return output * phs

    def _linop_randshift(self, shape, blk_shape, randshift):

        D = len(blk_shape)

        if randshift is True:
            axes = range(-D, 0)
            shift = [random.randint(0, blk_shape[s]) for s in axes]

            return linop.Circshift(shape, shift, axes)
        else:
            return linop.Identity(shape)

    def _linop_reshape(self):
        D = len(self.blk_shape)

        oshape = [util.prod(self.A.ishape[:-D]),
                  util.prod(self.A.num_blks),
                  util.prod(self.blk_shape)]

        R1 = linop.Reshape(oshape, self.A.oshape)
        R2 = linop.Transpose(R1.oshape, axes=(1, 0, 2))
        return R2 * R1
    
class LLRL1Reg_3d_Rad(Prox):
    r"""Local Low Rank L1 Regularization

    Args:
        shape (tuple of int): input shapes.
        lamda (float): regularization parameter.
        randshift (boolean): switch on random shift or not.
        blk_shape (tuple of int): block shape [default: (8, 8)].
        blk_strides (tuple of int): block strides [default: (8, 8)].

    References:
        * Cai JF, Candes EJ, Shen Z.
          A singular value thresholding algorithm
          for matrix completion.
          SIAM J Optim 20:1956-1982 (2010).

        * Trzasko J, Manduca A.
          Local versus global low-rank promotion
          in dynamic MRI series reconstruction.
          Proc. ISMRM 19:4371 (2011).

        * Zhang T, Pauly J, Levesque I.
          Accelerating parameter mapping with a locally low rank constraint.
          Magn Reson Med 73:655-661 (2015).

        * Saucedo A, Lefkimmiatis S, Rangwala N, Sung K.
          Improved computational efficiency of locally low rank
          MRI reconstruction using iterative random patch adjustments.
          IEEE Trans Med Imaging 36:1209-1220 (2017).

        * Hu Y, Wang X, Tian Q, Yang G, Daniel B, McNab J, Hargreaves B.
          Multi-shot diffusion-weighted MRI reconstruction
          with magnitude-based
          spatial-angular locally low-rank regularization (SPA-LLR).
          Magn Reson Med 83:1596-1607 (2020).

    Author:
        Zhengguo Tan <zhengguo.tan@gmail.com>
        Julius Glaer <julius-glaser@gmx.de>
    """

    def __init__(self, shape, lamda, randshift=True,
                 blk_shape=(8, 8), blk_strides=(8, 8),
                 slices_around_center=0,
                 bounding_box=None,
                 reg_magnitude=False,
                 normalization=False,
                 verbose=False):
        self.lamda = lamda
        self.randshift = randshift
        self.reg_magnitude = reg_magnitude
        self.normalization = normalization
        self.slices_around_center = slices_around_center

        assert len(blk_shape) == len(blk_strides)
        self.blk_shape = blk_shape
        self.blk_strides = blk_strides
        self.verbose = verbose

        self.run_svd_parallel = False
        self.iter = 0

        # construct forward linops
        # Construct forward slice-wise
        self.n_slice = shape[2]
        if self.slices_around_center == 0:
            self.slices_around_center = self.n_slice
        if bounding_box is not None:
            slice_x = slice(bounding_box[0][0], bounding_box[2][0])     #Just using one arbitrary bounding_box because the operator needs a constant size
            slice_y = slice(bounding_box[1][0], bounding_box[3][0])     # could be fixed by using multiple operators for different slices, but this is not implemented yet
        else:
            slice_x = slice(0, shape[-2])
            slice_y = slice(0, shape[-1])
        self.n_slice_chunk = self.blk_shape[0]
        shape = shape[0:2] + [self.n_slice_chunk] + [slice_y.stop - slice_y.start, slice_x.stop - slice_x.start]

        self.slice_x = slice_x
        self.slice_y = slice_y
        
        # print("Shape: ", shape)
        # print("blk_shape", blk_shape)

        self.RandShift = self._linop_randshift(shape, blk_shape, randshift)
        self.A = linop.ArrayToBlocks(shape, blk_shape, blk_strides)
        self.Reshape = self._linop_reshape()

        self.Fwd = self.Reshape * self.A * self.RandShift
        # print("Fwd inshape: ", self.Fwd.ishape)
        # print("Fwd out shape", self.Fwd.oshape)
        # print("A inshape: ", self.A.ishape)
        # print("A out shape", self.A.oshape)

        super().__init__(shape)

    def _check_blk(self):
        assert len(self.blk_shape) == len(self.blk_strides)

    @staticmethod
    def svd_worker(args):
        """
        Each worker:
        1. Attaches to the shared memory block (no copy of the full array)
        2. Reads its assigned slice
        3. Runs SVD + threshold
        4. Writes the result back in-place
        """
        (shm_name, shape, dtype,
        patch_start, patch_end,
        lamda, alpha, blk_shape_last, normalization, device) = args

        # Attach to the existing shared memory (created in the main process)
        existing_shm = shared_memory.SharedMemory(name=shm_name)
        output = np.ndarray(shape, dtype=dtype, buffer=existing_shm.buf)

        t = time.time()
        print(f">> SVD on patch [{patch_start}:{patch_end}]")

        # Pull the slice to a local array for SVD (numpy needs a contiguous array)
        output_part = np.array(output[patch_start:patch_end, ...])

        # Move to CPU if needed — adjust this to your backend's API
        # output_part = backend.to_device(output_part, device=-1)

        u, s, vh = np.linalg.svd(output_part, full_matrices=False)
        print(f">>> SVD time [{patch_start}:{patch_end}]: {time.time() - t:.3f}s")

        if normalization:
            s = s / blk_shape_last

        s_thresh = thresh.soft_thresh(lamda * alpha, s)

        if normalization:
            s_thresh = s_thresh * blk_shape_last

        result = (u * s_thresh[..., None, :]) @ vh

        # Move back to target device if needed
        # result = backend.to_device(result, device=device)

        # Write result directly into shared memory
        output[patch_start:patch_end, ...] = result

        existing_shm.close()  # detach (does NOT free the memory)

    def _prox(self, alpha, input):
        import time
        device = backend.get_device(input)
        # print("Prox_device = ", device)
        print("Iter: ", self.iter)
        xp = device.xp

        LLR_time = time.time()

        with device:
            

            if self.reg_magnitude:
                mag = xp.abs(input)
                phs = xp.exp(1j * xp.angle(input))

            else:
                mag = input.copy()
                phs = xp.ones_like(mag)

            #FIXME: transposing not necessary for other reconstruction
            mag = xp.transpose(mag, (0,1,4,3,2))  # [n_slice, N_diff, N_shot, N_y, N_x]

            # import h5py
            # with h5py.File(r'C:\Workspace\Temp_dir\RSR_LLR\meas_MID00180_FID03255_Multi_SlidingWindow_1000mum_20k.dat_reconstructed_steps_2_ADMM_total_it_5_lambda_0.001.h5', 'r+') as f:
            #     f.create_dataset(f'mag_iter_{self.iter}', data=backend.to_device(mag))
            # print("Mag shape: ", mag.shape)
            
            for n_slice in range(0,self.n_slice, self.n_slice_chunk):

                # print(f">> LLR on slice {n_slice} of {self.n_slice}")

                if n_slice >= (self.n_slice//2 - self.slices_around_center) and n_slice < (self.n_slice//2 + self.slices_around_center):
                    #takes time
                    output = self.Fwd(mag[:,:,n_slice:n_slice+self.n_slice_chunk,self.slice_y,self.slice_x])
                    # print("SVD computation")
                    # print(">>> shape of the array for SVD: ", output.shape)
                    
                    n_patches = output.shape[0]
                    # output_comb = xp.zeros_like(output)
                    if not self.run_svd_parallel:
                        steps = 10

                        for i in range(steps):

                            patch_start = i * (n_patches // steps)

                            if i == steps - 1:
                                patch_end = n_patches        # last step gets remainder
                            else:
                                patch_end = (i + 1) * (n_patches // steps)

                            output_part = output[patch_start:patch_end, ...]
                            output_part = backend.to_device(output_part, device=-1)
                            #apply LLR only on central slices
                            

                            # print(f">> SVD on patch {i} of {steps}")
                            # print("Patch start, patch end ", patch_start, patch_end)

                            u, s, vh = np.linalg.svd(output_part, full_matrices=False)

                            
                            # print('>>> SVD time: ' + str(time.time() - t) + ' seconds.')
                            # print("SVD component shapes: u {}, s {}, vh {}".format(u.shape, s.shape, vh.shape))

                            if self.normalization:
                                s = s / self.blk_shape[-1]

                            s_thresh = thresh.soft_thresh(self.lamda * alpha, s)

                            if self.normalization:
                                s_thresh = s_thresh * self.blk_shape[-1]

                            output_part = (u * s_thresh[..., None, :]) @ vh

                            output[patch_start:patch_end, ...] = backend.to_device(output_part, device=device)
                    else:
                        import psutil
                        output_np = np.array(backend.to_device(output, device=-1))
                        steps = 10#psutil.cpu_count(logical=False)//2-4 #For HPC 'FIXME: make config parameter for N_kernels
                        print(f">> running on {steps} kernels in parallel")
                        n_patches = output_np.shape[0]
                        # ── 1. Copy `output_np` into a shared memory block ──────────────────────────
                        #    All worker processes will map this same block — zero extra copies.
                        shm = shared_memory.SharedMemory(create=True, size=output_np.nbytes)
                        shared_arr = np.ndarray(output_np.shape, dtype=output_np.dtype, buffer=shm.buf)
                        shared_arr[:] = output_np          # one-time copy into shared memory

                        # ── 2. Build the argument list for each worker ───────────────────────────
                        chunk = n_patches // steps
                        args_list = []
                        for i in range(steps):
                            patch_start = i * chunk
                            patch_end   = n_patches if i == steps - 1 else (i + 1) * chunk
                            args_list.append((
                                shm.name,           # workers look up the block by name
                                output.shape,
                                output.dtype,
                                patch_start,
                                patch_end,
                                self.lamda,
                                alpha,
                                self.blk_shape[-1],      # only the scalar we actually need
                                self.normalization,
                                device,
                            ))

                        # ── 3. Launch all workers simultaneously ─────────────────────────────────
                        #    Pool size = steps so every patch gets its own process.
                        #    Cap it at os.cpu_count() if steps is large.
                        with Pool(processes=steps) as pool:
                            pool.map(self.svd_worker, args_list)

                        # ── 4. Copy results back from shared memory into the original array ───────
                        output_np[:] = shared_arr

                        # ── 5. Clean up shared memory ────────────────────────────────────────────
                        shm.close()
                        shm.unlink()   # frees the OS-level block; must be called exactly once

                        output = backend.to_device(xp.array(output_np), device=device)


                    output_LLR = self.Fwd.H(output)
                    output = mag[:,:,n_slice:n_slice+self.n_slice_chunk,...]    #extend FOV if bounding_box is used with original data
                    output[...,self.slice_y, self.slice_x] = output_LLR
                else:
                    output = mag[:,:,n_slice:n_slice+self.n_slice_chunk,...]
                # processed_out.append(backend.to_device(output))
                mag[:,:,n_slice:n_slice+self.n_slice_chunk,...] = output


            #FIXME: transposing not necessary for other reconstruction
            mag = xp.transpose(mag, (0,1,4,3,2))  # [n_slice, N_diff, N_shot, N_y, N_x]
            print(f"LLR time: {(time.time() - LLR_time)//60:.0f}:{(time.time() - LLR_time)%60:02.0f} min")

            return mag * phs

    def _linop_randshift(self, shape, blk_shape, randshift):

        D = len(blk_shape)

        if randshift is True:
            axes = range(-D, 0)
            shift = [random.randint(0, blk_shape[s]) for s in axes]

            return linop.Circshift(shape, shift, axes)
        else:
            return linop.Identity(shape)

    def _linop_reshape(self):
        D = len(self.blk_shape)

        oshape = [util.prod(self.A.ishape[:-D]),
                  util.prod(self.A.num_blks),
                  util.prod(self.blk_shape)]

        R1 = linop.Reshape(oshape, self.A.oshape)
        R2 = linop.Transpose(R1.oshape, axes=(1, 0, 2))
        return R2 * R1
    
    def _check_shape(self, input):
        pass


class SLRMCReg(Prox):
    r"""Structure Low Rank Matrix Completion as Regularization

    Args:
        shape (tuple of int): input shapes.
        lamda (float): regularization parameter.
        blk_shape (tuple of int): block shape [default: (7, 7)].
        blk_strides (tuple of int): block strides [default: (1, 1)].
        thresh (string): thresholding type ['soft' or 'hard'].

    References:
        * Mani M, Jacob M, Kelley D, Magnotta V.
          Multi-shot sensitivity-encoded diffusion data recovery using
          structured low-rank matrix completion (MUSSELS).
          Magn Reson Med 78:494-507 (2017).

        * Bilgic B, Chatnuntawech I, Manhard MK, Tian Q,
          Liao C, Iyer SS, Cauley SF, Huang SY,
          Polimeni JR, Wald LL, Setsompop K.
          Highly accelerated multishot echo planar imaging through
          synergistic machine learning and joint reconstruction.
          Magn Reson Med 82:1343-1358 (2019).

        * Dai E, Mani M, McNab JA.
          Multi-band multi-shot diffusion MRI reconstruction with
          joint usage of structured low-rank constraints
          and explicit phase mapping.
          Magn Reson Med 89:95-111 (2023).

    Author:
        Zhengguo Tan <zhengguo.tan@gmail.com>
    """
    def __init__(self, shape, lamda,
                 blk_shape=(7, 7), blk_strides=(1, 1),
                 thresh='hard', verbose=False):
        self.lamda = lamda

        assert len(blk_shape) == len(blk_strides)
        self.blk_shape = blk_shape
        self.blk_strides = blk_strides
        self.thresh = thresh
        self.verbose = verbose

        # construct forward linops
        self.A = linop.ArrayToBlocks(shape, blk_shape, blk_strides)
        self.Reshape = self._linop_reshape()

        self.Fwd = self.Reshape * self.A

        super().__init__(shape)

    def _prox(self, alpha, input):
        device = backend.get_device(input)
        xp = device.xp

        with device:

            output = self.Fwd(input)

            # SVD
            u, s, vh = xp.linalg.svd(output, full_matrices=False)

            if self.thresh == 'soft':  # soft thresholding

                s_thresh = thresh.soft_thresh(self.lamda * alpha, s)

                output = (u * s_thresh[..., None, :]) @ vh

            else:  # hard thresholding

                keep = int(self.lamda * alpha * len(s))

                if keep >= len(s):
                    keep = len(s)

                if self.verbose:
                    print('>>> shape of the array for SVD: ', output.shape)
                    print('>>> # of singular values kept ' + str(keep)
                          + ' of ' + str(len(s)))

                u_t, s_t, vh_t = u[..., :keep], s[:keep], vh[..., :keep, :]

                output = (u_t * s_t[..., None, :]) @ vh_t

            output = self.Fwd.H(output)

            return output

    def _linop_reshape(self):
        D = len(self.blk_shape)

        oshape1 = [util.prod(self.A.ishape[:-D]),
                   util.prod(self.A.num_blks),
                   util.prod(self.blk_shape)]

        R1 = linop.Reshape(oshape1, self.A.oshape)
        R2 = linop.Transpose(R1.oshape, axes=(0, 2, 1))

        oshape2 = [util.prod(R2.oshape[:-1]),
                   R2.oshape[-1]]

        R3 = linop.Reshape(oshape2, R2.oshape)
        R4 = linop.Transpose(R3.oshape, axes=(1, 0))

        return R4 * R3 * R2 * R1


class DAEReg(Prox):
    r"""Denoising AutoEncoder (DAE) as Regularization

    Args:
        shape (tuple of int): input shapes.
        DAE (nn.Module): Learned DAE model.

    References:
        * Mani M, Magnotta VA, Jacob M.
          qModeL: A plug-and-play model-based reconstruction
          for highly accelerated multi-shot diffusion MRI
          using learned priors.
          Magn Reson Med 86:835-851 (2021).

    Author:
        Zhengguo Tan <zhengguo.tan@gmail.com>
    """

    def __init__(self, shape, DAE: nn.Module):
        self.DAE = DAE
        self.shape = shape  # [N_diff, N_shot, N_c, N_z, N_y, N_x]

        super().__init__(shape)

    def _prox(self, alpha, input):
        self._check_shape(input)
        N_diff, N_shot, N_c, N_z, N_y, N_x = input_pt.shape

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        sp_device = backend.get_device(input)

        input_pt = torch.from_numpy(backend.to_device(input)).to(device)

        baseline = input[0]

        dwi_scale = torch.where(baseline!=0, torch.true_divide(input, baseline), torch.zeros_like(input))

        mag = torch.abs(dwi_scale)
        phs = torch.angle(dwi_scale)

        mag_r = mag.view(N_diff, -1).t()  # [-1, N_diff]

        reg_m = self.DAE(mag_r)
        reg_m = reg_m.t().view(self.shape)

        output = reg_m  * baseline * torch.exp(1j * phs)
        output = output.detach().cpu().numpy()

        return backend.to_device(output, device=sp_device)
