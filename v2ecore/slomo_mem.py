"""In-memory Super SloMo interpolator.

Accepts a numpy array of grayscale frames and returns interpolated frames
as a numpy array, with no intermediate disk I/O.
"""

import os
import numpy as np
import cv2
import torch
import torchvision.transforms as transforms

from tqdm import tqdm
import logging

import v2ecore.dataloader as dataloader
import v2ecore.model as model

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.functional")

logger = logging.getLogger(__name__)


class SuperSloMoMem:
    """Super SloMo interpolator that operates entirely in memory.

    Unlike SuperSloMo, this class accepts a numpy array of input frames and
    returns interpolated frames as a numpy array, avoiding all intermediate
    disk I/O.

    Parameters
    ----------
    model_path : str
        Path to the SuperSloMo PyTorch checkpoint.
    upsampling_factor : int
        Fixed upsampling factor (must be >= 2).
    batch_size : int
        Number of consecutive frame-pairs to process in one forward pass.
    """

    def __init__(
            self,
            model_path: str,
            upsampling_factor: int,
            batch_size: int = 1):

        if not isinstance(upsampling_factor, int) or upsampling_factor < 2:
            raise ValueError(
                f'upsampling_factor={upsampling_factor} must be an int >= 2')

        self.checkpoint = model_path
        self.upsampling_factor = upsampling_factor
        self.batch_size = batch_size

        if torch.cuda.is_available():
            self.device = 'cuda:0'
            logger.info('CUDA available, running on GPU')
        else:
            self.device = 'cpu'
            logger.warning('CUDA not available, running on CPU (slow)')

        # normalization constants mirrored from the original slomo.py
        self._mean = 0.428  # used by GPU normalisation
        self._to_tensor = self._build_to_tensor()

        self._flow_estimator = None
        self._warper = None
        self._interpolator = None
        self._loaded_dim = None  # dim the model was built for

    # ------------------------------------------------------------------
    # private helpers
    # ------------------------------------------------------------------

    def _build_to_tensor(self):
        mean = [self._mean]
        std = [1]
        normalize = transforms.Normalize(mean=mean, std=std)
        if self.device == 'cpu':
            return transforms.Compose([transforms.ToTensor()])
        return transforms.Compose([transforms.ToTensor(), normalize])

    def _load_model(self, dim):
        if not os.path.isfile(self.checkpoint):
            raise FileNotFoundError(
                f'SuperSloMo checkpoint {self.checkpoint!r} not found')
        logger.info(f'Loading SuperSloMo model from {self.checkpoint}')

        flow_est = model.UNet(2, 4).to(self.device)
        for p in flow_est.parameters():
            p.requires_grad = False

        interpolator = model.UNet(12, 5).to(self.device)
        for p in interpolator.parameters():
            p.requires_grad = False

        warper = model.backWarp(dim[0], dim[1], self.device).to(self.device)

        ckpt = torch.load(self.checkpoint, map_location=self.device, weights_only=False)
        interpolator.load_state_dict(ckpt['state_dictAT'])
        flow_est.load_state_dict(ckpt['state_dictFC'])

        self._flow_estimator = flow_est
        self._warper = warper
        self._interpolator = interpolator
        self._loaded_dim = dim

    def _tensor_to_uint8(self, t: torch.Tensor, ori_dim) -> np.ndarray:
        """Convert a [1, H, W] float tensor to a resized uint8 [H_out, W_out] numpy array.

        Handles the GPU normalisation offset so the output is in [0, 255].
        ori_dim is a (W, H) tuple (PIL/cv2 convention).
        """
        arr = t.cpu().detach()  # [1, H, W]
        if self.device != 'cpu':
            arr = arr + self._mean  # undo normalisation
        arr = arr.clamp(0.0, 1.0).squeeze(0).numpy()  # [H, W]  float32 in [0,1]
        arr = (arr * 255).astype(np.uint8)
        if (arr.shape[1], arr.shape[0]) != ori_dim:  # (W, H) comparison
            arr = cv2.resize(arr, ori_dim, interpolation=cv2.INTER_LINEAR)
        return arr

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def interpolate(self, frames: np.ndarray):
        """Interpolate between consecutive input frames.

        Parameters
        ----------
        frames : np.ndarray
            Greyscale input frames, shape ``[N, H, W]``, dtype uint8.

        Returns
        -------
        out_frames : np.ndarray
            Interpolated frames, shape ``[M, H, W]``, dtype uint8.
            M = sum of (upsampling_factor_i * num_batch_frames_i) over all
            batches, where upsampling_factor_i may vary per batch in
            auto_upsample mode.
        interp_times : np.ndarray
            Frame times expressed as fractions of the source frame interval.
            Multiply by the source frame interval (in seconds) to get absolute
            timestamps. Length equals M.
        avg_upsampling : float
            Mean upsampling factor across all batches.
        """
        if frames.ndim != 3:
            raise ValueError(f'frames must be [N, H, W], got shape {frames.shape}')

        N, H, W = frames.shape
        ori_dim = (W, H)  # PIL / cv2 (width, height)

        # Adjust batch size if the dataset is too small
        bs = self.batch_size
        while (N - 1) / bs < 1 and bs > 1:
            bs = max(1, bs // 2)
        if bs != self.batch_size:
            logger.warning(f'Reduced batch_size to {bs} for {N} input frames')

        dataset = dataloader.Frames(frames, transform=self._to_tensor)
        dim = dataset.dim  # (W_aligned, H_aligned) – multiple of 32

        loader = torch.utils.data.DataLoader(
            dataset, batch_size=bs, shuffle=False)

        # Lazy model load / reload if spatial size changed
        if self._loaded_dim != dim:
            self._load_model(dim)

        out_frames = []
        uf = self.upsampling_factor
        interp_times = None
        src_counter = 0

        with torch.no_grad():
            if len(loader) < 1:
                raise RuntimeError(
                    f'Need at least 1 batch. '
                    f'Increase input frames or reduce batch_size.')

            for frame0, frame1 in tqdm(loader, desc='slomo-interp', unit='batch'):
                I0 = frame0.to(self.device)
                I1 = frame1.to(self.device)
                B = I0.shape[0]  # actual frames in this batch

                flowOut = self._flow_estimator(torch.cat((I0, I1), dim=1))
                F_0_1 = flowOut[:, :2, :, :]
                F_1_0 = flowOut[:, 2:, :, :]

                # --- frame times for this batch ---
                num_out = uf * B
                dt = 1.0 / uf
                batch_times = src_counter + np.arange(num_out) * dt
                interp_times = (
                    batch_times if interp_times is None
                    else np.concatenate([interp_times, batch_times]))

                # Preallocate [B * uf, H, W] for this batch
                batch_buf = np.empty((num_out, H, W), dtype=np.uint8)

                # --- interpolation loop ---
                for k in range(uf):
                    t = (k + 0.5) / uf
                    tmp = -t * (1 - t)
                    fc = [tmp, t * t, (1 - t) * (1 - t), tmp]

                    F_t_0 = fc[0] * F_0_1 + fc[1] * F_1_0
                    F_t_1 = fc[2] * F_0_1 + fc[3] * F_1_0

                    g0 = self._warper(I0, F_t_0)
                    g1 = self._warper(I1, F_t_1)

                    intrp = self._interpolator(
                        torch.cat((I0, I1, F_0_1, F_1_0, F_t_1, F_t_0, g1, g0), dim=1))

                    F_t_0_f = intrp[:, :2, :, :] + F_t_0
                    F_t_1_f = intrp[:, 2:4, :, :] + F_t_1
                    V_t_0 = torch.sigmoid(intrp[:, 4:5, :, :])
                    V_t_1 = 1.0 - V_t_0

                    g0f = self._warper(I0, F_t_0_f)
                    g1f = self._warper(I1, F_t_1_f)

                    wc = [1 - t, t]
                    Ft_p = (wc[0] * V_t_0 * g0f + wc[1] * V_t_1 * g1f) / \
                           (wc[0] * V_t_0 + wc[1] * V_t_1)

                    # Batch GPU→CPU transfer for all frames at this interp step
                    frames_cpu = Ft_p.cpu().detach()          # [B, 1, h, w]
                    if self.device != 'cpu':
                        frames_cpu = frames_cpu + self._mean  # undo normalisation
                    frames_cpu = frames_cpu.clamp(0.0, 1.0).squeeze(1).numpy()  # [B, h, w]
                    frames_u8 = (frames_cpu * 255).astype(np.uint8)

                    for b in range(B):
                        arr = frames_u8[b]
                        if (arr.shape[1], arr.shape[0]) != ori_dim:
                            arr = cv2.resize(arr, ori_dim, interpolation=cv2.INTER_LINEAR)
                        batch_buf[uf * b + k] = arr

                out_frames.append(batch_buf)
                src_counter += B

        out_frames = np.concatenate(out_frames, axis=0)  # [M, H, W]
        logger.info(
            f'Returned {len(out_frames)} interpolated frames with '
            f'upsampling_factor={uf}')
        return out_frames, interp_times
