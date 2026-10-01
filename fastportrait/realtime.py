"""Real-time webcam driver for LivePortrait.

Why this instead of the shipped `inference.py`
----------------------------------------------
`inference.py` is built around file inputs and re-derives the source every run.
For a live session the work splits very unevenly:

  ONCE per source image   crop, appearance feature extraction (f_s), source
                          keypoints -- the expensive part
  PER driving frame       crop the face, motion extraction, warp + SPADE decode,
                          paste back -- a few ms

So this class does the source work once in `set_source()` and keeps only the
per-frame path in `process()`. Measured on an RTX 4080 Laptop the per-frame path
is a couple of milliseconds for the neural part, which is what makes live
animation at webcam rates possible at all -- and is the whole reason this project
exists next to PersonaLive (a 4-step diffusion model at ~5 FPS).

Everything here delegates to LivePortrait's own `LivePortraitWrapper` and
`Cropper`; nothing about the model is reimplemented.
"""
from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch

# LivePortrait's package root must be importable; its configs use relative paths.
_LP_ROOT = Path(__file__).resolve().parent.parent / "LivePortrait"
if str(_LP_ROOT) not in sys.path:
    sys.path.insert(0, str(_LP_ROOT))

from src.config.crop_config import CropConfig                 # noqa: E402
from src.config.inference_config import InferenceConfig       # noqa: E402
from src.live_portrait_pipeline import LivePortraitPipeline    # noqa: E402
from src.utils.camera import get_rotation_matrix               # noqa: E402
from src.utils.crop import (crop_image, prepare_paste_back,     # noqa: E402
                            paste_back)
from src.utils.helper import dct2device                       # noqa: E402


@dataclass
class FastConfig:
    """Tunables that matter for a live session."""

    device_id: int = 0
    # FP16 halves the SPADE decoder cost and is what the 14.8 ms/frame figure was
    # measured with.
    half: bool = True
    # torch.compile costs a long first call but the paper measured 14.8 ms/frame
    # (vs ~25 ms eager) on a 4090-class GPU with it. Worth it for a live session,
    # bad for a one-shot call, so it is opt-in. NOTE: not usable on Windows --
    # torch 2.1's dynamo backend rejects this code path.
    torch_compile: bool = False
    # Motion smoothing across frames. LivePortrait's own video path uses a Kalman
    # filter for exactly this; without it the output jitters on camera noise.
    smooth: bool = True
    smooth_variance: float = 3e-7
    # "all" | "exp" | "pose" | "lip" | "eyes"
    animation_region: str = "all"
    # Keep the source's own pose instead of copying the driver's (useful when the
    # source portrait is a nice front-facing shot and you only want expressions).
    relative_motion: bool = True
    # Amplify (or damp) the transferred motion. LivePortrait's own video path applies
    #     x_d_i_new = x_s + (x_d_i_new - x_s) * driving_multiplier
    # (src/live_portrait_pipeline.py line 439) and the first version of this file
    # simply never did.
    #
    # Default is 0.7, not 1.0: at 1.0 the transferred expression reads as exaggerated
    # on real camera input (reported). Below 1.0 damps it, which looks calmer.
    driving_multiplier: float = 0.7
    stitching: bool = True
    # Paste the animated head back into the driving frame.
    #
    # OFF is the correct mode for a live portrait, and it is both simpler and much
    # faster. LivePortrait animates *the source portrait*; pasting that into the
    # webcam frame only makes sense when the two share a framing, and when they do
    # not it produces a visible rectangular seam of the source's own background
    # (measured: a 682x1023 portrait pasted into a 1280x720 frame showed the
    # portrait's background as a rectangle). With pasteback off, the output is the
    # animated source -- the intended product -- and the ~23 ms paste-back cost
    # disappears.
    pasteback: bool = False
    # crop / detect
    det_thresh: float = 0.1
    scale: float = 2.3
    # below this the face is too small to animate sensibly
    min_face_px: int = 64


@dataclass
class FrameStats:
    detect_ms: float = 0.0
    crop_ms: float = 0.0
    motion_ms: float = 0.0
    warp_ms: float = 0.0
    paste_ms: float = 0.0
    total_ms: float = 0.0
    applied: bool = False
    note: str = ""

    def report(self) -> str:
        if not self.applied:
            return f"skipped ({self.note or 'no reason'}) {self.total_ms:.1f}ms"
        return (f"{self.total_ms:5.1f}ms [det {self.detect_ms:4.1f} "
                f"crop {self.crop_ms:4.1f} motion {self.motion_ms:4.1f} "
                f"warp {self.warp_ms:4.1f} paste {self.paste_ms:4.1f}]")


class LivePortraitFast:
    """Animate a source portrait with a live webcam feed."""

    def __init__(self, config: FastConfig | None = None) -> None:
        self.cfg = config or FastConfig()
        self.device = f"cuda:{self.cfg.device_id}" if torch.cuda.is_available() else "cpu"

        inf = InferenceConfig(
            device_id=self.cfg.device_id,
            flag_use_half_precision=self.cfg.half and self.device != "cpu",
            flag_do_torch_compile=self.cfg.torch_compile,
            flag_relative_motion=self.cfg.relative_motion,
            flag_stitching=self.cfg.stitching,
            flag_pasteback=self.cfg.pasteback,
            flag_do_crop=True,
            animation_region=self.cfg.animation_region,
            driving_smooth_observation_variance=self.cfg.smooth_variance,
        )
        crop = CropConfig(device_id=self.cfg.device_id, det_thresh=self.cfg.det_thresh,
                          scale=self.cfg.scale)
        self.pipe = LivePortraitPipeline(inference_cfg=inf, crop_cfg=crop)
        self.wrapper = self.pipe.live_portrait_wrapper
        self.cropper = self.pipe.cropper

        # source state, filled by set_source()
        self.f_s: torch.Tensor | None = None
        self.x_s: torch.Tensor | None = None
        self.x_s_info: dict | None = None
        self.x_c_s: torch.Tensor | None = None
        self.R_s: torch.Tensor | None = None
        self.source_rgb: np.ndarray | None = None
        self.source_M_c2o: np.ndarray | None = None
        self.source_lmk: np.ndarray | None = None
        self.mask_ori: np.ndarray | None = None
        # Paste-back mask for the current driving frame size. It only depends on the
        # frame size, so it is built once per resolution instead of every frame
        # (~10 ms of the ~35 ms paste-back).
        self._mask_d: np.ndarray | None = None
        self._mask_key: tuple[int, int] | None = None
        # first driving frame anchors relative motion
        self.R_d_0: torch.Tensor | None = None
        self.x_d_0_info: dict | None = None
        # last successfully animated frame, held while no face is visible
        self._last_output: np.ndarray | None = None
        # what the driver's last frame looked like, for the UI readout
        self.last_pose: dict = {}
        self.last_motion: float = 0.0
        self.last = FrameStats()

    # -- source ------------------------------------------------------------
    def set_source(self, source_bgr: np.ndarray) -> bool:
        """Prepare the portrait to animate. Call once per source image."""
        h, w = source_bgr.shape[:2]
        # LivePortrait works on 256x256 crops; a huge source just slows cropping.
        maxdim = max(h, w)
        if maxdim > 1280:
            s = 1280.0 / maxdim
            source_bgr = cv2.resize(source_bgr, (int(w * s), int(h * s)),
                                    interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2RGB)

        crop_info = self.cropper.crop_source_image(rgb, self.cropper.crop_cfg)
        if crop_info is None:
            return False
        self.source_rgb = rgb
        self.source_M_c2o = crop_info["M_c2o"]
        self.source_lmk = crop_info["lmk_crop"]

        I_s = self.wrapper.prepare_source(crop_info["img_crop_256x256"])
        # Every model forward must run inside the wrapper's autocast context.
        #
        # `flag_use_half_precision` is not applied automatically -- it is only read
        # by `LivePortraitWrapper.inference_ctx()`, which returns a
        # torch.autocast(float16) block that callers must enter. Omitting it (as the
        # first version of this file did) silently runs the whole pipeline in fp32,
        # roughly doubling the cost.
        with self.wrapper.inference_ctx():
            self.x_s_info = self.wrapper.get_kp_info(I_s)
            self.x_c_s = self.x_s_info["kp"]
            self.R_s = get_rotation_matrix(self.x_s_info["pitch"], self.x_s_info["yaw"],
                                           self.x_s_info["roll"])
            self.f_s = self.wrapper.extract_feature_3d(I_s)
            self.x_s = self.wrapper.transform_keypoint(self.x_s_info)
        # fp32 from here on: the cached source tensors are reused every frame and
        # must not stay fp16, or they collide with the fp32 rotation maths.
        self.x_s_info = {k: (v.float() if torch.is_tensor(v) else v)
                         for k, v in self.x_s_info.items()}
        self.x_s = self.x_s.float()
        self.R_s = self.R_s.float()
        self.f_s = self.f_s.float()
        if self.cfg.pasteback:
            self.mask_ori = prepare_paste_back(
                self.wrapper.inference_cfg.mask_crop, self.source_M_c2o,
                dsize=(rgb.shape[1], rgb.shape[0]))
        # relative motion is computed against the first driving frame
        self.R_d_0 = None
        self.x_d_0_info = None
        return True

    # -- per frame ---------------------------------------------------------
    def process(self, frame_bgr: np.ndarray,
                face=None) -> np.ndarray:
        """Animate `frame_bgr` into the source portrait's appearance."""
        t0 = time.perf_counter()
        st = FrameStats()
        if self.f_s is None:
            st.note = "no source"
            st.total_ms = (time.perf_counter() - t0) * 1000
            self.last = st
            return frame_bgr

        t = time.perf_counter()
        inf = self.wrapper.inference_cfg
        device = self.wrapper.device
        cfg = self.cropper.crop_cfg

        # 1. detect + crop the driving frame exactly as the source was cropped.
        #
        # There is no `crop_driving_image` helper, so this mirrors
        # `Cropper.crop_source_image`: detect with 106-point landmarks, then
        # `crop_image` with the same cfg. Using identical parameters is what makes
        # the paste-back matrix valid. Detection IS the first half of cropping, so
        # it is not repeated as a separate step.
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        faces = self.cropper.face_analysis_wrapper.get(
            frame_bgr, flag_do_landmark_2d_106=True,
            direction=cfg.direction, max_face_num=1)
        st.detect_ms = (time.perf_counter() - t) * 1000
        if not faces or getattr(faces[0], "landmark_2d_106", None) is None:
            # No face in the driver frame -- someone covered the lens, or looked away.
            # Returning the driver frame here makes the whole output flip to the raw
            # webcam, which looks like a crash. Hold the last animated frame instead,
            # so the portrait simply freezes for those frames.
            st.note = "no face - holding last frame"
            st.total_ms = (time.perf_counter() - t0) * 1000
            self.last = st
            return self._last_output if self._last_output is not None else frame_bgr
        lmks = faces[0].landmark_2d_106
        bw = float(lmks[:, 0].max() - lmks[:, 0].min())
        bh = float(lmks[:, 1].max() - lmks[:, 1].min())
        if min(bw, bh) < self.cfg.min_face_px:
            st.note = "face too small"
            st.total_ms = (time.perf_counter() - t0) * 1000
            self.last = st
            return frame_bgr

        t = time.perf_counter()
        dct = crop_image(rgb, lmks, dsize=cfg.dsize,
                         scale=cfg.scale, vx_ratio=cfg.vx_ratio,
                         vy_ratio=cfg.vy_ratio, flag_do_rot=cfg.flag_do_rot)
        img_crop_256 = cv2.resize(dct["img_crop"], (256, 256),
                                  interpolation=cv2.INTER_AREA)
        M_c2o = dct["M_c2o"]
        st.crop_ms = (time.perf_counter() - t) * 1000

        # 2. motion + warp/decode
        t = time.perf_counter()
        # `prepare_videos` expects a LIST (it appends a trailing axis itself, giving
        # TxHxWx3x1 -> Tx1x3xHxW); passing an ndarray makes it 4-D and the permute
        # fails. Then take the single frame back out as 1x3xHxW.
        I_d = self.wrapper.prepare_videos([img_crop_256])[0]
        # autocast, as in set_source -- see the note there
        with self.wrapper.inference_ctx():
            x_d_i_info = self.wrapper.get_kp_info(I_d)
        # Back to fp32 for all the arithmetic below.
        #
        # The model outputs come out as fp16 (autocast), while the cached source
        # tensors and the rotation matrices are fp32. Mixing them raises "expected
        # scalar type Half but found Float" -- so normalise right after the forward
        # pass and keep the maths single-precision. It is a handful of scalar ops.
        x_d_i_info = {k: (v.float() if torch.is_tensor(v) else v)
                      for k, v in x_d_i_info.items()}
        R_d_i = get_rotation_matrix(x_d_i_info["pitch"], x_d_i_info["yaw"],
                                    x_d_i_info["roll"])
        st.motion_ms = (time.perf_counter() - t) * 1000

        if self.R_d_0 is None:
            self.R_d_0 = R_d_i
            self.x_d_0_info = {k: (v.float() if torch.is_tensor(v) else v)
                               for k, v in x_d_i_info.items()}

        delta_new = self.x_s_info["exp"].clone()
        R_new = self.R_s
        if inf.flag_relative_motion:
            if inf.animation_region in ("all", "pose"):
                # Relative head rotation: driver-now vs driver-at-t0, composed with the
                # source's own pose.
                R_new = (R_d_i @ self.R_d_0.permute(0, 2, 1)) @ self.R_s
            if inf.animation_region in ("all", "exp"):
                delta_new = self.x_s_info["exp"] + (x_d_i_info["exp"] - self.x_d_0_info["exp"])
        else:
            R_new = R_d_i
            delta_new = x_d_i_info["exp"]

        # THE BUG THAT MADE HEAD TURNS INVISIBLE.
        #
        # `x_s` already went through transform_keypoint(), i.e. it is
        #     scale * (kp @ R_s + exp_s) + t_s
        # so adding an expression delta to it only ever moves the mouth/eyes. The
        # driver's rotation matrix R_new was computed above and then never used, so
        # pose never reached the network at all.
        #
        # Measuring this: rendering with R_new actually applied rotates the head
        # cleanly to at least yaw 60 / pitch 45 (see docs/testD_deg.jpg), whereas the
        # old code changed the output by only 0.95/255 across a 40 degree yaw change.
        #
        # So rebuild the driving keypoints the same way transform_keypoint does, but
        # with the driver's rotation and expression:
        #     Eqn.2:  s * (R * x_c + exp) + t
        kp_src = self.x_s_info["kp"]
        exp_drv = self.x_s_info["exp"] + (
            (x_d_i_info["exp"] - self.x_d_0_info["exp"])
            if inf.flag_relative_motion else x_d_i_info["exp"]
        )
        kp_new = kp_src @ R_new + exp_drv.view(kp_src.shape)
        kp_new = kp_new * self.x_s_info["scale"][..., None]
        kp_new[:, :, 0:2] += self.x_s_info["t"][:, None, 0:2]
        x_d_i_new = kp_new

        if inf.flag_stitching:
            x_d_i_new = self.wrapper.stitching(self.x_s, x_d_i_new)

        # Scale the transferred motion, matching LivePortrait's own pipeline. Without
        # this the pose delta is applied at 1.0 and motion reads as exaggerated.
        if self.cfg.driving_multiplier != 1.0:
            x_d_i_new = self.x_s + (x_d_i_new - self.x_s) * self.cfg.driving_multiplier

        # remember what was detected, so the UI can show the live pose/expression
        try:
            self.last_pose = {
                "pitch": float(x_d_i_info["pitch"].item()),
                "yaw": float(x_d_i_info["yaw"].item()),
                "roll": float(x_d_i_info["roll"].item()),
                "scale": float(x_d_i_info["scale"].item()),
            }
            if self.x_d_0_info is not None:
                de = (x_d_i_info["exp"] - self.x_d_0_info["exp"])
                self.last_motion = float(de.abs().mean().item())
        except Exception:
            pass

        t = time.perf_counter()
        with self.wrapper.inference_ctx():
            out = self.wrapper.warp_decode(self.f_s, self.x_s, x_d_i_new)
        parsed = self.wrapper.parse_output(out["out"])
        # parse_output returns 1xHxWx3 uint8; we want a single HxWx3 image
        I_p = parsed[0] if isinstance(parsed, np.ndarray) and parsed.ndim == 4 else parsed
        st.warp_ms = (time.perf_counter() - t) * 1000

        # 3. paste back into the driving frame.
        #
        # The mask must be built for the DRIVING frame's geometry. The source and
        # driving frames generally differ in resolution (a 682x1023 portrait being
        # driven by a 1280x720 webcam frame), and reusing the source-sized mask
        # raises "could not be broadcast (1023,682,3) (720,1280,3)".
        #
        # It depends only on the frame SIZE and the crop matrix, so it is cached per
        # (w, h). Rebuilding it measured ~10 ms of the ~35 ms paste-back, for an
        # identical result on every frame at a fixed webcam resolution.
        t = time.perf_counter()
        if self.cfg.pasteback:
            key = (frame_bgr.shape[1], frame_bgr.shape[0])
            if self._mask_key != key:
                self._mask_d = prepare_paste_back(
                    self.wrapper.inference_cfg.mask_crop, M_c2o,
                    dsize=(frame_bgr.shape[1], frame_bgr.shape[0]))
                self._mask_key = key
            I_p = paste_back(I_p, M_c2o, frame_bgr, self._mask_d)
        st.paste_ms = (time.perf_counter() - t) * 1000
        # LivePortrait's whole pipeline works in RGB (the crop step converts BGR->RGB
        # on the way in). The output therefore has to be converted back before it is
        # treated as BGR, or every consumer that assumes BGR -- cv2.imwrite, the Qt
        # preview, the Deep-Live-Cam frame loop -- shows a severe blue cast.
        I_p = cv2.cvtColor(I_p, cv2.COLOR_RGB2BGR)

        st.applied = True
        st.total_ms = (time.perf_counter() - t0) * 1000
        self.last = st
        # kept so the next no-face frame can hold this instead of showing the webcam
        self._last_output = I_p
        return I_p

    def warmup(self, runs: int = 3) -> float:
        """Run the per-frame path on a synthetic frame to absorb first-call cost."""
        t0 = time.perf_counter()
        canvas = np.full((720, 1280, 3), 96, dtype=np.uint8)
        for _ in range(max(1, runs)):
            try:
                self.process(canvas)
            except Exception:
                pass
        self.R_d_0 = None
        self.x_d_0_info = None
        return (time.perf_counter() - t0) * 1000.0
