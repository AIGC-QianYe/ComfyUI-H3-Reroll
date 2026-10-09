# -*- coding: utf-8 -*-
"""
H3 Reroll —— AV latent 的存 / 读（"抽卡"用）

Reroll = 一采阶段反复重抽（每次换 seed 重跑），把每次的 AV latent 存到磁盘；
         挑到满意的那次后，直接读回它的 latent 去放大 + 二采，不再重跑 SelfLift。

设计要点：
  - 格式自主（safetensors: video + audio），不依赖任何第三方插件
  - Save 是 passthrough（latent 原样往下传，可插在链上任意位置）
  - Load 返回完整 latent，**不做任何裁剪**
  - Save 的 IS_CHANGED 恒为 NaN（每次都真执行，否则新抽的那份不会写盘）
  - Load 的 IS_CHANGED 用文件 mtime + size 做指纹：文件没变就跳过重跑
"""

import os
import re

import torch
import safetensors.torch
import folder_paths

try:
    from comfy.nested_tensor import NestedTensor
    HAS_NESTED = True
except Exception:  # pragma: no cover
    HAS_NESTED = False

DEFAULT_SUBFOLDER = "H3一采抽卡"
FORMAT_TAG = "h3_av_latent_v1"


def _resolve_folder(folder: str) -> str:
    """把用户填的目录解析成绝对路径；留空则用默认 output/H3一采抽卡"""
    folder = str(folder or "").strip().strip('"')
    if not folder:
        folder = DEFAULT_SUBFOLDER
    if not os.path.isabs(folder):
        folder = os.path.join(folder_paths.get_output_directory(), folder)
    return folder


def _path_of(folder: str, scene_number: int) -> str:
    folder = _resolve_folder(folder)
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, "scene_%03d.latent" % int(scene_number))


def _next_free_number(folder: str) -> int:
    """扫描目录，返回第一个没有被占用的编号（用于自动编号模式）"""
    root = _resolve_folder(folder)
    used = set()
    if os.path.isdir(root):
        for name in os.listdir(root):
            m = re.match(r"^scene_(\d+)\.latent$", name, re.IGNORECASE)
            if m:
                used.add(int(m.group(1)))
    n = 1
    while n in used:
        n += 1
    return n


def _split_av(samples):
    """把 H3 的 AV latent 拆成 (video, audio)"""
    if getattr(samples, "is_nested", False):
        parts = tuple(samples.unbind())
        video = parts[0] if len(parts) > 0 else None
        audio = parts[1] if len(parts) > 1 else None
        return video, audio
    return samples, None


class H3RerollSave:
    """存一采的 AV latent 到磁盘（passthrough，不影响主链数据流）"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "folder": ("STRING", {
                    "default": DEFAULT_SUBFOLDER,
                    "tooltip": "存放目录。相对路径基于 ComfyUI 的 output 文件夹；留空则用默认。",
                }),
                "scene_number": ("INT", {
                    "default": 1, "min": 1, "max": 9999, "step": 1,
                    "tooltip": "抽卡编号。自动编号关闭时用它：同一个编号再跑会覆盖自己那一份。",
                }),
            },
            "optional": {
                "auto_next": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "开 = 自动找下一个没用过的编号，永不覆盖（适合一次抽很多份、回头慢慢挑）。\n"
                        "关 = 用上面填的编号，同名覆盖（适合「抽一张看一张、不满意就重抽」的循环，磁盘永远只留一份）。"
                    ),
                }),
            },
        }

    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("latent", "saved_path")
    FUNCTION = "save"
    CATEGORY = "H3 Reroll"

    @classmethod
    def IS_CHANGED(cls, latent, folder, scene_number, auto_next=False):
        # 存盘动作每次都要真执行（否则会被缓存跳过，导致新抽的那份没写进磁盘）
        return float("NaN")

    def save(self, latent, folder, scene_number, auto_next=False):
        if auto_next:
            scene_number = _next_free_number(folder)
            print("[H3 Reroll] auto-number -> scene_%03d" % int(scene_number))
        samples = latent.get("samples")
        video, audio = _split_av(samples)
        if not isinstance(video, torch.Tensor) or video.ndim != 5:
            print("[H3 Reroll] skip: not a 5D video latent")
            return (latent, "")

        path = _path_of(folder, scene_number)
        payload = {"video": video.detach().to("cpu").contiguous()}
        if isinstance(audio, torch.Tensor) and audio.numel() > 0:
            payload["audio"] = audio.detach().to("cpu").contiguous()

        safetensors.torch.save_file(
            payload, path,
            metadata={"format": FORMAT_TAG, "scene": str(int(scene_number))},
        )
        print("[H3 Reroll] saved scene_%03d -> %s" % (int(scene_number), path))
        return (latent, path)


class H3RerollLoad:
    """读回完整的 AV latent（不裁剪），喂给放大器 / 二采"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder": ("STRING", {"default": DEFAULT_SUBFOLDER}),
                "scene_number": ("INT", {
                    "default": 1, "min": 1, "max": 9999, "step": 1,
                    "tooltip": "要读哪一次抽卡（填你挑中的那个编号）。",
                }),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "load"
    CATEGORY = "H3 Reroll"

    @classmethod
    def IS_CHANGED(cls, folder, scene_number):
        path = _path_of(folder, scene_number)
        if not os.path.isfile(path):
            return float("NaN")  # 文件不存在：总是重跑（进而报错提示）
        st = os.stat(path)
        return "%d:%d" % (int(st.st_mtime), int(st.st_size))

    def load(self, folder, scene_number):
        path = _path_of(folder, scene_number)
        if not os.path.isfile(path):
            raise FileNotFoundError(
                "找不到 latent 文件：%s\n（先用「H3 Reroll · Save Latent」跑一次这个编号）" % path
            )

        tensors = safetensors.torch.load_file(path, device="cpu")
        video = tensors.get("video")
        if not isinstance(video, torch.Tensor) or video.ndim != 5:
            raise ValueError("latent 文件损坏或格式不对：%s" % path)

        audio = tensors.get("audio")
        if HAS_NESTED:
            if isinstance(audio, torch.Tensor) and audio.numel() > 0:
                samples = NestedTensor((video, audio))
            else:
                # 没有音轨时用空音频占位，满足 H3 的 AV 联合结构
                empty = torch.zeros(
                    (int(video.shape[0]), 32, 2, 1), dtype=video.dtype
                )
                samples = NestedTensor((video, empty))
        else:  # pragma: no cover
            samples = video

        print("[H3 Reroll] loaded scene_%03d <- %s  video=%s"
              % (int(scene_number), path, tuple(video.shape)))
        return ({"samples": samples},)


NODE_CLASS_MAPPINGS = {
    "H3RerollSave": H3RerollSave,
    "H3RerollLoad": H3RerollLoad,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "H3RerollSave": "H3 Reroll · Save Latent",
    "H3RerollLoad": "H3 Reroll · Load Latent",
}
