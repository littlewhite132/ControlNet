from share import *

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from tutorial_dataset import MyDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict


# Configs
resume_path = '/media/NAS01/yyfangspx/stable-diffusion_generate/logs/2025-12-25T14-02-50_ct_generate-kl-8/checkpoints/last.ckpt'
batch_size = 4
logger_freq = 300
learning_rate = 1e-5
sd_locked = True
only_mid_control = False
max_epochs = 50


def main():
    # First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
    # 使用与自训 SD 对齐、且指向本地 CLIP 的配置
    model = create_model('./models/cldm_ct.yaml').cpu()
    # 允许从仅包含部分模块的 checkpoint 加载（例如无 ControlNet 权重）；对形状不匹配的权重直接跳过
    ckpt_state = load_state_dict(resume_path, location='cpu')
    model_state = model.state_dict()
    filtered_state = {}
    for k, v in ckpt_state.items():
        if k in model_state:
            target_shape = model_state[k].shape
            # 如果只差一个末尾的维度为1（如 Conv2d 权重需要 [out, in, 1,1]，ckpt 为 [out, in,1]），则自动补齐
            if v.shape != target_shape and v.ndim + 1 == len(target_shape) and target_shape[-1] == 1:
                v = v.unsqueeze(-1)
            if v.shape == target_shape:
                filtered_state[k] = v
            else:
                print(f"[load_state_dict] skip key: {k} due to shape mismatch {v.shape} vs {target_shape}")
        else:
            print(f"[load_state_dict] skip key: {k} because it does not exist in current model")
    model.load_state_dict(filtered_state, strict=False)
    model.learning_rate = learning_rate
    model.sd_locked = sd_locked
    model.only_mid_control = only_mid_control

    # Datasets & loaders
    train_dataset = MyDataset(split="train")
    val_dataset = MyDataset(split="val")

    train_dataloader = DataLoader(train_dataset, num_workers=0, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, num_workers=0, batch_size=batch_size, shuffle=False)

    # Callbacks
    logger = ImageLogger(batch_frequency=logger_freq)
    checkpoint = ModelCheckpoint(
        monitor="val/loss",
        mode="min",
        save_top_k=3,
        save_last=True,
        every_n_epochs=1
    )

    # Trainer
    trainer = pl.Trainer(
        gpus=1,
        precision=32,
        callbacks=[logger, checkpoint],
        max_epochs=max_epochs,
        check_val_every_n_epoch=1
    )

    # Train!
    trainer.fit(model, train_dataloader, val_dataloader)


if __name__ == "__main__":
    main()
