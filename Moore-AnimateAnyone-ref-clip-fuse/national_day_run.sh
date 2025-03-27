# CUDA_VISIBLE_DEVICES=1 python -m scripts.pose2vid --config /home/xk/pht/Moore-AnimateAnyone-ref-clip-fuse/configs/prompts/phtanimate3.yaml -W 512 -H 784 -L 64 --alpha 0.1 --mode motion_fuse_2
# CUDA_VISIBLE_DEVICES=1 python -m scripts.pose2vid --config /home/xk/pht/Moore-AnimateAnyone-ref-clip-fuse/configs/prompts/phtanimate3.yaml -W 512 -H 784 -L 64 --alpha 0.3 --mode motion_fuse_3
# CUDA_VISIBLE_DEVICES=1 python -m scripts.pose2vid --config /home/xk/pht/Moore-AnimateAnyone-ref-clip-fuse/configs/prompts/phtanimate3.yaml -W 512 -H 784 -L 64 --alpha 0.4 --mode motion_fuse_4

CUDA_VISIBLE_DEVICES=1  python -m scripts.pose2vid --config /home/xk/pht/Moore-AnimateAnyone-ref-clip-fuse/configs/prompts/phtanimate3.yaml -W 512 -H 784 -L 64 --alpha 0.5 --beta 0.8 --mode line_ref_5_motion_2

CUDA_VISIBLE_DEVICES=1  python -m scripts.pose2vid --config /home/xk/pht/Moore-AnimateAnyone-ref-clip-fuse/configs/prompts/phtanimate3.yaml -W 512 -H 784 -L 64 --alpha 0.9 --beta 0.8 --mode show_ref_9_motion_2