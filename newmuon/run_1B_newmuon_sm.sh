LOCAL_BATCH_SIZE=4
GRAD_ACC=16
NPROC=4
# total batchsize = LOCAL_BATCH_SIZE * GRAD_ACC * NPROC

TAG=1B_bf16_newmuon_sm_orthM_400k
PORT=13204
# export CUDA_VISIBLE_DEVICES=5
export CUDA_VISIBLE_DEVICES=4,5,6,7
# nohup python -m torch.distributed.launch --nproc_per_node 4 --master-port $PORT dp_main.py > /inspire/hdd/project/yunweiyuhuifu/p-shangli/quant/nohupout/$TAG.log \
nohup python -m torch.distributed.launch --nproc_per_node $NPROC --master-port $PORT dp_main.py > /inspire/hdd/project/yunweiyuhuifu/p-shangli/Metis-quantization-muon/nohupout/$TAG.log \
    --chkpt-dir /inspire/hdd/project/yunweiyuhuifu/p-shangli/Metis-quantization-muon/checkpoint \
    --dataset-path /inspire/hdd/global_user/p-shangli/DCLM-cleaned/ \
    --log-dir /inspire/hdd/project/yunweiyuhuifu/p-shangli/Metis-quantization-muon/log \
    --tokenizer-path /inspire/hdd/global_user/p-shangli/tokenizers/r50k_base.tiktoken \
    --tag $TAG \
    --reg-lambda 0 \
    --layers 32 \
    --embed-dim 1024 \
    --max-epochs 1 \
    --heads 32 \
    --lr-warmup-steps 50 \
    --q-forward-input fp32 \
    --q-forward-weight fp32 \
    --q-backward-input fp32 \
    --q-backward-weight fp32 \
    --q-backward-outputgrad fp32 \
    --grad-clipping 2.0 \
    --win-size 1024 \
    --forward-svd-warmup-steps 0 \
    --forward-svd-merge-steps -1 \
    --batch-size $LOCAL_BATCH_SIZE \
    --lr 1e-4 \
    --merged-lr 1e-4 \
    --grad-acc $GRAD_ACC \
    --train-steps 400000 \
    --optimizer_name newmuon_sm \
    --orthM True \
    --save-steps 500 \