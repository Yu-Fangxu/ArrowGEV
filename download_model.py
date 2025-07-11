from huggingface_hub import snapshot_download

base_model_dir = "/mnt/gemininjceph3/geminicephfs/pr-others-prctrans/fangxuyu/time-r1/ckpts/Time-R1-7B"
snapshot_download(repo_id="Boshenxx/Time-R1-7B", local_dir=base_model_dir)