1. Install the [Google Cloud CLI](https://cloud.google.com/sdk/docs/install)
2. Request access to the [RAIL TPU Project](https://console.cloud.google.com/compute/tpus?project=rail-tpus&pli=1)
3. Check free TPUs at http://tpus.kevin.black
4. Login into a TPU with ```gcloud alpha compute tpus tpu-vm ssh [insert_tpu_name] --zone=[insert_tpu_location]```
5. Create personal bucket on the same region to store checkpoints
6. Clone and install the Orca repo at ```/nfs/nfs(1,2)/users/[your_name]/orca``` (mount nfs if needed with instructions below)
7. ```python train.py --config config.py:transformer_bc_bridge --name=orca_bridge --config.dataset_kwargs.data_kwargs_list[0].data_dir=gs://your_dataset --config.save_dir=gs://your_bucket```

### TPU Troubleshooting
If you encounter an error of ```TpuStatesManager::GetOrCreate(): no tpu system exists``` reboot and mount nfs:
1. ```sudo reboot now```
2. ```sudo apt -y update && sudo apt install nfs-common```
3. For central1 location ```sudo mkdir -p -m 777 /nfs/nfs1``` and for central2 ```sudo mkdir -p -m 777 /nfs/nfs2```
4. For central1 location ```sudo mount -o rw,intr 10.244.23.202:/nfs1 /nfs/nfs1``` and for central2 ```sudo mount -o rw,intr 10.30.175.26:/nfs2 /nfs/nfs2```

If you get an error ```tensorflow.python.framework.errors_impl.PermissionDeniedError: Error executing an HTTP request: HTTP response code 403 with body``` you might need to authenticate with ```gcloud auth application-default login```

Also, remember that the dataset, bucket and TPU need to be in the same region.
