1. Clone repo.
2. Install requirements.
3. Put data into `data/` directory.
4. Split data on train and validate:
    ```commandline
    python ./split_train_val.py    
    ```
5. Train ResNet-20 model:
    ```commandline
    python ./train.py    
    ```
   use `--use-cuda` if you need.
6. Create kaggle submission:
   ```commandline
    python ./create_submission.py    
   ```
