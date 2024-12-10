Structure of files:

Files with code
    - my_utils.py: has all the util files needed for the training in general
    - basic_steps_one.py: code example for applying the concept of SL operation with one split
    - basic_steps_two.py: code example for applying the concept of SL operation with two splits
    - /application: folder that contain an example implementation of a whole framework -- no networking
            - /application/model.py: implement a class for model (simple-cnn) that is split
            - /application/train.py: main code, i.e., server side 
            - /application/client.py: client side, note that this could be a standalone thread/process

Files with notes:
    - requirements.txt: important libraries
    - set_up_rpi.txt: notes on how to set-up RPI
    - set_up_jtson.txt: notes on how to set-up Jtson
    - git_list.txt: list of github repos



Other notes:

Dataloaders and partition: In order to build data loaders ...
