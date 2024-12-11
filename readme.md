Structure of files:

Files with code
    - my_utils.py: has all the util files needed for the training in general
    - basic_steps_one.py: code example for applying the concept of SL operation with one split -- it shows just the basic operations between one client a server
    - basic_steps_two.py: code example for applying the concept of SL operation with two splits -- it shows just the basic operations between one client a server
    - /application: folder that contain an example implementation of a whole framework for SplitFed -- it does not contain networking
            - /application/model.py: implement a class for model (simple-cnn) that returns a model part of a model
            - /application/train.py: main code, i.e., server side 
            - /application/client.py: client side, note that this could be a standalone thread/process

Files with notes:
    - requirements.txt: important libraries
    - set_up_rpi.txt: notes on how to set-up RPI
    - set_up_jtson.txt: notes on how to set-up Jtson
    - git_list.txt: list of github repos



Other notes:

Dataloaders and partition: In order to build data loaders we need to define the type of data partition. For the following experiments we use IID, but in the my_utilis.py file,
the generate_partitioner() function contains other types of partiotions.
