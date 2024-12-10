This folder contains an example implementation for a SplitFed framework.
Note that here each client has its own version of the first model part and they have a shared one at the server.
As a result at the end of each training round we need to aggregate the first model parts.