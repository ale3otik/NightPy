This folder contains sample solution

Files:
    - create_model.ipynb - Jupyter Notebook example file with simple example of model creating/training/saving
    - predict_online.py - runnable client that load previously created model an do prediction
    - hackathon_protocol.py - implementation of net protocol to interact with check_solution_server.py.
        To use it:
            import hackathon_protocol
        Make sure, that file is in current directory

    Your prediction model should be implemented in predict_online.py.

How to start on local machine:
    1. Start check_solution_server.py in parent folder, wait until 'Server listening on port N' message appeared
    2. Run create_model.ipynb, after finish check for file 'my_model.txt'
    3. Run predict_online.py, after finish check output for score number:
    Completed! items_processed: 499781, time_elapsed: 123.262 sec, score: 0.189826

How to run with docker:
    1. Make sure check_solution_server.py is running
    2. Start run_solution_in_docker.py <directory with solution>

How to submit:
    TODO
