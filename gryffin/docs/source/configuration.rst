Configuration
=============

Gryffin provides a flexible configuration interface, it accepts either a path to a config.json file or a python dict. 


.. code-block:: python
    
    gryffin = Gryffin(config_file='/path/to/your/config.json')

.. code-block:: python

    gryffin = Gryffin(config_dict={})


Gryffin exposes 5 configurable modules, `general`, `database`, `model`, `parameters` and `objectives`.

.. code-block:: JSON   

    {
        "general": {},
        "database": {},
        "model": {},
        "parameters": [],
        "objectives": []
    }

.. code-block:: python

    config = {
        "general": {},
        "database": {},
        "model": {},
        "parameters": [],
        "objectives": []  
        
    }

General Configuration
---------------------

.. list-table::
    :header-rows: 1

    * - Parameter [type]
      - Definition
    * - num_cpus [int | string]
      - Number of CPUs to use, options are a number or 'all' (default: 1)
    * - boosted [bool]
      - Use kernel boosting (default: True)
    * - caching [bool]
      - Use kernel caching (default: True)
    * - auto_desc_gen [bool]
      - Use automatic descriptor generation (default: False)
    * - batches [int]
      - (default: 1)
    * - sampling_strategies [int]
      - (default: 2)
    * - softness [float]
      - Softness of Chimera for multiobj optimizations (default: 0.001)
    * - feas_approach [string]
      - Approach to unknown feasibility constraints, options are 'fwa' (feasibility-weighted acquisition), 'fca' (feasibility-constrained acquisition) or 'fia' (feasibility-interpolated acquisition). (default: 'fwa')
    * - feas_param [int]
      - Sensitivity to feasibility constraints (default: 1)
    * - dist_param [float]
      - Factor modulating density-based penalty in sample selector (default: 0.5)
    * - random_seed [None | int]
      - Set random seed (default: None)
    * - save_database [bool]
      - (default: False)
    * - aquisition_optimizer [string]
      - Set aquisition optimization method, options are 'adam' or 'genetic' (default: 'adam')
    * - obj_transform [None | string]
      - Set objective transform, options are None, 'sqrt', 'cbrt' or 'square' (default: 'sqrt')
    * - num_random_samples [int]
      - Number of samples per dimension to sample when optimizing acquisition function (default: 200)
    * - reject_tol [int]
      - Tolerance in rejection sampling, relevant when known constraints or fca used (default: 1000)
    * - vebosity [int]
      - Set verbosity level, from 0 to 5. 0: FATAL, 1: ERROR, 2: WARNING, 3: STATS, 4: INFO, 5: DEBUG (default: 4)

Database Configuration
----------------------

.. list-table::
    :header-rows: 1

    * - Parameter [type]
      - Definition
    * - format [string]
      - (default: 'sqlite')
    * - path [int]
      - (default: './SearchProgress')
    * - log_observations [bool]
      - (default: True)
    * - log_runtimes [bool]
      - (default: True)

Model Configuration
-------------------

.. list-table::
    :header-rows: 1

    * - Parameter
      - Definition
    * - num_epochs [int]
      - Number of training epochs (default: 2e3)
    * - learning_rate [float]
      - Model learning rate (default: 5e-2)
    * - num_draws [int]
      - (default: 1e3)
    * - num_layers [int]
      - Set the number of hidden layers in the model (default: 3)
    * - hidden_shape [int]
      - Set the dimensionality of the hidden layers (default: 6)
    * - weight_loc [float]
      - (default: 0.0)
    * - weight_scale [float]
      - (default: 1.0)
    * - bias_loc [float]
      - (default: 0.0)
    * - bias_scale [float]
      - (default: 1.0)
    

Parameters Configuration
------------------------

Gryffin supports 3 parameter types, `continuous`, `discrete` and `categorical`. Each parameter is configured as elements of the root level parameters list:

.. code-block:: JSON

    {
        "parameters": [
                {}   
        ]
    }

Continuous Parameters:

.. list-table::
    :header-rows: 1

    * - Parameter [type]
      - Definition
    * - name [string]
      - Human-readable parameter name 
    * - type [string]
      - Selects parameter type, either 'continuous', 'discrete' or 'categorical'
    * - low [float]
      - Lower bound of continuous parameter
    * - high [float]
      - Upper bound of continuous parameter. Note: high must be larger than low.
    * - periodic [bool]
      - Boolean flag indicating that the parameter is periodic

Discrete Parameters:

.. list-table::
    :header-rows: 1

    * - Parameter [type]
      - Definition
    * - name [string]
      - Human-readable parameter name 
    * - type [string]
      - Selects parameter type, either 'continuous', 'discrete' or 'categorical'
    * - low [float]
      - Lower bound of discrete parameter
    * - high [float]
      - Upper bound of continuous parameter. Note: high must be larger than low.
    * - options [List[]]
      - ToDo: Need explanation of options
    * - descriptors [List[]]
      - ToDo: Need explanation of descriptors

Categorical Parameters:

.. list-table::
    :header-rows: 1

    * - Parameter [type]
      - Definition
    * - name [string]
      - Human-readable parameter name 
    * - type [string]
      - Selects parameter type, either 'continuous', 'discrete' or 'categorical'
    * - options [List[]]
      - ToDo: Need explanation of options
    * - descriptors [List[]]
      - ToDo: Need explanation of descriptors
    * - category_details [List[]]
      - ToDo: Need explanation of category_details


Objective Configuration
-----------------------

Each objective is configured as elements of the root level objective list:

.. code-block:: JSON

    {
        "objectives": [
                {},      
        ]
    }

.. list-table::
    :header-rows: 1

    * - Parameter [type]
      - Definition
    * - name [string]
      - Human-readable parameter name 
    * - goal [string]
      - Optimization objective, options are 'min' or 'max'
    * - tolerance [float]
      - Termination tolerance on parameter changes
    * - absolute [bool]
      - Boolean flag indicating if objective is absolute



