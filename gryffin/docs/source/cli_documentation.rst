Gryffin CLI
============

``Gryffin`` also exposes a CLI interface that makes integrating the package into your workflow even more flexible. To run ``Gryffin`` from the command line simply run the command:

.. code-block:: console

    $ gryffin -f FILE -c JSON


There are two required argurments to run ``Gryffin`` from the command line. A filepath to an Excel or CSV file with all previous experiments must be provided and the usual ``Gryffin`` configuration file must be provided. Please refer to the Configuration section for the details of all configuration parameters.

Required arguments:
  -f                Excel/CSV file with all previous experiments.
  -c                Json configuration file with parameters and objectives.

The ``Gryffin`` CLI also exposes a number of optional argurments. For convenince, these arguments expose over-writable ``Gryffin`` configuration.

Optional arguments:
  -h, --help        show this help message and exit
  -n                Number of experiments to suggest. Default is 1. Note that Gryffin will alternate between exploration and exploitation.
  --num_cpus        Number of CPUs to use. Default is 1.
  --optimizer       Algorithm to use to optimize the acquisition function. Choices are "adam" or "genetic". Default is "adam".
  --dynamic         Whether to use dynamic Gryffin. Default is False.
  --feas_approach   Approach to unknown feasibility constraints. Choices are: "fwa" (feasibility-weighted acquisition), "fca" (feasibility-constrained acquisition), "fia" (feasibility-interpolated
                    acquisition). Default is "fia".
  --boosted         Whether to use boosting. Default is False.
  --cached          Whether to use caching. Default is False.
  --seed            Random seed used for initialization. Default is 42.