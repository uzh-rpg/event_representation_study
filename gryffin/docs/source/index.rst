Gryffin: Bayesian Optimization of Continuous and Categorical Variables
======================================================================

.. image:: https://github.com/aspuru-guzik-group/gryffin/actions/workflows/continuous-integration.yml/badge.svg
    :target: https://github.com/aspuru-guzik-group/gryffin/actions/workflows/continuous-integration.yml
.. image:: https://codecov.io/gh/aspuru-guzik-group/gryffin/branch/master/graph/badge.svg?token=pHQ8Z50qf8
    :target: https://codecov.io/gh/aspuru-guzik-group/gryffin

Welcome to **Gryffin**!

Designing functional molecules and advanced materials requires complex design choices: tuning
continuous process parameters such as temperatures or flow rates, while simultaneously selecting
catalysts or solvents. 

To date, the development of data-driven experiment planning strategies for
autonomous experimentation has largely focused on continuous process parameters despite the urge
to devise efficient strategies for the selection of categorical variables. Here, we introduce Gryffin,
a general purpose optimization framework for the autonomous selection of categorical variables
driven by expert knowledge.

Features
--------

* Gryffin extends the ideas of the `Phoenics <https://pubs.acs.org/doi/10.1021/acscentsci.8b00307>`_ optimizer to categorical variables. Phoenics is a linear-scaling Bayesian optimizer for continuous spaces which uses a kernel regression surrogate. Gryffin extends this approach to categorical and mixed continuous-categorical spaces. 
* Gryffin is linear-scaling appraoch to Bayesian optimization, whose acquisition function natively supports batched optimization. Gryffin's acquisition function uses an intuitive sampling parameter to bias its behaviour between exploitation and exploration. 
* Gryffin is capable of leveraging expert knowledge in the form of physicochemical descriptors to enhance its optimization performance (static formulation). Also, Gryffin can refine the provided descriptors to further accelerate the optimization (dynamic formulation) and foster scientific understanding. 

Use cases of Gryffin/Phoenics
-----------------------------

* `Self-driving lab to optimize multicomponet organic photovoltaic systems <https://onlinelibrary.wiley.com/doi/full/10.1002/adma.201907801>`_
* `Self-driving laboratory for accelerated discovery of thin-film materials <https://www.science.org/doi/10.1126/sciadv.aaz8867>`_
* `Data-science driven autonomous process optimization <https://www.nature.com/articles/s42004-021-00550-x>`_ 
* `Self-driving platform for metal nanoparticle synthesis <https://onlinelibrary.wiley.com/doi/full/10.1002/adfm.202106725>`_
* `Optimization of photophyscial properties of organic dye laser molecules <https://pubs.acs.org/doi/10.1021/acscentsci.1c01002>`_

.. toctree::
   :maxdepth: 2
   :caption: Contents

   getting_started
   configuration
   tutorial
   api_documentation
   cli_documentation
   citation

License
-------
**Gryffin** is distributed under an Apache Licence 2.0.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`






