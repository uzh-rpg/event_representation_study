Citation
--------

If you use **Gryffin** in scientific publications, please cite the following papers depending on which aspects of the
code you used.

If you optimized **continuous variables**, please cite `this publication <https://pubs.acs.org/doi/abs/10.1021/acscentsci.8b00307>`_:

::

    @article{phoenics,
      title = {Phoenics: A Bayesian Optimizer for Chemistry},
      author = {Florian Häse and Loïc M. Roch and Christoph Kreisbeck and Alán Aspuru-Guzik},
      year = {2018}
      journal = {ACS Central Science},
      number = {9},
      volume = {4},
      pages = {1134--1145}
      }


If you optimized **categorical variables**, please cite `this publication <https://aip.scitation.org/doi/full/10.1063/5.0048164>`_:

::

    @article{gryffin,
      title = {Gryffin: An algorithm for Bayesian optimization of categorical variables informed by expert knowledge},
      author = {Florian Häse and Matteo Aldeghi and Riley J. Hickman and Loïc M. Roch and Alán Aspuru-Guzik},
      year = {2021},
      journal = {Applied Physics Reviews},
      number = {8},
      pages = {031406}
      }

If you performed a **multi-objective optimization**, or used **periodic variables**, please cite
`this publication <https://pubs.rsc.org/en/content/articlelanding/2018/sc/c8sc02239a#!divAbstract>`_:

::

    @article{chimera,
      title = {Chimera: enabling hierarchy based multi-objective optimization for self-driving laboratories},
      author = {Florian Häse and Loïc M. Roch and Alán Aspuru-Guzik},
      year = {2018},
      journal = {Chemical Science},
      number = {9},
      pages = {7642--7655}
      }

If you performed an optimization with **known or unknown feasibility constraints**, or used ``genetic`` as the
optimization algorithm for the acquisition, please cite `this publication <https://arxiv.org/abs/2203.17241>`_:

::

    @article{gryffin_known_constraints,
      title={Bayesian optimization with known experimental and design constraints for chemistry applications},
      author={Hickman, Riley J. and Aldeghi, Matteo and Häse, Florian and Aspuru-Guzik, Alán},
      year={2022},
      journal = {arXiv:2203.17241 [math.OC]},
      }