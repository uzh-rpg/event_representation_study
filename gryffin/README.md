[![build](https://github.com/aspuru-guzik-group/gryffin/actions/workflows/continuous-integration.yml/badge.svg)](https://github.com/aspuru-guzik-group/gryffin/actions/workflows/continuous-integration.yml)
[![Documentation Status](https://readthedocs.org/projects/gryffin/badge/?version=latest)](http://gryffin.readthedocs.io/?badge=latest)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

Gryffin: Bayesian Optimization of Continuous and Categorical Variables
======================================================================

Welcome to **Gryffin**!

Designing functional molecules and advanced materials requires complex design choices: tuning
continuous process parameters such as temperatures or flow rates, while simultaneously selecting
catalysts or solvents. 

To date, the development of data-driven experiment planning strategies for
autonomous experimentation has largely focused on continuous process parameters despite the urge
to devise efficient strategies for the selection of categorical variables. Here, we introduce Gryffin,
a general purpose optimization framework for the autonomous selection of categorical variables
driven by expert knowledge.

## Features

* Gryffin extends the ideas of the [Phoenics](https://pubs.acs.org/doi/10.1021/acscentsci.8b00307) optimizer to categorical variables. Phoenics is a linear-scaling Bayesian optimizer for continuous spaces which uses a kernel regression surrogate. Gryffin extends this approach to categorical and mixed continuous-categorical spaces. 
* Gryffin is linear-scaling appraoch to Bayesian optimization, whose acquisition function natively supports batched optimization. Gryffin's acquisition function uses an intuitive sampling parameter to bias its behaviour between exploitation and exploration. 
* Gryffin is capable of leveraging expert knowledge in the form of physicochemical descriptors to enhance its optimization performance (static formulation). Also, Gryffin can refine the provided descriptors to further accelerate the optimization (dynamic formulation) and foster scientific understanding. 

## Use cases of Gryffin/Phoenics

* [Self-driving lab to optimize multicomponet organic photovoltaic systems](https://onlinelibrary.wiley.com/doi/full/10.1002/adma.201907801)
* [Self-driving laboratory for accelerated discovery of thin-film materials](https://www.science.org/doi/10.1126/sciadv.aaz8867)
* [Data-science driven autonomous process optimization](https://www.nature.com/articles/s42004-021-00550-x)
* [Self-driving platform for metal nanoparticle synthesis](https://onlinelibrary.wiley.com/doi/full/10.1002/adfm.202106725)
* [Optimization of photophyscial properties of organic dye laser molecules](https://pubs.acs.org/doi/10.1021/acscentsci.1c01002)


## Requirements

* Python version >= 3.7


## Installation

To install ``gryffin`` from [PyPI](https://pypi.org/project/gryffin/):

```console
$ pip install gryffin
```

To install ``gryffin`` from source:

``` console
$ git clone git@github.com:aspuru-guzik-group/gryffin.git
$ cd gryffin
$ pip install .
```

## Example Usage 


This is a minimalist example of Gryffin in action.


```python

    from gryffin import Gryffin
    import experiment

    # load config
    config = {
        "parameters": [
            {"name": "param_0", "type": "continuous", "low": 0.0, "high": 1.0},
        ],
        objectives: [
            {"name": "obj", "goal": "min"},
        ]
    }

    # initialize gryffin
    gryffin = Gryffin(
        config_dict=config
    )

    observations = [] 
    for iter in range(ITER_BUDGET):

        # query gryffin for new params
        params  = gryffin.recommend(observations=observations)

        # evaluate the proposed parameters
        merit = experiment.run(params)
        params['obj'] = merit

        observations.append(params)
```

## Documentation

Please refer to the [documentation](https://gryffin.readthedocs.io/en/latest/) website for:

* [Getting Started](https://gryffin.readthedocs.io/en/latest/getting_started.html)
* [Configuration](https://gryffin.readthedocs.io/en/latest/configuration.html)
* [Tutorials](https://gryffin.readthedocs.io/en/latest/tutorial.html)
* [API Reference](https://gryffin.readthedocs.io/en/latest/api_documentation.html)
* [CLI Reference](https://gryffin.readthedocs.io/en/latest/cli_documentation.html)


## Citation

If you found Gryffin useful, please include the relevant [citation](https://gryffin.readthedocs.io/en/latest/citation.html) in your work.

## License

[Apache License 2.0](https://choosealicense.com/licenses/apache-2.0/)






