# Models

During model developments multiple approaches have been tested. As a start simple models that are not hiearchical are located in the simple folder. These have been developed to test different model setups. The hierarchical folder holds the hieararchical implementations that will be used for the thesis.

Experimentation:
1. Dummy coding vs. index variables: Due to some confusion about interpretation of intercept and uncertainty in the estimates I also created a dummy coded version. Not sure which is best. Maybe they are even identical in performance.


Problems:
1. Mean estimates are fine, but the distributions are very wide (credible intervals only moves slightly from the prior)
* For the index variable models, this was fixed by doing sum-to-0 constraints on the parameters. Much better estimation. Probably this is linked to the confusion about the intercept, since it was not implemented as a reference level.
2. Warnings about some parameter samplings
* Probably not that problematic. But I found ways to fix it, by either choosing some constraints on the parameters or add a small constant. Maybe this is not the best for the sampling.